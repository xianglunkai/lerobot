import json
import logging
import time

import cv2
import numpy as np

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_agilex_cobot_zmq import AgilexCobotZmqConfig
from .protocol import decode_observation, encode_command

logger = logging.getLogger(__name__)


class AgilexCobotZmq(Robot):
    """
    AgileX cobot client that talks to a ROS1 bridge over ZMQ.

    The class name reported to policies stays ``agilex_cobot`` so a model trained
    against that embodiment still matches. Select this implementation with
    ``--robot.type=agilex_cobot_zmq``.
    """

    config_class = AgilexCobotZmqConfig
    name = "agilex_cobot"
    # Joint targets are positions. Re-sending the last target holds the arms,
    # which remote inference requires while a chunk is in flight.
    supports_position_hold = True

    def __init__(self, config: AgilexCobotZmqConfig):
        super().__init__(config)
        self.config = config
        # Images arrive over ZMQ. These entries are not local cameras; dataset
        # setup uses len(self.cameras) to size the image writers.
        self.cameras = {name: None for name in self.camera_names}
        self._is_connected = False
        self._zmq = None
        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None
        self.zmq_rpc_socket = None

        self._latest_joints: dict[str, float] = {}
        self._latest_endpose: dict[str, np.ndarray] = {}
        self._latest_images: dict[str, np.ndarray] = {}
        self._last_obs_time = 0.0
        self._stale_warned = False

        self._last_call_eef_time = None
        self._filtered_vel = np.zeros(6, dtype=np.float64)
        self._integrated_q = None
        self._servo_joint_state_initialized = False
        self._kinematics = None

    @property
    def left_arm_joints(self) -> list[str]:
        return list(self.config.left_arm_joints)

    @property
    def right_arm_joints(self) -> list[str]:
        return list(self.config.right_arm_joints)

    @property
    def all_joints(self) -> list[str]:
        joints = []
        if self.config.with_l_arm:
            joints.extend(self.left_arm_joints)
        if self.config.with_r_arm:
            joints.extend(self.right_arm_joints)
        return joints

    @property
    def camera_names(self) -> list[str]:
        names = []
        if self.config.with_front_camera:
            names.append("high")
        if self.config.with_left_camera:
            names.append("left")
        if self.config.with_right_camera:
            names.append("right")
        return names

    @property
    def motors_features(self) -> dict[str, type]:
        features = {f"{joint}.pos": float for joint in self.all_joints}
        if self.config.with_mobile_base:
            features.update({f"{joint}.pos": float for joint in self.config.mobile_base_joints})
        return features

    @property
    def camera_features(self) -> dict[str, tuple]:
        shape = (self.config.image_height, self.config.image_width, 3)
        return {name: shape for name in self.camera_names}

    @property
    def observation_features(self) -> dict:
        return {**self.motors_features, **self.camera_features}

    @property
    def action_features(self) -> dict:
        return self.motors_features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """Home both arms through the bridge reset RPC before the rollout captures a pose."""
        logger.info("Sending client reset to the AgileX ZMQ bridge")
        self.reset_to_default_positions()

    def configure(self) -> None:
        return None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        import zmq

        self._zmq = zmq
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_cmd_socket.setsockopt(zmq.SNDTIMEO, self.config.command_send_timeout_ms)
        self.zmq_cmd_socket.connect(f"tcp://{self.config.remote_ip}:{self.config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_observation_socket.connect(
            f"tcp://{self.config.remote_ip}:{self.config.port_zmq_observations}"
        )

        self.zmq_rpc_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_rpc_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_rpc_socket.setsockopt(zmq.RCVTIMEO, self.config.rpc_timeout_ms)
        self.zmq_rpc_socket.setsockopt(zmq.SNDTIMEO, self.config.rpc_timeout_ms)
        self.zmq_rpc_socket.connect(f"tcp://{self.config.remote_ip}:{self.config.port_zmq_rpc}")

        deadline = time.time() + self.config.connect_timeout_s
        while time.time() < deadline:
            if self._poll_observation() is not None:
                self._is_connected = True
                logger.info("Connected to AgileX ZMQ bridge at %s", self.config.remote_ip)
                if calibrate:
                    self.calibrate()
                return
            time.sleep(0.05)
        self.disconnect()
        raise DeviceNotConnectedError(
            f"Timed out waiting for the AgileX ZMQ bridge at {self.config.remote_ip}. "
            "Start src/lerobot/robots/agilex_cobot_zmq/run_agilex_cobot_zmq_bridge.sh in the ROS1 Python 3.8 environment first."
        )

    def disconnect(self) -> None:
        self._is_connected = False
        for socket in (self.zmq_cmd_socket, self.zmq_observation_socket, self.zmq_rpc_socket):
            if socket is not None:
                socket.close(linger=0)
        if self.zmq_context is not None:
            self.zmq_context.term()
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None
        self.zmq_rpc_socket = None
        self.zmq_context = None
        self._last_call_eef_time = None
        self._servo_joint_state_initialized = False
        self._integrated_q = None
        logger.info("Disconnected from AgileX ZMQ bridge")

    @check_if_not_connected
    def get_observation(self) -> dict:
        self._poll_observation()
        age = time.time() - self._last_obs_time
        if self._last_obs_time == 0.0:
            raise DeviceNotConnectedError("No observation has arrived from the ZMQ bridge.")
        if age > self.config.obs_stale_timeout_s:
            if not self._stale_warned:
                logger.warning("AgileX observation is %.2fs old", age)
                self._stale_warned = True
        else:
            self._stale_warned = False

        observation = {f"{joint}.pos": float(self._latest_joints.get(joint, 0.0)) for joint in self.all_joints}
        if self.config.with_mobile_base:
            for joint in self.config.mobile_base_joints:
                observation[f"{joint}.pos"] = float(self._latest_joints.get(joint, 0.0))
        placeholder = np.zeros((self.config.image_height, self.config.image_width, 3), dtype=np.uint8)
        for name in self.camera_names:
            observation[name] = self._latest_images.get(name, placeholder)
        return observation

    @check_if_not_connected
    def send_action(self, action: dict) -> dict:
        if self.config.use_external_commands:
            return action
        if "delta_x" in action and "delta_y" in action and "delta_z" in action:
            return self.send_action_from_eef_vel(action)

        payload = {
            "type": "arm",
            "left": self._arm_targets(action, self.left_arm_joints) if self.config.with_l_arm else None,
            "right": self._arm_targets(action, self.right_arm_joints) if self.config.with_r_arm else None,
        }
        if self.config.with_mobile_base:
            payload["base"] = [
                float(action.get(f"{joint}.pos", 0.0)) for joint in self.config.mobile_base_joints
            ]
        self._send_command(payload)
        return action

    @check_if_not_connected
    def reset_to_default_positions(self) -> None:
        self._rpc({"type": "reset"})

    @check_if_not_connected
    def control_robot_with_continuous(self, left_pos_cmd=None, right_pos_cmd=None) -> None:
        self._rpc(
            {
                "type": "continuous",
                "left": None if left_pos_cmd is None else [float(value) for value in left_pos_cmd],
                "right": None if right_pos_cmd is None else [float(value) for value in right_pos_cmd],
            }
        )

    def get_eef_pose(self) -> dict[str, np.ndarray]:
        if not self.is_connected:
            return {"left": np.zeros(7), "right": np.zeros(7)}
        self.get_observation()
        return {
            "left": np.asarray(self._latest_endpose.get("left", np.zeros(7)), dtype=np.float64),
            "right": np.asarray(self._latest_endpose.get("right", np.zeros(7)), dtype=np.float64),
        }

    def send_action_from_eef(self, ee_command: dict) -> dict:
        import transforms3d as t3d

        now = time.time()
        if self._last_call_eef_time is not None:
            if now - self._last_call_eef_time > self.config.eef_command_timeout_s:
                self._servo_joint_state_initialized = False
        self._last_call_eef_time = now

        q_raw = self._current_arm_position("right")
        if q_raw is None:
            logger.error("Cannot read the right arm for end-effector control")
            return ee_command
        q = np.array(q_raw[:-1], dtype=np.float64)
        kinematics = self._ensure_kinematics()
        current_ee_pose = kinematics.forward_kinematics(q)

        delta_p = np.array(
            [
                ee_command.get("delta_x", 0.0),
                ee_command.get("delta_y", 0.0),
                ee_command.get("delta_z", 0.0),
            ],
            dtype=np.float64,
        )
        r_abs = t3d.euler.euler2mat(
            ee_command.get("delta_roll", 0.0),
            ee_command.get("delta_pitch", 0.0),
            ee_command.get("delta_yaw", 0.0),
        )
        desired = np.eye(4, dtype=float)
        desired[:3, :3] = current_ee_pose[:3, :3] @ r_abs
        desired[:3, 3] = current_ee_pose[:3, 3] + delta_p
        desired_q = kinematics.inverse_kinematics(q, desired)
        return self._publish_right_arm_target(desired_q, q_raw, ee_command.get("gripper", 1.0))

    def send_action_from_eef_vel(self, ee_command: dict) -> dict:
        now = time.time()
        if self._last_call_eef_time is not None:
            if now - self._last_call_eef_time > self.config.eef_command_timeout_s:
                self._servo_joint_state_initialized = False
                self._integrated_q = None
                self._filtered_vel.fill(0.0)
        self._last_call_eef_time = now

        dt = 1.0 / self.config.eef_input_fps
        raw_vel = np.array(
            [
                ee_command.get("delta_x", 0.0),
                ee_command.get("delta_y", 0.0),
                ee_command.get("delta_z", 0.0),
                ee_command.get("delta_roll", 0.0),
                ee_command.get("delta_pitch", 0.0),
                ee_command.get("delta_yaw", 0.0),
            ],
            dtype=np.float64,
        )
        alpha = float(np.clip(2.0 * np.pi * self.config.velocity_filter_cutoff_hz * dt, 0.0, 1.0))
        self._filtered_vel = (1.0 - alpha) * self._filtered_vel + alpha * raw_vel

        q_raw = self._current_arm_position("right")
        if q_raw is None:
            logger.error("Cannot read the right arm for end-effector velocity control")
            return ee_command
        q = np.array(q_raw[:-1], dtype=np.float64)
        kinematics = self._ensure_kinematics()
        if self._integrated_q is None or not self._servo_joint_state_initialized:
            self._integrated_q = q.copy() if kinematics.use_rad else np.deg2rad(q.copy())
            self._servo_joint_state_initialized = True

        j_pinv = kinematics.get_damped_pinv(self._integrated_q, damping=getattr(self.config, "dls_damping", 0.1))
        self._integrated_q = self._integrated_q + (j_pinv @ self._filtered_vel) * dt
        desired_q = self._integrated_q if kinematics.use_rad else np.rad2deg(self._integrated_q)
        return self._publish_right_arm_target(desired_q, q_raw, ee_command.get("gripper", 1.0))

    def _publish_right_arm_target(self, desired_q: np.ndarray, q_raw: list[float], gripper: float) -> dict:
        q_target = np.zeros(7)
        q_target[:-1] = desired_q
        if gripper == 0.0:
            q_target[-1] = 0.0
        elif gripper == 2.0:
            q_target[-1] = 0.03
        else:
            q_target[-1] = q_raw[-1]

        if self.config.with_r_arm:
            self._send_command({"type": "arm", "left": None, "right": q_target.tolist()})

        action = {}
        for joint in self.left_arm_joints:
            if self.config.with_l_arm:
                action[f"{joint}.pos"] = float(self._latest_joints.get(joint, 0.0))
        for index, joint in enumerate(self.right_arm_joints):
            action[f"{joint}.pos"] = float(q_target[index])
        return action

    def _ensure_kinematics(self):
        if self._kinematics is None:
            from lerobot.model.kinematics import RobotKinematics

            self._kinematics = RobotKinematics(
                urdf_path=self.config.urdf_path,
                target_frame_name=self.config.ik_target_frame_name,
                joint_names=self.config.ik_joint_names,
                use_rad=True,
            )
        return self._kinematics

    def _current_arm_position(self, side: str) -> list[float] | None:
        self.get_observation()
        names = self.right_arm_joints if side == "right" else self.left_arm_joints
        if any(name not in self._latest_joints for name in names):
            return None
        return [float(self._latest_joints[name]) for name in names]

    def _arm_targets(self, action: dict, joints: list[str]) -> list[float]:
        return [float(action.get(f"{joint}.pos", 0.0)) for joint in joints]

    def _send_command(self, payload: dict) -> None:
        try:
            self.zmq_cmd_socket.send_string(encode_command(payload))
        except self._zmq.Again:
            logger.warning("Dropped an arm command because the ZMQ bridge is not reading")

    def _rpc(self, payload: dict) -> dict:
        try:
            self.zmq_rpc_socket.send_string(json.dumps(payload))
            reply = json.loads(self.zmq_rpc_socket.recv_string())
        except self._zmq.Again as exc:
            self._reopen_rpc()
            raise TimeoutError("AgileX ZMQ bridge did not answer the reset command") from exc
        if not reply.get("ok", False):
            raise RuntimeError(reply.get("error", "AgileX ZMQ bridge rejected the command"))
        return reply

    def _reopen_rpc(self) -> None:
        zmq = self._zmq
        if self.zmq_rpc_socket is not None:
            self.zmq_rpc_socket.close(linger=0)
        self.zmq_rpc_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_rpc_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_rpc_socket.setsockopt(zmq.RCVTIMEO, self.config.rpc_timeout_ms)
        self.zmq_rpc_socket.setsockopt(zmq.SNDTIMEO, self.config.rpc_timeout_ms)
        self.zmq_rpc_socket.connect(f"tcp://{self.config.remote_ip}:{self.config.port_zmq_rpc}")

    def _poll_observation(self) -> dict | None:
        zmq = self._zmq
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        events = dict(poller.poll(self.config.polling_timeout_ms))
        if self.zmq_observation_socket not in events:
            return None
        latest = None
        while True:
            try:
                latest = self.zmq_observation_socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if latest is None:
            return None
        try:
            header = decode_observation(latest)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Ignored a bad observation packet: %s", exc)
            return None
        self._latest_joints = {str(name): float(value) for name, value in header.get("joints", {}).items()}
        self._latest_endpose = {
            side: np.asarray(pose, dtype=np.float64) for side, pose in header.get("endpose", {}).items()
        }
        images = {}
        for name, blob in header.get("images", {}).items():
            frame = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                images[name] = frame
        if images:
            self._latest_images = images
        self._last_obs_time = time.time()
        return header
