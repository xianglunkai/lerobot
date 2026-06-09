# agilex_cobot_robot.py
import logging
from typing import Dict, Any, Optional

import numpy as np
import threading
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_agilex_cobot import AgilexCobotConfig
from lerobot.utils.agilex_cobot_ros_manager import AgilexCobotROSManager

import time
logger = logging.getLogger(__name__)


# import RobotKinematics here to avoid circular import
from lerobot.model.kinematics import RobotKinematics
import transforms3d as t3d


class AgilexCobotBase(Robot):
    """Base class for AgilexCobot robot with common functionality."""
    
    config_class = AgilexCobotConfig
    name = "agilex_cobot"
    
    def __init__(self, config: AgilexCobotConfig):
        super().__init__(config)
        self.config = config
        
        # Connection state
        self._connected = False
        self._connection_lock = threading.Lock()
        
        # Build joint lists
        self.all_joints =  []
        if config.ros_config.with_l_arm:
            self.all_joints += config.ros_config.left_arm_joints
            
        if config.ros_config.with_r_arm:
            self.all_joints += config.ros_config.right_arm_joints
        
        self.robot_base = (
            config.ros_config.mobile_base_joints if config.ros_config.with_mobile_base else []
        )
        
        # ROS manager (will be initialized in connect)
        self.ros_manager: Optional[AgilexCobotROSManager] = None
        
        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)
        
        
        self.kinematics = RobotKinematics(
            urdf_path=self.config.urdf_path, 
            target_frame_name=self.config.ik_target_frame_name, 
            joint_names=self.config.ik_joint_names,
            use_rad=True,
        )
        
        logger.info(f"AgilexCobot robot initialized with {len(self.all_joints)} joints")
    
    @property
    def _joint_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.all_joints}
    
    @property
    def _robot_base_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.robot_base}
    
    @property
    def motors_features(self) -> dict[str, type]:
        if self.config.ros_config.with_mobile_base:
            return {
                **dict.fromkeys(self._joint_features.keys(), float),
                **dict.fromkeys(self._robot_base_features.keys(), float),
            }
        else:
            return dict.fromkeys(self._joint_features.keys(), float)
    
    @property
    def camera_features(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) 
            for cam in self.config.cameras
        }
    
    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self.motors_features, **self.camera_features}

    @property
    def action_features(self) -> dict[str, type]:
        return self.motors_features
    
    def calibrate(self) -> None:
        """Execute robot calibration procedure."""
        logger.info("Starting calibration procedure...")
        # Add calibration logic here
        logger.info("Calibration completed.")

    def configure(self, config_dict: Dict[str, Any]) -> None:
        """Configure robot parameters."""
        logger.info(f"Configuring robot with: {config_dict}")
        # Add configuration logic here

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return getattr(self, '_calibration_status', False)


class AgilexCobot(AgilexCobotBase):
    """
    AgilexCobot robot using singleton ROS manager.
    """
    
    def __init__(self, config: AgilexCobotConfig):
        super().__init__(config)
        self._calibration_status = False
        self._last_call_eef_time = None

        # MoveIt2-like online smoothing (Ruckig). Fallback is direct target command.
        self._ruckig_otg = None
        self._ruckig_input = None
        self._ruckig_output = None
        self._ruckig_result_working = None
        self._ruckig_result_finished = None
        self._ruckig_active = False
        self._servo_joint_state_initialized = False
        self._fallback_qd = np.zeros(6, dtype=np.float64)
        self._fallback_qdd = np.zeros(6, dtype=np.float64)

        try:
            from ruckig import InputParameter, OutputParameter, Result, Ruckig  # type: ignore[import-not-found]

            dof = 6
            self._ruckig_otg = Ruckig(dof, 1.0 / max(self.config.eef_input_fps, 1.0))
            self._ruckig_input = InputParameter(dof)
            self._ruckig_output = OutputParameter(dof)
            self._ruckig_result_working = Result.Working
            self._ruckig_result_finished = Result.Finished
            self._ruckig_active = True
        except Exception as exc:
            logger.warning("Ruckig unavailable, using internal rate-limited smoother: %s", exc)

    
    @property
    def is_connected(self) -> bool:
        """Check if robot is connected via ROS."""
        return self.ros_manager is not None and self.ros_manager.is_connected()
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect to ROS via the singleton ROS manager."""
        with self._connection_lock:
            if self._connected:
                logger.warning("Already connected")
                return
            
            try:
                # Initialize ROS manager (singleton)
                # Note: We need to convert our teleop config to ROS manager config
                self.ros_manager = AgilexCobotROSManager(self.config.ros_config)
                
                # Wait for initial state updates
                logger.info("Waiting for initial state updates...")
                start_time = time.time()
                timeout = getattr(self.config.ros_config, 'connection_timeout', 5.0)
                
                while not self.is_connected and (time.time() - start_time) < timeout:
                    time.sleep(0.1)
                
                if not self.is_connected:
                    raise ConnectionError("Failed to receive initial state updates within timeout")
                
                self._connected = True
                logger.info(f"{self.name} connected successfully")
                
                if calibrate:
                    self.calibrate()
                
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                self._connected = False
                raise
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current robot observation."""
        if not self.is_connected:
            # Return simulated observation for training
            return self._get_simulated_observation()
        
        # Get observation from ROS manager
        joint_states, camera_images, endpose = self.ros_manager.get_synchronized_observation()
        
        # Build observation dictionary
        observation = {}
        
        # Add joint positions
        for joint in self.all_joints:
            key = f"{joint}.pos"
            observation[key] = joint_states.get(joint, 0.0)
            
        # Add mobile base data
        if self.config.ros_config.with_mobile_base:
            for joint in self.robot_base:
                key = f"{joint}.pos"
                observation[key] = joint_states.get(joint, 0.0)
            
        # Add camera images
        for cam_name in self.cameras:
            observation[cam_name] = camera_images.get(cam_name, np.zeros(
                (self.config.cameras[cam_name].height, 
                 self.config.cameras[cam_name].width, 3), 
                dtype=np.uint8
            ))
        
        return observation
    
    def _get_simulated_observation(self) -> Dict[str, Any]:
        """Get simulated observation for training."""
        observation = {f"{joint}.pos": 0.0 for joint in self.all_joints}
        
        if self.config.ros_config.with_mobile_base:
            observation.update({
                "vx.pos": 0.0,
                "vy.pos": 0.0,
                "vtheta.pos": 0.0,
            })
        
        # Add simulated camera data
        for cam_name in ['high', 'left', 'right']:
            if hasattr(self.config, f'cam_{cam_name}_topic'):
                observation[f'camera_{cam_name}'] = np.zeros((480, 640, 3), dtype=np.uint8)
                
        return observation
    
    def get_eef_pose(self) -> Dict[str, np.ndarray]:
        """Get end-effector poses."""
        if not self.is_connected:
            return {
                "left": np.zeros((7,)),
                "right": np.zeros((7,)),
            }
        
        endpose = self.ros_manager.get_end_effector_poses()
        return endpose
    
    def send_action(self, action: Dict[str, float]) -> Dict[str, float]:
        """Send action commands to robot."""
        if not self.is_connected:
            return action
        
        if self.config.use_external_commands:
            return action
    
        # check that action is eef position
        if (
                "delta_x" in action
            and "delta_y" in action
            and "delta_z" in action
        ):
            return self.send_action_from_eef(action)
        
        # Extract arm commands
        left_arm_positions = [
            action.get(f"{joint}.pos", 0.0) 
            for joint in self.config.ros_config.left_arm_joints
        ]
        right_arm_positions = [
            action.get(f"{joint}.pos", 0.0) 
            for joint in self.config.ros_config.right_arm_joints
        ]
        
        # Extract mobile base commands
        vel_cmd = []
        if self.config.ros_config.with_mobile_base:
            vel_cmd = [
                action.get(f"{joint}.pos", 0.0) 
                for joint in self.robot_base
            ]
        
        # Publish commands via ROS manager
        if self.config.ros_config.with_l_arm:
            self.ros_manager.publish_left_arm_command(left_arm_positions)
        if self.config.ros_config.with_r_arm:
            self.ros_manager.publish_right_arm_command(right_arm_positions)
        if self.config.ros_config.with_mobile_base:
            self.ros_manager.publish_mobile_base_command(vel_cmd)
        
        return action
    
    
    def send_action_from_eef(self, ee_command: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Send action commands to robot based on end-effector command."""
        # ---- Timeout handling: re‑initialize servo state if too much time elapsed ----
        now = time.time()
        if self._last_call_eef_time is not None:
            if now - self._last_call_eef_time > self.config.eef_command_timeout_s:
                self._servo_joint_state_initialized = False
        self._last_call_eef_time = now

        # Extract delta commands
        delta_x = ee_command.get("delta_x", 0.0)
        delta_y = ee_command.get("delta_y", 0.0)
        delta_z = ee_command.get("delta_z", 0.0)
        delta_roll = ee_command.get("delta_roll", 0.0)
        delta_pitch = ee_command.get("delta_pitch", 0.0)
        delta_yaw = ee_command.get("delta_yaw", 0.0)
        gripper = ee_command.get("gripper", 1.0) # default to stay

        # get current joint positions (right arm only for now)
        q_raw = self.ros_manager._get_current_arm_position('right')
        q = np.array(q_raw[:-1], dtype=np.float64)  # exclude gripper joint

        # compute current end-effector pose
        current_ee_pose = self.kinematics.forward_kinematics(q)

        # compute desired end-effector pose
        ref = current_ee_pose.copy()
        delta_p = np.array([delta_x, delta_y, delta_z], dtype=np.float64)
        r_abs = t3d.euler.euler2mat(delta_roll, delta_pitch, delta_yaw)
        desired = np.eye(4, dtype=float)
        desired[:3, :3] = ref[:3, :3] @ r_abs
        desired[:3, 3] = ref[:3, 3] + delta_p

        # compute IK to get desired joint positions
        desired_q = self.kinematics.inverse_kinematics(q, desired)

        q_target = np.zeros(7)
        # todo: remaining work to debug 
        # q_target[:-1] = self._maybe_apply_ruckig(q, desired_q, dt=self.config.eef_input_fps)
        q_target[:-1] = desired_q

        # add gripper command to q_target if applicable
        if gripper == 0.0:      # close
            q_target[-1] = 0.0
        elif gripper == 1.0:    # stay
            q_target[-1] = q_raw[-1]
        elif gripper == 2.0:    # open
            q_target[-1] = 0.03
        else:
            q_target[-1] = q_raw[-1]

        # Publish commands via ROS manager
        if self.config.ros_config.with_r_arm:
            self.ros_manager.publish_right_arm_command(q_target.tolist())

        action = {}
        for i, name in enumerate(self.motors_features.keys()):
            action[name] = float(q_target[i])

        return action

    def _apply_internal_smoother(self, q_current: np.ndarray, q_target: np.ndarray, dt: float) -> np.ndarray:
        """Fallback smoother with velocity/acceleration/jerk limits.

        This is used when ruckig is not installed, to keep joint commands continuous.
        """
        dt = max(float(dt), 1e-3)
        v_lim = np.asarray(self.config.joint_velocity_limits, dtype=np.float64)
        a_lim = np.asarray(self.config.joint_acceleration_limits, dtype=np.float64)
        j_lim = np.asarray(self.config.joint_jerk_limits, dtype=np.float64)

        # Desired velocity to move toward target in current control step.
        v_des = np.clip((q_target - q_current) / dt, -v_lim, v_lim)
        a_des = np.clip((v_des - self._fallback_qd) / dt, -a_lim, a_lim)

        # Jerk-limit the acceleration update.
        da = a_des - self._fallback_qdd
        da_lim = j_lim * dt
        a_cmd = self._fallback_qdd + np.clip(da, -da_lim, da_lim)

        # Integrate with limits.
        v_cmd = self._fallback_qd + a_cmd * dt
        v_cmd = np.clip(v_cmd, -v_lim, v_lim)
        q_cmd = q_current + v_cmd * dt

        self._fallback_qdd = a_cmd
        self._fallback_qd = v_cmd
        return q_cmd

    def _maybe_apply_ruckig(self, q_current: np.ndarray, q_target: np.ndarray, dt: float) -> np.ndarray:
        """Apply Ruckig online trajectory generation if available."""
        if not self._ruckig_active:
            print("1. rucking is not active,use internal smoother")
            return self._apply_internal_smoother(q_current, q_target, dt)
        if self._ruckig_input is None or self._ruckig_output is None or self._ruckig_otg is None:
            print("2. rucking is not active,use internal smoother")
            return self._apply_internal_smoother(q_current, q_target, dt)

        if not self._servo_joint_state_initialized:
            self._ruckig_input.current_position = q_current.tolist()
            self._ruckig_input.current_velocity = [0.0] * 6
            self._ruckig_input.current_acceleration = [0.0] * 6
            self._servo_joint_state_initialized = True

        self._ruckig_input.target_position = q_target.tolist()
        self._ruckig_input.target_velocity = [0.0] * 6
        self._ruckig_input.target_acceleration = [0.0] * 6
        self._ruckig_input.max_velocity = [float(v) for v in self.config.joint_velocity_limits]
        self._ruckig_input.max_acceleration = [float(a) for a in self.config.joint_acceleration_limits]
        self._ruckig_input.max_jerk = [float(j) for j in self.config.joint_jerk_limits]

        result = self._ruckig_otg.update(self._ruckig_input, self._ruckig_output)
        if result in (self._ruckig_result_working, self._ruckig_result_finished):
            q_cmd = np.array(self._ruckig_output.new_position, dtype=np.float64)
            self._ruckig_output.pass_to_input(self._ruckig_input)
            return q_cmd

        print("3. rucking is not active,use internal smoother")
        return self._apply_internal_smoother(q_current, q_target, dt)

    def reset_to_default_positions(self) -> None:
        """Reset robot to default positions."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Define default positions
        reset_position_left = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, 
                              -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 0.07] 
        reset_position_right = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, 
                               -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 0.07]
        
        # Use continuous publishing for smooth reset
        self.ros_manager.publish_continuous_arm_commands(
            reset_position_left, reset_position_right
        )
        
        if self.config.ros_config.with_mobile_base:
            self.ros_manager.publish_mobile_base_command([0,0])
        
        logger.info(f"{self} reset to default positions")
        
    def control_robot_with_continuous(self, left_pos_cmd=None, right_pos_cmd=None):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
                # Define default positions
        # Use continuous publishing for smooth reset
        self.ros_manager.publish_continuous_arm_commands(
            left_pos_cmd, right_pos_cmd
        )
        
        logger.info(f"{self} reset to default positions")
    def disconnect(self) -> None:
        """Disconnect from ROS."""
        with self._connection_lock:
            if not self._connected:
                return
            
            # Note: ROS manager is singleton, so we don't shut it down here
            # It will be shut down when the program exits or when explicitly called
            self.ros_manager = None
            self._connected = False
            self._last_call_eef_time = None
            self._servo_joint_state_initialized = False
            self._fallback_qd[:] = 0.0
            self._fallback_qdd[:] = 0.0
            logger.info(f"{self.name} disconnected")
            
        