#!/usr/bin/env python3
"""Simulate EEF velocity servo with gamepad dx/dy/dz (no ROS).

Mirrors AgilexCobot.send_action_from_eef logic using RobotKinematics only,
and plots input deltas, filtered EE velocities, joint positions, and joint velocities.

Usage:
    python examples/teleop/eef_servo_gamepad_sim.py [--history 900] [--save out.png]

Close the plot window or press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.teleoperators.gamepad.teleop_gamepad import GamepadTeleop
from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpacemouseTeleop

logger = logging.getLogger("eef_servo_gamepad_sim")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF_PATH = REPO_ROOT / "examples/hil-serl/aloha_new_description/urdf/robot.urdf"

DEFAULT_ARM_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07]
DEFAULT_AGILEX_ARM_Q = [
    -0.00133514404296875,
    0.00438690185546875,
    0.034523963928222656,
    -0.053597450256347656,
    -0.00476837158203125,
    -0.00209808349609375,
    0.07,
]

# ROS / Agilex names -> joints in examples/hil-serl single-arm robot.urdf
ROS_TO_REPO_JOINT_ALIASES = {
    "right_joint0": "right_joint0",
    "right_joint1": "right_joint1",
    "right_joint2": "right_joint2",
    "right_joint3": "right_joint3",
    "right_joint4": "right_joint4",
    "right_joint5": "right_joint5",
    "right_joint6": "fr_joint7",
}

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except Exception:
    logger.error("matplotlib is required to display live plots")
    raise


@dataclass
class EefServoSimConfig:
    urdf_path: str = str(DEFAULT_URDF_PATH)
    ik_target_frame_name: str = "gripper_frame_link"
    ik_joint_names: tuple[str, ...] = (
        "right_joint0",
        "right_joint1",
        "right_joint2",
        "right_joint3",
        "right_joint4",
        "right_joint5",
        "fr_joint7",
    )
    eef_input_fps: float = 30.0
    eef_command_timeout_s: float = 0.2
    use_delta_filter: bool = True
    velocity_filter_cutoff_hz: float =5.0
    dls_damping: float = 0.1


def resolve_ik_joint_names(urdf_path: str, joint_names: tuple[str, ...]) -> tuple[str, ...]:
    """Map ROS joint names (e.g. right_joint*) to URDF joints when needed."""
    try:
        import placo  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "placo is required for kinematics. Install with: pip install 'lerobot[kinematics]'"
        ) from exc

    robot = placo.RobotWrapper(urdf_path)
    available = set(robot.joint_names())
    resolved: list[str] = []
    aliases_used: dict[str, str] = {}

    for name in joint_names:
        if name in available:
            resolved.append(name)
            continue
        alias = ROS_TO_REPO_JOINT_ALIASES.get(name)
        if alias and alias in available:
            aliases_used[name] = alias
            resolved.append(alias)
            continue
        raise ValueError(
            f"Joint '{name}' not found in URDF '{urdf_path}'. "
            f"Available: {sorted(available)}. "
            "Pass --urdf-path for your robot model, or use matching --ik-joint-names."
        )

    if aliases_used:
        logger.warning(
            "URDF '%s' uses different joint names; mapped ROS names: %s",
            urdf_path,
            aliases_used,
        )

    return tuple(resolved)


def agilex_sim_config() -> EefServoSimConfig:
    """Build sim config aligned with AgilexCobotConfig (no ROS)."""
    from lerobot.robots.agilex_cobot.config_agilex_cobot import AgilexCobotConfig

    robot_cfg = AgilexCobotConfig()
    return EefServoSimConfig(
        urdf_path=robot_cfg.urdf_path,
        ik_target_frame_name=robot_cfg.ik_target_frame_name,
        ik_joint_names=tuple(robot_cfg.ros_config.right_arm_joints),
        eef_input_fps=robot_cfg.eef_input_fps,
        eef_command_timeout_s=robot_cfg.eef_command_timeout_s,
    )


class EefServoSimulator:
    """ROS-free stand-in for send_action_from_eef velocity servo."""

    def __init__(self, config: EefServoSimConfig, initial_q: list[float]):
        self.config = config
        ik_joint_names = resolve_ik_joint_names(config.urdf_path, config.ik_joint_names)
        self.kinematics = RobotKinematics(
            urdf_path=config.urdf_path,
            target_frame_name=config.ik_target_frame_name,
            joint_names=list(ik_joint_names),
            use_rad=True,
        )
        self._q = np.array(initial_q, dtype=np.float64)
        self._filtered_vel = np.zeros(6, dtype=np.float64) if config.use_delta_filter else None
        self._integrated_q: Optional[np.ndarray] = None
        self._servo_joint_state_initialized = False
        self._last_call_eef_time: Optional[float] = None

    def step(self, ee_command: dict) -> dict:
        """One control step; returns logged state for plotting."""
        now = time.time()
        if self._last_call_eef_time is not None:
            if now - self._last_call_eef_time > self.config.eef_command_timeout_s:
                self._servo_joint_state_initialized = False
                self._integrated_q = None
                self._filtered_vel.fill(0.0)
                print("timeout, reset servo joint state")
        self._last_call_eef_time = now

        dt = 1.0 / self.config.eef_input_fps
        delta_x = ee_command.get("delta_x", 0.0)
        delta_y = ee_command.get("delta_y", 0.0)
        delta_z = ee_command.get("delta_z", 0.0)
        delta_roll = ee_command.get("delta_roll", 0.0)
        delta_pitch = ee_command.get("delta_pitch", 0.0)
        delta_yaw = ee_command.get("delta_yaw", 0.0)
        gripper = ee_command.get("gripper", 1.0)

        raw_vel = np.array(
            [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw],
            dtype=np.float64,
        )/dt

        if self.config.use_delta_filter:
            fc = self.config.velocity_filter_cutoff_hz
            alpha = np.clip(2.0 * np.pi * fc * dt, 0.0, 1.0)
            self._filtered_vel = (1.0 - alpha) * self._filtered_vel + alpha * raw_vel
            v_filt = self._filtered_vel
        else:
            v_filt = raw_vel

        q = self._q[:-1]
        if self._integrated_q is None or not self._servo_joint_state_initialized:
            self._integrated_q = q.copy()
            self._servo_joint_state_initialized = True

        j_pinv = self.kinematics.get_damped_pinv(self._integrated_q, damping=self.config.dls_damping)
        q_dot = j_pinv @ v_filt
  
        self._integrated_q = self._integrated_q + q_dot * dt
 
        q_target = np.zeros(7, dtype=np.float64)
        q_target[:-1] = self._integrated_q

        if gripper == 0.0:
            q_target[-1] = 0.0
        elif gripper == 1.0:
            q_target[-1] = self._q[-1]
        elif gripper == 2.0:
            q_target[-1] = 0.03
        else:
            q_target[-1] = self._q[-1]

        self._q = q_target

        return {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
            "vx": v_filt[0],
            "vy": v_filt[1],
            "vz": v_filt[2],
            "q_pos": q_target[:-1].copy(),
            "q_dot": q_dot.copy(),
        }


def run_sim(
    history: int = 900,
    save: Optional[str] = None,
    urdf_path: Optional[str] = None,
    eef_fps: float = 30.0,
    target_frame: Optional[str] = None,
    ik_joint_names: Optional[list[str]] = None,
    agilex: bool = False,
    initial_q: Optional[list[float]] = None,
):
    if agilex:
        sim_config = agilex_sim_config()
        if eef_fps != 30.0:
            sim_config.eef_input_fps = eef_fps
        arm_q = initial_q or DEFAULT_AGILEX_ARM_Q
    else:
        sim_config = EefServoSimConfig(eef_input_fps=eef_fps)
        arm_q = initial_q or DEFAULT_ARM_Q

    if urdf_path is not None:
        sim_config.urdf_path = urdf_path
    if target_frame is not None:
        sim_config.ik_target_frame_name = target_frame
    if ik_joint_names is not None:
        sim_config.ik_joint_names = tuple(ik_joint_names)

    sim = EefServoSimulator(sim_config, arm_q)
    dt = 1.0 / sim.config.eef_input_fps

    # teleop_config = GamepadTeleop.config_class()
    # teleop_config.use_gripper = False
    # teleop = GamepadTeleop(teleop_config)
    # teleop.connect()
    # if not teleop.is_connected:
    #     raise RuntimeError("GamepadTeleop failed to connect — plug in a gamepad and retry")


    config_cls = SpacemouseTeleop.config_class
    config = config_cls()

    teleop = SpacemouseTeleop(config)
    teleop.connect()

    if not teleop.is_connected():
        raise RuntimeError("SpacemouseTeleop failed to connect to device")

    buf_lock = threading.Lock()
    state_buffer: Deque[tuple[float, dict]] = deque(maxlen=history)
    stop_event = threading.Event()

    def control_loop():
        logger.info("EEF servo control loop started at %.1f Hz", sim.config.eef_input_fps)
        while not stop_event.is_set():
            try:
                action = teleop.get_action()
                result = sim.step(action)
            except Exception:
                logger.exception("Error in control loop; stopping")
                stop_event.set()
                break

            with buf_lock:
                state_buffer.append((time.time(), result))

            time.sleep(dt)

        logger.info("Control loop exiting")

    controller = threading.Thread(target=control_loop, daemon=True, name="EEFServoLoop")
    controller.start()

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    delta_labels = ["delta_x", "delta_y", "delta_z"]
    vel_labels = ["vx", "vy", "vz"]
    q_labels = [f"q{i}" for i in range(6)]
    qd_labels = [f"qd{i}" for i in range(6)]

    lines_delta = [axes[0].plot([], [], label=label)[0] for label in delta_labels]
    lines_vel = [axes[1].plot([], [], label=label)[0] for label in vel_labels]
    lines_q = [axes[2].plot([], [], label=label)[0] for label in q_labels]
    lines_qd = [axes[3].plot([], [], label=label)[0] for label in qd_labels]

    for ax, title in zip(
        axes,
        [
            "Gamepad deltas (m/step)",
            "Filtered EE velocity (m/s)",
            "Joint position (rad)",
            "Joint velocity (rad/s)",
        ],
    ):
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=3)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("EEF velocity servo smoothness (gamepad sim, no ROS)")
    start_time = time.time()

    def update(_frame):
        with buf_lock:
            data = list(state_buffer)
        if not data:
            return lines_delta + lines_vel + lines_q + lines_qd

        timestamps = np.array([t - start_time for t, _ in data], dtype=float)

        for i, label in enumerate(delta_labels):
            values = np.array([d[label] for _, d in data], dtype=float)
            lines_delta[i].set_data(timestamps, values)

        for i, label in enumerate(vel_labels):
            values = np.array([d[label] for _, d in data], dtype=float)
            lines_vel[i].set_data(timestamps, values)

        for i in range(6):
            values = np.array([d["q_pos"][i] for _, d in data], dtype=float)
            lines_q[i].set_data(timestamps, values)
            values = np.array([d["q_dot"][i] for _, d in data], dtype=float)
            lines_qd[i].set_data(timestamps, values)

        for ax in axes:
            ax.relim()
            ax.autoscale_view(True, True, True)

        if len(timestamps) > 0:
            x_min = max(0.0, timestamps[-1] - 15.0)
            for ax in axes:
                ax.set_xlim(x_min, timestamps[-1] + 0.1)

        return lines_delta + lines_vel + lines_q + lines_qd

    anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

    logger.info("Opening live plot (close window to stop)")
    try:
        plt.show()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    stop_event.set()
    controller.join(timeout=2.0)

    if save:
        try:
            fig.savefig(save, dpi=150)
            logger.info("Saved plot to %s", save)
        except Exception:
            logger.exception("Failed to save figure")

    try:
        teleop.disconnect()
    except Exception:
        logger.exception("Error disconnecting teleop")


def main():
    parser = argparse.ArgumentParser(description="Gamepad EEF servo simulation (no ROS)")
    parser.add_argument("--history", type=int, default=900, help="Max samples kept for plotting")
    parser.add_argument("--save", type=str, default=None, help="Optional PNG path when window closes")
    parser.add_argument("--urdf-path", type=str, default=None, help="Override robot URDF path")
    parser.add_argument(
        "--target-frame",
        type=str,
        default=None,
        help="IK target frame (default: gripper_frame_link)",
    )
    parser.add_argument(
        "--ik-joint-names",
        nargs=7,
        metavar="JOINT",
        default=None,
        help="Seven joint names: 6 arm joints + gripper joint (for RobotKinematics)",
    )
    parser.add_argument("--eef-fps", type=float, default=30.0, help="Control loop rate (Hz)")
    parser.add_argument(
        "--agilex",
        action="store_true",
        help="Use AgilexCobotConfig URDF/joint names (right_joint* -> repo URDF aliases)",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        run_sim(
            history=args.history,
            save=args.save,
            urdf_path=args.urdf_path,
            eef_fps=args.eef_fps,
            target_frame=args.target_frame,
            ik_joint_names=args.ik_joint_names,
            agilex=args.agilex,
        )
    except Exception:
        logger.exception("EEF servo gamepad simulation failed")


if __name__ == "__main__":
    main()