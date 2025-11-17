import time
from dataclasses import dataclass

import draccus
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    agilex_cobot,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    gamepad,
    make_teleoperator_from_config,
    agilex_cobot_teleop,
)

from lerobot.configs import parser
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.import_utils import register_third_party_devices

@dataclass
class FindJointLimitsConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second. By default, no limit.
    teleop_time_s: float = 3
    # Display all cameras on screen
    display_data: bool = True
    urdf_path: str = ""
    target_frame_name: str = ""
    joint_names: list[str] = ""


@parser.wrap()
def find_joint_and_ee_bounds(cfg: FindJointLimitsConfig):
    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop.connect()
    
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    time.sleep(3)

    start_episode_t = time.perf_counter()

    kinematics = RobotKinematics(cfg.urdf_path, cfg.target_frame_name, joint_names=cfg.joint_names, use_rad=True)

    # Initialize min/max values
    observation = robot.get_observation()
    print(f"observation.keys() : {observation.keys()}")
    print(f"action.key(): {robot.action_features.keys()}")
    
    joint_positions = np.array([observation[f"{key}"] for key in robot.action_features.keys()])
    joint_positions = joint_positions[:6]
    print(f"joint_positions : {joint_positions}")
    
    ee_pos = kinematics.forward_kinematics(joint_positions)[:3, 3]
    print(f"ee_pos: {ee_pos}")

    max_pos = joint_positions.copy()
    min_pos = joint_positions.copy()
    max_ee = ee_pos.copy()
    min_ee = ee_pos.copy()

    while True:
        action = teleop.get_action()
        # print(f"action : {action}")
        
        robot.send_action(action)

        observation = robot.get_observation()
        
        joint_positions = np.array([observation[f"{key}"] for key in robot.action_features.keys()])
        joint_positions = joint_positions[:6]

        ee_pos = kinematics.forward_kinematics(joint_positions)[:3, 3]
        
        if robot.robot_type == "agilex_cobot":
            ee_pos_from_ros = robot.get_eef_pose()
            print(f"ee_pos_from_ros[left] : {ee_pos_from_ros['left'][:3]}\n ee_pose_from_fk: {ee_pos}")

        # Skip initial warmup period
        if (time.perf_counter() - start_episode_t) < 5:
            busy_wait(0.01)
            continue

        # Update min/max values
        max_ee = np.maximum(max_ee, ee_pos)
        min_ee = np.minimum(min_ee, ee_pos)
        max_pos = np.maximum(max_pos, joint_positions)
        min_pos = np.minimum(min_pos, joint_positions)

        if time.perf_counter() - start_episode_t > cfg.teleop_time_s:
            print(f"Max ee position {np.round(max_ee, 4).tolist()}")
            print(f"Min ee position {np.round(min_ee, 4).tolist()}")
            print(f"Max joint pos position {np.round(max_pos, 4).tolist()}")
            print(f"Min joint pos position {np.round(min_pos, 4).tolist()}")
            break

        busy_wait(0.01)


def main():
    register_third_party_devices()
    find_joint_and_ee_bounds()


if __name__ == "__main__":
    main()
