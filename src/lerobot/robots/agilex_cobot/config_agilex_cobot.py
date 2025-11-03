# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Dict, Any

from lerobot.cameras import CameraConfig
from ..config import RobotConfig


@RobotConfig.register_subclass("agilex_cobot")
@dataclass
class AgilexCobotConfig(RobotConfig):
    """
    Configuration class for ROS1-based dual arm robot with elevator.
    """
    
    # ROS2 node configuration
    node_name: str = "lerobot_agilex_cobot_robot_node"
    connection_timeout = 5.0  # seconds
    
    
    # meters or radians per control step
    arm_steps_length = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]  
    
    # Tag for external commands control
    # Set to True if you use an external commands system to control the robot,
    # such as the official teleoperation application: https://github.com/pollen-robotics/Reachy2Teleoperation
    # If True, robot.send_action() will not send commands to the robot.
    use_external_commands: bool = False
    
    # Robot parts
    # Set to False to not add the corresponding joints part to the robot list of joints.
    # By default, all parts are set to True.
    with_mobile_base: bool = False
    with_l_arm: bool = True
    with_r_arm: bool = True
    with_endpose: bool = True
    

  
    # Control command topics (for reading actions)
    left_arm_command_topic: str = "/master/joint_left"
    right_arm_command_topic: str = "/master/joint_right" 
    mobile_command_topic: str = "/cmd_vel"
    endpose_left_cmd_topic: str = "/pos_cmd_left"
    endpose_right_cmd_topic: str = "/pos_cmd_right"
    

    # Joint state topic (for reading robot state)
    left_joint_states_topic: str = "/puppet/joint_left"
    right_joint_states_topic: str = "/puppet/joint_right"
    mobile_base_state_topic: str = "/odom_raw"
    endpose_left_topic: str = "/puppet/end_pose_left"
    endpose_right_topic: str = "/puppet/end_pose_right"


    # Joint names configuration
    left_arm_joints: list[str] = field(default_factory=lambda: ['left_joint0', 'left_joint1', 'left_joint2', 'left_joint3', 'left_joint4', 'left_joint5', 'left_joint6'])
    right_arm_joints: list[str] = field(default_factory=lambda:['right_joint0', 'right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6'])
    mobile_base_joints: list[str] = field(default_factory=lambda: ['vx', 'vy', 'vtheta'])
    endpose_left_joints: list[str] = field(default_factory=lambda: ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'])
    endpose_right_joints: list[str] = field(default_factory=lambda: ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'])
   
   
    # Robot cameras
    # Set to True if you want to use the corresponding cameras in the observations.
    # By default, only the teleop cameras are used.
    with_left_camera: bool = True
    with_right_camera: bool = True
    with_front_camera: bool = True
    
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    cam_high_topic: str = "/camera_f/color/image_raw"
    cam_left_topic: str = "/camera_l/color/image_raw"
    cam_right_topic: str = "/camera_r/color/image_raw"

    use_depth_image:  bool = False
    img_front_depth_topic: str = "/camera_f/depth/image_raw"
    img_left_depth_topic: str = "/camera_l/depth/image_raw"
    img_right_depth_topic: str = "/camera_r/depth/image_raw"
    
    # Control rate
    control_rate: float = 50.0  # Hz

    def __post_init__(self):
        
        # Add cameras with same ip_address as the robot
        # if self.with_left_teleop_camera:
        #     self.cameras["teleop_left"] = Reachy2CameraConfig(
        #         name="teleop",
        #         image_type="left",
        #         ip_address=self.ip_address,
        #         fps=15,
        #         width=640,
        #         height=480,
        #         color_mode=ColorMode.RGB,
        #     )
        # if self.with_right_teleop_camera:
        #     self.cameras["teleop_right"] = Reachy2CameraConfig(
        #         name="teleop",
        #         image_type="right",
        #         ip_address=self.ip_address,
        #         fps=15,
        #         width=640,
        #         height=480,
        #         color_mode=ColorMode.RGB,
        #     )
        # if self.with_torso_camera:
        #     self.cameras["torso_rgb"] = Reachy2CameraConfig(
        #         name="depth",
        #         image_type="rgb",
        #         ip_address=self.ip_address,
        #         fps=15,
        #         width=640,
        #         height=480,
        #         color_mode=ColorMode.RGB,
        #     )
        
        
        super().__post_init__()
        
        if not (
            self.with_mobile_base
            or self.with_l_arm
            or self.with_r_arm
        ):
            raise ValueError(
                "No Cobot magic teleop part used.\n"
                "At least one part of the robot must be set to True "
                "(with_mobile_base, with_l_arm, with_r_arm)"
            )