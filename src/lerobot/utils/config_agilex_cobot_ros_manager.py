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

@dataclass
class AgilexCobotROSManagerConfig:
    """
    Configuration class for ROS1-based dual arm robot with elevator.
    """
    
    # ROS2 node configuration
    node_name: str = "lerobot_agilex_cobot_ros_node"
    connection_timeout = 5.0  # seconds
    
    # meters or radians per control step
    arm_steps_length = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]  
    use_eef_control: bool = False  # If True, use end-effector control for arms
    
    # Robot parts
    # Set to False to not add the corresponding joints part to the robot list of joints.
    # By default, all parts are set to True.
    with_mobile_base: bool = False
    with_l_arm: bool = True
    with_r_arm: bool = True
    
    

    # Control command topics (for reading actions)
    left_arm_command_topic: str = "/master/joint_left"
    right_arm_command_topic: str = "/master/joint_right" 
    mobile_command_topic: str = "/cmd_vel"
    endpose_left_cmd_topic: str = "/pos_cmd_left"
    endpose_right_cmd_topic: str = "/pos_cmd_right"
    

    # Joint state topic (for reading robot state)
    left_joint_states_topic: str = "/puppet/joint_left"
    right_joint_states_topic: str = "/puppet/joint_right"
    mobile_base_state_topic: str = "/odom"
    endpose_left_topic: str = "/puppet/end_pose_left"
    endpose_right_topic: str = "/puppet/end_pose_right"
   
   
    # Robot cameras
    # Set to True if you want to use the corresponding cameras in the observations.
    # By default, only the teleop cameras are used.
    with_left_camera: bool = True
    with_right_camera: bool = True
    with_front_camera: bool = True
    

    cam_high_topic: str = "/camera_f/color/image_raw"
    cam_left_topic: str = "/camera_l/color/image_raw"
    cam_right_topic: str = "/camera_r/color/image_raw"

    use_depth_image:  bool = False
    img_front_depth_topic: str = "/camera_f/depth/image_raw"
    img_left_depth_topic: str = "/camera_l/depth/image_raw"
    img_right_depth_topic: str = "/camera_r/depth/image_raw"
    
    # Control rate
    control_rate: float = 50.0  # Hz
    
    # observation  and actions features
    left_arm_joints: list[str] = field(default_factory=lambda: ['left_joint0', 'left_joint1', 'left_joint2', 'left_joint3', 'left_joint4', 'left_joint5', 'left_joint6'])
    right_arm_joints: list[str] = field(default_factory=lambda:['right_joint0', 'right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6'])
    mobile_base_joints: list[str] = field(default_factory=lambda: ['vx', 'vy', 'vtheta'])
    endpose_left_joints: list[str] = field(default_factory=lambda: ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'])
    endpose_right_joints: list[str] = field(default_factory=lambda: ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'])
