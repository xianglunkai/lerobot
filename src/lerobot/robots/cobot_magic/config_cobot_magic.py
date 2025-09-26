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


@RobotConfig.register_subclass("cobot_magic")
@dataclass
class CobotMagicConfig(RobotConfig):
    """
    Configuration class for ROS1-based dual arm robot with elevator.
    
    This config supports a bimanual robot system with:
    - Left arm: 7 joints (left_arm_joint_1 to left_arm_joint_7)
    - Right arm: 7 joints (right_arm_joint_1 to right_arm_joint_7)  
    - Left gripper: 1 joint (left_gripper_joint)
    - Right gripper: 1 joint (right_gripper_joint)
    - Elevator: 1 joint (elevator_joint)
    Total: 17 degrees of freedom
    """
    
    # ROS2 node configuration
    node_name: str = "lerobot_cobot_magic_node"
    
    # Joint state topic (for reading robot state)
    left_joint_states_topic: str = "/puppet/joint_left"
    right_joint_states_topic: str = "/puppet/joint_right"
    
    # Control command topics (for sending actions)
    left_arm_command_topic: str = "/master/joint_left"
    right_arm_command_topic: str = "/master/joint_right" 

    # Joint names configuration
    left_arm_joints: list[str] = field(default_factory=lambda: ['left_joint0', 'left_joint1', 'left_joint2', 'left_joint3', 'left_joint4', 'left_joint5', 'left_joint6'])
    
    right_arm_joints: list[str] = field(default_factory=lambda:['right_joint0', 'right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6'])
   
    
    # Camera configuration (if any)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    cam_high_topic: str = "/camera_f/color/image_raw"
    cam_left_topic: str = "/camera_l/color/image_raw"
    cam_right_topic: str = "/camera_r/color/image_raw"
    
    # Control rate
    control_rate: float = 100.0  # Hz

    def __post_init__(self):
        super().__post_init__()
        
        # Validate joint configuration
        if len(self.left_arm_joints) != 7:
            raise ValueError(f"Left arm must have 7 joints, got {len(self.left_arm_joints)}")
        if len(self.right_arm_joints) != 7:
            raise ValueError(f"Right arm must have 7 joints, got {len(self.right_arm_joints)}")
 
            
        # Check for duplicate joint names
        all_joints = (self.left_arm_joints + self.right_arm_joints)
        
        if len(all_joints) != len(set(all_joints)):
            raise ValueError("Duplicate joint names found in configuration") 