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


@RobotConfig.register_subclass("qnbot_w")
@dataclass
class QnbotWConfig(RobotConfig):
    """
    Configuration class for ROS2-based dual arm robot with elevator.
    
    This config supports a bimanual robot system with:
    - Left arm: 7 joints (left_arm_joint_1 to left_arm_joint_7)
    - Right arm: 7 joints (right_arm_joint_1 to right_arm_joint_7)  
    - Left gripper: 1 joint (left_gripper_joint)
    - Right gripper: 1 joint (right_gripper_joint)
    - Elevator: 1 joint (elevator_joint)
    Total: 17 degrees of freedom
    """
    
    # ROS2 node configuration
    node_name: str = "lerobot_qnbot_w_node"
    
    # Joint state topic (for reading robot state)
    joint_states_topic: str = "/joint_states"
    
    # Control command topics (for sending actions)
    left_arm_command_topic: str = "/left_arm_forward_position_controller/commands"
    right_arm_command_topic: str = "/right_arm_forward_position_controller/commands" 
    left_gripper_command_topic: str = "/left_gripper_position_controller/commands"
    right_gripper_command_topic: str = "/right_gripper_position_controller/commands"
    elevator_command_topic: str = "/elevator_controller/joint_trajectory"
    
    # Joint names configuration
    left_arm_joints: list[str] = field(default_factory=lambda: [
        "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3", 
        "left_arm_joint_4", "left_arm_joint_5", "left_arm_joint_6", "left_arm_joint_7"
    ])
    
    right_arm_joints: list[str] = field(default_factory=lambda: [
        "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3",
        "right_arm_joint_4", "right_arm_joint_5", "right_arm_joint_6", "right_arm_joint_7"
    ])
    
    left_gripper_joints: list[str] = field(default_factory=lambda: ["left_gripper_joint"])
    right_gripper_joints: list[str] = field(default_factory=lambda: ["right_gripper_joint"])
    elevator_joints: list[str] = field(default_factory=lambda: ["elevator_joint"])
    
    # Camera configuration (if any)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    # Control rate
    control_rate: float = 100.0  # Hz
    
    # Elevator trajectory parameters
    max_elevator_velocity: float = 0.15      # m/s
    max_elevator_acceleration: float = 0.15  # m/s²
    min_elevator_position: float = 0.0       # m
    max_elevator_position: float = 1.2       # m
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate joint configuration
        if len(self.left_arm_joints) != 7:
            raise ValueError(f"Left arm must have 7 joints, got {len(self.left_arm_joints)}")
        if len(self.right_arm_joints) != 7:
            raise ValueError(f"Right arm must have 7 joints, got {len(self.right_arm_joints)}")
        if len(self.left_gripper_joints) != 1:
            raise ValueError(f"Left gripper must have 1 joint, got {len(self.left_gripper_joints)}")
        if len(self.right_gripper_joints) != 1:
            raise ValueError(f"Right gripper must have 1 joint, got {len(self.right_gripper_joints)}")
        if len(self.elevator_joints) != 1:
            raise ValueError(f"Elevator must have 1 joint, got {len(self.elevator_joints)}")
            
        # Check for duplicate joint names
        all_joints = (self.left_arm_joints + self.right_arm_joints + 
                     self.left_gripper_joints + self.right_gripper_joints + 
                     self.elevator_joints)
        if len(all_joints) != len(set(all_joints)):
            raise ValueError("Duplicate joint names found in configuration") 