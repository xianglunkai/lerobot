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

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("cobot_magic_teleop")
@dataclass
class CobotMagicTeleopConfig(TeleoperatorConfig):
    """
    Configuration class for dual arm teleoperator.
    
    For ROS2-based dual arm recording, teleoperator is typically not needed as data 
    is recorded directly from robot master arm.
    """
    
    # ROS2 node configuration
    node_name: str = "lerobot_cobot_magic_teleop_node"
    connection_timeout = 5.0  # seconds
    
    # Whether to use the present position of the joints as actions
    # if False, the goal position of the joints will be used
    use_present_position: bool = False

    # Whether to use delta actions (relative changes) or absolute actions (absolute changes)
    use_delta_actions:  bool = True
    delta_actions_mask: list[bool] = field(default_factory=lambda: [True, True, True, True, True, True, True])
    
    # Which parts of the robot to use
    with_mobile_base: bool = False
    with_l_arm: bool = True
    with_r_arm: bool = True
    
    # Control command topics (for reading actions)
    left_arm_command_topic: str = "/master/joint_left"
    right_arm_command_topic: str = "/master/joint_right" 
    mobile_command_topic: str = "/cmd_vel"
    
    # Joint state topic (for reading robot state)
    left_joint_states_topic: str = "/puppet/joint_left"
    right_joint_states_topic: str = "/puppet/joint_right"
    mobile_base_state_topic: str = "/odom_raw"
    
    # Joint names configuration
    left_arm_joints: list[str] = field(default_factory=lambda: ['left_joint0', 'left_joint1', 'left_joint2', 'left_joint3', 'left_joint4', 'left_joint5', 'left_joint6'])
    
    right_arm_joints: list[str] = field(default_factory=lambda:['right_joint0', 'right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6'])
   
    
    def __post_init__(self):
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
