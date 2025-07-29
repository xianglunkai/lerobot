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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("qnbot_w_teleop")
@dataclass
class QnbotWTeleopConfig(TeleoperatorConfig):
    """
    Configuration class for dual arm teleoperator.
    
    Note: This is primarily included for completeness. For ROS2-based 
    dual arm recording, teleoperator is typically not needed as data 
    is recorded directly from robot state.
    """
    
    # Configuration for the left arm teleoperator
    left_teleop: TeleoperatorConfig
    
    # Configuration for the right arm teleoperator
    right_teleop: TeleoperatorConfig
    
    def __post_init__(self):
        super().__post_init__()
        
        # Basic validation
        if self.left_teleop.id == self.right_teleop.id:
            raise ValueError(
                f"Left and right teleoperator must have different IDs. "
                f"Both teleoperators currently have ID: '{self.left_teleop.id}'"
            ) 