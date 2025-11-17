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
from lerobot.utils.agilex_cobot_ros_manager import AgilexCobotROSManagerConfig

@TeleoperatorConfig.register_subclass("agilex_cobot_teleop")
@dataclass
class AgilexCobotTeleopConfig(TeleoperatorConfig):
 
    ros_config: AgilexCobotROSManagerConfig = field(default_factory=AgilexCobotROSManagerConfig)
    
    # Whether to use the present position of the joints as actions
    # if False, the goal position of the joints will be used
    use_present_position: bool = True
    
    use_eef_pose_action: bool = False


    # Whether to use delta actions (relative changes) or absolute actions (absolute changes)
    use_delta_actions:  bool = False
    delta_actions_mask: list[bool] = field(default_factory=lambda: [True, True, True, True, True, True, False,
                                                                    True, True, True, True, True, True, False,
                                                                    False, False, False])  # For mobile base x, y, theta
    