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


from ..config import RobotConfig
from lerobot.utils.agilex_cobot_ros_manager import AgilexCobotROSManagerConfig

from lerobot.cameras import CameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCameraConfig

@RobotConfig.register_subclass("agilex_cobot")
@dataclass
class AgilexCobotConfig(RobotConfig):
    
    ros_config: AgilexCobotROSManagerConfig = field(default_factory=AgilexCobotROSManagerConfig)
 
    # Tag for external commands control
    # Set to True if you use an external commands system to control the robot,
    # such as the official teleoperation application: https://github.com/pollen-robotics/Reachy2Teleoperation
    # If True, robot.send_action() will not send commands to the robot.
    use_external_commands: bool = False
    
   
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


    def __post_init__(self):
        
        if self.ros_config.with_front_camera:
            self.cameras["high"] = OpenCVCameraConfig(
                    index_or_path = 0,
                    fps=50,
                    width=640,
                    height=480,
                    color_mode=ColorMode.RGB,
                )
            
        if self.ros_config.with_left_camera:
            self.cameras["left"] = OpenCVCameraConfig(
                    index_or_path = 1,
                    fps=50,
                    width=640,
                    height=480,
                    color_mode=ColorMode.RGB,
                )
       
        if self.ros_config.with_right_camera:
            self.cameras["right"] = OpenCVCameraConfig(
                    index_or_path = 2,
                    fps=50,
                    width=640,
                    height=480,
                    color_mode=ColorMode.RGB,
                )
        super().__post_init__()
    
        
  