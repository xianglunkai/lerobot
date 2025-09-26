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

import logging
from functools import cached_property
from typing import Any, Dict

from ..teleoperator import Teleoperator
from ..utils import make_teleoperator_from_config
from .config_cobot_magic_teleop import CobotMagicTeleopConfig

logger = logging.getLogger(__name__)


class CobotMagicTeleop(Teleoperator):
    """
    CobotMagic teleoperator that wraps two single-arm teleoperators into a bimanual system.
    
    Note: This implementation is provided for completeness. For ROS2-based CobotMagic
    recording, teleoperator is typically not needed as data is recorded directly
    from robot joint states.
    
    This class coordinates two individual Teleoperator instances to work as a unified 
    dual-arm teleoperation system. It handles action and feedback space combination, 
    prefixing keys with 'left_' and 'right_' to distinguish between the two arms.
    
    Example usage:
        ```python
        left_config = SO100LeaderConfig(id="left_leader", port="/dev/ttyUSB0")
        right_config = SO100LeaderConfig(id="right_leader", port="/dev/ttyUSB1")
        
        qnbot_w_config = CobotMagicTeleopConfig(
            id="qnbot_w_teleop",
            left_teleop=left_config,
            right_teleop=right_config
        )
        
        teleop = CobotMagicTeleop(qnbot_w_config)
        teleop.connect()
        ```
    """
    
    config_class = CobotMagicTeleopConfig
    name = "qnbot_w_teleop"
    
    def __init__(self, config: CobotMagicTeleopConfig):
        super().__init__(config)
        self.config = config
        
        # Create individual teleoperator instances for left and right arms
        self.left_teleop = make_teleoperator_from_config(config.left_teleop)
        self.right_teleop = make_teleoperator_from_config(config.right_teleop)
    
    @cached_property
    def action_features(self) -> dict:
        """
        Combine action features from both teleoperators with appropriate prefixes.
        
        Returns:
            Dictionary with combined action features, where left arm features 
            are prefixed with 'left_' and right arm features with 'right_'.
        """
        left_action = self.left_teleop.action_features
        right_action = self.right_teleop.action_features
        
        combined_action = {}
        
        # Add left arm features with prefix
        for key, value in left_action.items():
            prefixed_key = f"left_{key}" if not key.startswith(('left_', 'right_')) else key
            combined_action[prefixed_key] = value
                
        # Add right arm features with prefix
        for key, value in right_action.items():
            prefixed_key = f"right_{key}" if not key.startswith(('left_', 'right_')) else key
            combined_action[prefixed_key] = value
                
        return combined_action
    
    @cached_property
    def feedback_features(self) -> dict:
        """
        Combine feedback features from both teleoperators with appropriate prefixes.
        
        Returns:
            Dictionary with combined feedback features, where left arm features 
            are prefixed with 'left_' and right arm features with 'right_'.
        """
        left_feedback = self.left_teleop.feedback_features
        right_feedback = self.right_teleop.feedback_features
        
        combined_feedback = {}
        
        # Add left arm features with prefix
        for key, value in left_feedback.items():
            prefixed_key = f"left_{key}" if not key.startswith(('left_', 'right_')) else key
            combined_feedback[prefixed_key] = value
                
        # Add right arm features with prefix
        for key, value in right_feedback.items():
            prefixed_key = f"right_{key}" if not key.startswith(('left_', 'right_')) else key
            combined_feedback[prefixed_key] = value
                
        return combined_feedback
    
    @property
    def is_connected(self) -> bool:
        """Check if both teleoperators are connected."""
        return self.left_teleop.is_connected and self.right_teleop.is_connected
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect both teleoperators."""
        logger.info(f"Connecting {self.left_teleop}")
        self.left_teleop.connect(calibrate=calibrate)
        
        logger.info(f"Connecting {self.right_teleop}")
        self.right_teleop.connect(calibrate=calibrate)
        
        logger.info(f"{self} connected successfully")
    
    @property
    def is_calibrated(self) -> bool:
        """Check if both teleoperators are calibrated."""
        return self.left_teleop.is_calibrated and self.right_teleop.is_calibrated
    
    def calibrate(self) -> None:
        """Calibrate both teleoperators."""
        logger.info(f"Calibrating {self.left_teleop}")
        self.left_teleop.calibrate()
        
        logger.info(f"Calibrating {self.right_teleop}")
        self.right_teleop.calibrate()
        
        logger.info(f"{self} calibration completed")
    
    def configure(self) -> None:
        """Configure both teleoperators."""
        self.left_teleop.configure()
        self.right_teleop.configure()
    
    def get_action(self) -> Dict[str, Any]:
        """
        Get combined actions from both teleoperators.
        
        Returns:
            Dictionary with actions from both teleoperators, with appropriate prefixes.
        """
        left_action = self.left_teleop.get_action()
        right_action = self.right_teleop.get_action()
        
        combined_action = {}
        
        # Add left arm actions with prefix
        for key, value in left_action.items():
            prefixed_key = f"left_{key}" if not key.startswith(('left_', 'right_')) else key
            combined_action[prefixed_key] = value
                
        # Add right arm actions with prefix
        for key, value in right_action.items():
            prefixed_key = f"right_{key}" if not key.startswith(('left_', 'right_')) else key
            combined_action[prefixed_key] = value
                
        return combined_action
    
    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Send feedback to both teleoperators.
        
        Args:
            feedback: Dictionary with feedback for both teleoperators. Keys should be 
                     prefixed with 'left_' or 'right_' to indicate which teleoperator 
                     they belong to.
        """
        # Separate feedback for left and right teleoperators
        left_feedback = {}
        right_feedback = {}
        
        for key, value in feedback.items():
            if key.startswith("left_"):
                # Remove prefix for the individual teleoperator
                left_key = key[5:]  # Remove "left_"
                left_feedback[left_key] = value
            elif key.startswith("right_"):
                # Remove prefix for the individual teleoperator
                right_key = key[6:]  # Remove "right_"
                right_feedback[right_key] = value
            else:
                logger.warning(f"Feedback key '{key}' has no left_/right_ prefix. Skipping.")
        
        # Send feedback to individual teleoperators
        if left_feedback:
            self.left_teleop.send_feedback(left_feedback)
        if right_feedback:
            self.right_teleop.send_feedback(right_feedback)
    
    def disconnect(self) -> None:
        """Disconnect both teleoperators."""
        logger.info(f"Disconnecting {self.left_teleop}")
        self.left_teleop.disconnect()
        
        logger.info(f"Disconnecting {self.right_teleop}")
        self.right_teleop.disconnect()
        
        logger.info(f"{self} disconnected") 