# agilex_cobot_robot.py
import logging
from typing import Dict, Any, Optional

import numpy as np
import threading
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_agilex_cobot import AgilexCobotConfig
from lerobot.utils.agilex_cobot_ros_manager import AgilexCobotROSManager

import time
logger = logging.getLogger(__name__)


class AgilexCobotBase(Robot):
    """Base class for AgilexCobot robot with common functionality."""
    
    config_class = AgilexCobotConfig
    name = "agilex_cobot"
    
    def __init__(self, config: AgilexCobotConfig):
        super().__init__(config)
        self.config = config
        
        # Connection state
        self._connected = False
        self._connection_lock = threading.Lock()
        
        # Build joint lists
        self.all_joints =  []
        if config.ros_config.with_l_arm:
            self.all_joints += config.ros_config.left_arm_joints
            
        if config.ros_config.with_r_arm:
            self.all_joints += config.ros_config.right_arm_joints
        
        self.robot_base = (
            config.ros_config.mobile_base_joints if config.ros_config.with_mobile_base else []
        )
        
        # ROS manager (will be initialized in connect)
        self.ros_manager: Optional[AgilexCobotROSManager] = None
        
        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)
        
        logger.info(f"AgilexCobot robot initialized with {len(self.all_joints)} joints")
    
    @property
    def _joint_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.all_joints}
    
    @property
    def _robot_base_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.robot_base}
    
    @property
    def motors_features(self) -> dict[str, type]:
        if self.config.ros_config.with_mobile_base:
            return {
                **dict.fromkeys(self._joint_features.keys(), float),
                **dict.fromkeys(self._robot_base_features.keys(), float),
            }
        else:
            return dict.fromkeys(self._joint_features.keys(), float)
    
    @property
    def camera_features(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) 
            for cam in self.config.cameras
        }
    
    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self.motors_features, **self.camera_features}

    @property
    def action_features(self) -> dict[str, type]:
        return self.motors_features
    
    def calibrate(self) -> None:
        """Execute robot calibration procedure."""
        logger.info("Starting calibration procedure...")
        # Add calibration logic here
        logger.info("Calibration completed.")

    def configure(self, config_dict: Dict[str, Any]) -> None:
        """Configure robot parameters."""
        logger.info(f"Configuring robot with: {config_dict}")
        # Add configuration logic here

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return getattr(self, '_calibration_status', False)


class AgilexCobot(AgilexCobotBase):
    """
    AgilexCobot robot using singleton ROS manager.
    """
    
    def __init__(self, config: AgilexCobotConfig):
        super().__init__(config)
        self._calibration_status = False
    
    @property
    def is_connected(self) -> bool:
        """Check if robot is connected via ROS."""
        return self.ros_manager is not None and self.ros_manager.is_connected()
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect to ROS via the singleton ROS manager."""
        with self._connection_lock:
            if self._connected:
                logger.warning("Already connected")
                return
            
            try:
                # Initialize ROS manager (singleton)
                # Note: We need to convert our teleop config to ROS manager config
                self.ros_manager = AgilexCobotROSManager(self.config.ros_config)
                
                # Wait for initial state updates
                logger.info("Waiting for initial state updates...")
                start_time = time.time()
                timeout = getattr(self.config.ros_config, 'connection_timeout', 5.0)
                
                while not self.is_connected and (time.time() - start_time) < timeout:
                    time.sleep(0.1)
                
                if not self.is_connected:
                    raise ConnectionError("Failed to receive initial state updates within timeout")
                
                self._connected = True
                logger.info(f"{self.name} connected successfully")
                
                if calibrate:
                    self.calibrate()
                
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                self._connected = False
                raise
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current robot observation."""
        if not self.is_connected:
            # Return simulated observation for training
            return self._get_simulated_observation()
        
        # Get observation from ROS manager
        joint_states, camera_images, endpose = self.ros_manager.get_synchronized_observation()
        
        # Build observation dictionary
        observation = {}
        
        # Add joint positions
        for joint in self.all_joints:
            key = f"{joint}.pos"
            observation[key] = joint_states.get(joint, 0.0)
            
        # Add mobile base data
        if self.config.ros_config.with_mobile_base:
            for joint in self.robot_base:
                key = f"{joint}.pos"
                observation[key] = joint_states.get(joint, 0.0)
            
        # Add camera images
        for cam_name in self.cameras:
            observation[cam_name] = camera_images.get(cam_name, np.zeros(
                (self.config.cameras[cam_name].height, 
                 self.config.cameras[cam_name].width, 3), 
                dtype=np.uint8
            ))
        
        return observation
    
    def _get_simulated_observation(self) -> Dict[str, Any]:
        """Get simulated observation for training."""
        observation = {f"{joint}.pos": 0.0 for joint in self.all_joints}
        
        if self.config.ros_config.with_mobile_base:
            observation.update({
                "vx.pos": 0.0,
                "vy.pos": 0.0,
                "vtheta.pos": 0.0,
            })
        
        # Add simulated camera data
        for cam_name in ['high', 'left', 'right']:
            if hasattr(self.config, f'cam_{cam_name}_topic'):
                observation[f'camera_{cam_name}'] = np.zeros((480, 640, 3), dtype=np.uint8)
                
        return observation
    
    def get_eef_pose(self) -> Dict[str, np.ndarray]:
        """Get end-effector poses."""
        if not self.is_connected:
            return {
                "left": np.zeros((7,)),
                "right": np.zeros((7,)),
            }
        
        endpose = self.ros_manager.get_end_effector_poses()
        return endpose
    
    def send_action(self, action: Dict[str, float]) -> Dict[str, float]:
        """Send action commands to robot."""
        if not self.is_connected:
            return action
        
        if self.config.use_external_commands:
            return action
        
        # Extract arm commands
        left_arm_positions = [
            action.get(f"{joint}.pos", 0.0) 
            for joint in self.config.ros_config.left_arm_joints
        ]
        right_arm_positions = [
            action.get(f"{joint}.pos", 0.0) 
            for joint in self.config.ros_config.right_arm_joints
        ]
        
        # Extract mobile base commands
        vel_cmd = []
        if self.config.ros_config.with_mobile_base:
            vel_cmd = [
                action.get(f"{joint}.pos", 0.0) 
                for joint in self.robot_base
            ]
        
        if self.config.use_external_commands:
            return action
        
        # Publish commands via ROS manager
        if self.config.ros_config.with_l_arm:
            self.ros_manager.publish_left_arm_command(left_arm_positions)
        if self.config.ros_config.with_r_arm:
            self.ros_manager.publish_right_arm_command(right_arm_positions)
        if self.config.ros_config.with_mobile_base:
            self.ros_manager.publish_mobile_base_command(vel_cmd)
        
        return action
    
    def reset_to_default_positions(self) -> None:
        """Reset robot to default positions."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Define default positions
        reset_position_left = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, 
                              -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 0.07] 
        reset_position_right = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, 
                               -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 0.07]
        
        # Use continuous publishing for smooth reset
        self.ros_manager.publish_continuous_arm_commands(
            reset_position_left, reset_position_right
        )
        
        if self.config.ros_config.with_mobile_base:
            self.ros_manager.publish_mobile_base_command([0,0])
        
        logger.info(f"{self} reset to default positions")
        
    def control_robot_with_continuous(self, left_pos_cmd=None, right_pos_cmd=None):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
                # Define default positions
        # Use continuous publishing for smooth reset
        self.ros_manager.publish_continuous_arm_commands(
            left_pos_cmd, right_pos_cmd
        )
        
        logger.info(f"{self} reset to default positions")
    def disconnect(self) -> None:
        """Disconnect from ROS."""
        with self._connection_lock:
            if not self._connected:
                return
            
            # Note: ROS manager is singleton, so we don't shut it down here
            # It will be shut down when the program exits or when explicitly called
            self.ros_manager = None
            self._connected = False
            logger.info(f"{self.name} disconnected")