# agilex_cobot_teleop.py
import logging
import time
import threading
from typing import Any, Dict, Optional, List

from ..teleoperator import Teleoperator
from .config_agilex_cobot_teleop import AgilexCobotTeleopConfig
from lerobot.utils.agilex_cobot_ros_manager import AgilexCobotROSManager

logger = logging.getLogger(__name__)


class AgilexCobotTeleop(Teleoperator):
    """
    Agilex Cobot Teleoperator using the singleton ROS manager.
    This class focuses on teleoperation logic while delegating ROS communication to the ROS manager.
    """
    
    config_class = AgilexCobotTeleopConfig
    name = "agilex_cobot_teleop"
    
    def __init__(self, config: AgilexCobotTeleopConfig):
        super().__init__(config)
        self.config = config
        
        # ROS manager (singleton, will be initialized in connect)
        self.ros_manager: Optional[AgilexCobotROSManager] = None
        
        # Connection state
        self._connected = False
        self._connection_lock = threading.Lock()
        
        # Build joint lists
        self.all_joints = (
            config.ros_config.left_arm_joints + 
            config.ros_config.right_arm_joints
        )
        
        self.robot_base = (
            config.ros_config.mobile_base_joints if config.ros_config.with_mobile_base else []
        )
        
        # Create action features and delta mask
        self._action_features_list = self._create_action_features_list()
        self._delta_mask_dict = self._create_delta_mask_dict()
        
        logger.info(f"{self.name} initialized with delta mask: {self._delta_mask_dict}")
    
    def _create_action_features_list(self) -> List[str]:
        """Create action features list for delta_actions_mask mapping."""
        features = []
        
        if self.config.ros_config.with_l_arm:
            features.extend(self.config.ros_config.left_arm_joints)
        
        if self.config.ros_config.with_r_arm:
            features.extend(self.config.ros_config.right_arm_joints)
        
        if self.config.ros_config.with_mobile_base:
            features.extend(self.config.ros_config.mobile_base_joints)
        
        return features
    
    def _create_delta_mask_dict(self) -> Dict[str, bool]:
        """Create delta_actions_mask mapping dictionary."""
        mask_dict = {}
        
        for i, feature_name in enumerate(self._action_features_list):
            if i < len(self.config.delta_actions_mask):
                mask_dict[feature_name] = self.config.delta_actions_mask[i]
            else:
                mask_dict[feature_name] = True
                logger.warning(f"Delta actions mask not specified for {feature_name}, using default (True)")
        
        return mask_dict

    @property
    def _joint_features(self) -> dict[str, type]:
        """Get joint position features."""
        return {f"{joint}.pos": float for joint in self.all_joints}
    
    @property
    def _robot_base_features(self) -> dict[str, type]:
        """Get mobile base features."""
        return {f"{joint}.pos": float for joint in self.robot_base}
    
    @property
    def motors_features(self) -> dict[str, type]:
        """Get all motor features including joints and mobile base."""
        if self.config.ros_config.with_mobile_base:
            return {
                **dict.fromkeys(self._joint_features.keys(), float),
                **dict.fromkeys(self._robot_base_features.keys(), float),
            }
        else:
            return dict.fromkeys(self._joint_features.keys(), float)
        
    @property
    def action_features(self) -> dict[str, type]:
        """Get action features (same as motors_features for teleoperator)."""
        return self.motors_features

    @property
    def feedback_features(self) -> dict[str, type]:
        """Return feedback features dictionary."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if connected to ROS with recent updates."""
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

    @property
    def is_calibrated(self) -> bool:
        """Check if calibrated."""
        return getattr(self, '_calibration_status', True)

    def calibrate(self) -> None:
        """Execute calibration procedure."""
        logger.info("Starting calibration procedure...")
        # Add actual calibration logic here
        self._calibration_status = True
        logger.info("Calibration completed.")

    def configure(self) -> None:
        """Configure robot parameters."""
        logger.info("Configuration procedure would be implemented here")

    def get_action(self) -> dict[str, float]:
        """
        Get current action state including joint positions and mobile base velocity.
        
        Returns:
            Dictionary containing joint positions and mobile base velocities.
            Keys are formatted as "joint_name.pos" for joints and "mobile_base.vx", etc. for base.
        """
        start = time.perf_counter()
        
        if not self.is_connected:
            logger.warning("Not connected, cannot get action state")
            return {}
        
        try:
            # Get synchronized observation from ROS manager
            slave_positions = self.ros_manager.get_slave_joint_states()
            master_positions = self.ros_manager.get_master_joint_states()
            robot_base = self.ros_manager.get_robot_base_state()
            endpose = self.ros_manager.get_end_effector_poses()
            
            # The motor_positions dictionary already contains:
            # - left_joint0, left_joint1, ..., left_joint6
            # - right_joint0, right_joint1, ..., right_joint6  
            # - vx, vtheta (if with_mobile_base)
            
            # Format the keys to match the expected action features
            action_dict = {}
            
            if self.config.use_present_position:
            # Add joint positions with proper formatting
                for joint_name, position in slave_positions.items():
                    if joint_name.startswith(('left_', 'right_')):
                        # Joints: convert "left_joint0" to "left_joint0.pos"
                        action_dict[f"{joint_name}.pos"] = position
            else:
                # Use goal positions if not using present positions
                for joint_name, position in master_positions.items():
                    if joint_name.startswith(('left_', 'right_')):
                        action_dict[f"{joint_name}.pos"] = position
            
            # Add mobile base velocities if applicable
            if self.config.ros_config.with_mobile_base:
                for joint_name, velocity in robot_base.items():
                    action_dict[f"{joint_name}.pos"] = velocity
            
            if self.config.use_eef_pose_action:
                for joint_name, position in endpose.items():
                    # End effector poses: convert "x" to "x.pos", etc.
                    action_dict[f"{joint_name}.pos"] = position
    
                
            return action_dict
            
        except Exception as e:
            logger.error(f"Error getting action state: {e}")
            return {}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Send feedback information (not implemented for this teleoperator)."""
        raise NotImplementedError("Feedback sending not implemented for AgilexCobotTeleop")

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

    def __del__(self):
        """Destructor to ensure proper disconnection."""
        self.disconnect()
    
    @property
    def mobile_base_pose(self) -> Dict[str, float]:
        """Get mobile base pose information from ROS manager."""
        if not self.is_connected:
            return {}
        
        try:
            # Get the latest observation
            motor_positions, _, _ = self.ros_manager.get_synchronized_observation()
            
            # Extract pose information if available
            # Note: This would need to be enhanced based on what information
            # is actually available from the ROS manager
            pose = {}
            
            # If we have access to the robot_base_deque in ROS manager,
            # we could extract actual pose information
            # For now, return empty or placeholder
            if hasattr(self.ros_manager, 'robot_base_deque') and len(self.ros_manager.robot_base_deque) > 0:
                # Extract pose from the latest odometry message
                odom_msg = self.ros_manager.robot_base_deque[-1]
                pose = {
                    "position_x": odom_msg.pose.pose.position.x,
                    "position_y": odom_msg.pose.pose.position.y,
                    "position_z": odom_msg.pose.pose.position.z,
                    "orientation_yaw": self._extract_yaw_from_odom(odom_msg),
                }
            
            return pose
            
        except Exception as e:
            logger.error(f"Error getting mobile base pose: {e}")
            return {}

    def _extract_yaw_from_odom(self, odom_msg):
        """Extract yaw angle from odometry message."""
        try:
            from tf.transformations import euler_from_quaternion
            orientation = odom_msg.pose.pose.orientation
            quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
            _, _, yaw = euler_from_quaternion(quaternion)
            return yaw
        except Exception as e:
            logger.warning(f"Could not extract yaw from odometry: {e}")
            return 0.0

    @property
    def mobile_base_velocity(self) -> Dict[str, float]:
        """Get mobile base velocity information from ROS manager."""
        if not self.is_connected:
            return {}
        
        try:
            # Get the latest observation
            motor_positions, _, _ = self.ros_manager.get_synchronized_observation()
            
            # Extract velocity information
            velocity = {}
            
            if 'vx' in motor_positions:
                velocity["linear_x"] = motor_positions['vx']
            if 'vy' in motor_positions:
                velocity["linear_y"] = motor_positions['vy']
            if 'vtheta' in motor_positions:
                velocity["angular_z"] = motor_positions['vtheta']
            
            return velocity
            
        except Exception as e:
            logger.error(f"Error getting mobile base velocity: {e}")
            return {}

    def get_joint_positions(self) -> Dict[str, float]:
        """Get current joint positions from ROS manager."""
        if not self.is_connected:
            return {}
        
        try:
            joint_states = self.ros_manager.get_joint_states()
            # Format the keys to include .pos suffix
            formatted_states = {}
            for joint_name, position in joint_states.items():
                formatted_states[f"{joint_name}.pos"] = position
            return formatted_states
        except Exception as e:
            logger.error(f"Error getting joint positions: {e}")
            return {}

    def get_end_effector_poses(self) -> Dict[str, Any]:
        """Get end effector poses from ROS manager."""
        if not self.is_connected:
            return {"left": None, "right": None}
        
        try:
            endpose = self.ros_manager.get_end_effector_poses()
            return endpose
        except Exception as e:
            logger.error(f"Error getting end effector poses: {e}")
            return {"left": None, "right": None}

    def reset_to_default_positions(self) -> None:
        """Reset robot to default positions using ROS manager."""
        if not self.is_connected:
            logger.warning("Not connected, cannot reset to default positions")
            return
        
        try:
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
            
            logger.info(f"{self.name} reset to default positions")
            
        except Exception as e:
            logger.error(f"Error resetting to default positions: {e}")