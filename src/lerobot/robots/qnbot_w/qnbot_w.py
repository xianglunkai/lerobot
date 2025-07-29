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
import threading
import time
import math
from functools import cached_property
from typing import Any, Dict, Optional

import numpy as np

# 检查ROS2依赖是否可用
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    # 在没有ROS2的情况下，创建mock类以避免导入错误
    Node = object
    JointState = object
    Float64MultiArray = object
    JointTrajectory = object
    JointTrajectoryPoint = object
    Duration = object
    
    # 创建一个简单的mock rclpy模块
    class MockRclpy:
        @staticmethod
        def init():
            pass
        
        @staticmethod
        def ok():
            return False
        
        class executors:
            class SingleThreadedExecutor:
                def __init__(self):
                    pass
                def add_node(self, node):
                    pass
                def spin(self):
                    pass
                def shutdown(self):
                    pass
    
    rclpy = MockRclpy()
    logging.warning("ROS2 not available. QnbotW robot will not be functional but can be imported for training.")

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_qnbot_w import QnbotWConfig

logger = logging.getLogger(__name__)


class QnbotWROS2Node(Node):
    """ROS2 node for QnbotW robot communication."""
    
    def __init__(self, config: QnbotWConfig):
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2 is required for QnbotW robot operation. Install ROS2 to use QnbotW robot.")
        
        super().__init__(config.node_name)
        self.config = config
        
        # Joint state storage
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}
        self.joint_state_lock = threading.Lock()
        self.last_joint_state_time = None
        
        # Joint state subscriber
        self.joint_state_sub = self.create_subscription(
            JointState,
            config.joint_states_topic,
            self.joint_state_callback,
            10
        )
        
        # Command publishers
        self.left_arm_pub = self.create_publisher(
            Float64MultiArray, 
            config.left_arm_command_topic, 
            10
        )
        self.right_arm_pub = self.create_publisher(
            Float64MultiArray, 
            config.right_arm_command_topic, 
            10
        )
        self.left_gripper_pub = self.create_publisher(
            Float64MultiArray, 
            config.left_gripper_command_topic, 
            10
        )
        self.right_gripper_pub = self.create_publisher(
            Float64MultiArray, 
            config.right_gripper_command_topic, 
            10
        )
        self.elevator_pub = self.create_publisher(
            JointTrajectory, 
            config.elevator_command_topic, 
            10
        )
        
        logger.info(f"QnbotWROS2Node initialized with node name: {config.node_name}")
    
    def joint_state_callback(self, msg: JointState):
        """Callback for joint state messages."""
        with self.joint_state_lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.joint_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.joint_velocities[name] = msg.velocity[i]
                if i < len(msg.effort):
                    self.joint_efforts[name] = msg.effort[i]
            self.last_joint_state_time = time.time()
    
    def get_joint_states(self) -> Dict[str, float]:
        """Get current joint positions."""
        with self.joint_state_lock:
            return self.joint_positions.copy()
    
    def get_joint_velocities(self) -> Dict[str, float]:
        """Get current joint velocities."""
        with self.joint_state_lock:
            return self.joint_velocities.copy()
    
    def publish_left_arm_command(self, positions: list[float]):
        """Publish command to left arm controller."""
        msg = Float64MultiArray()
        msg.data = positions
        self.left_arm_pub.publish(msg)
    
    def publish_right_arm_command(self, positions: list[float]):
        """Publish command to right arm controller."""
        msg = Float64MultiArray()
        msg.data = positions
        self.right_arm_pub.publish(msg)
    
    def publish_left_gripper_command(self, positions: list[float]):
        """Publish command to left gripper controller."""
        msg = Float64MultiArray()
        msg.data = positions
        self.left_gripper_pub.publish(msg)
    
    def publish_right_gripper_command(self, positions: list[float]):
        """Publish command to right gripper controller."""
        msg = Float64MultiArray()
        msg.data = positions
        self.right_gripper_pub.publish(msg)
    
    def publish_elevator_trajectory(self, trajectory: JointTrajectory):
        """Publish trajectory to elevator controller."""
        self.elevator_pub.publish(trajectory)


class QnbotWBase(Robot):
    """Base class for QnbotW robot with common functionality."""
    
    config_class = QnbotWConfig
    name = "qnbot_w"
    
    def __init__(self, config: QnbotWConfig):
        super().__init__(config)
        self.config = config
        
        # All joint names in order
        self.all_joints = (
            config.left_arm_joints + 
            config.right_arm_joints + 
            config.left_gripper_joints + 
            config.right_gripper_joints + 
            config.elevator_joints
        )
        
        logger.info(f"QnbotW robot initialized with {len(self.all_joints)} joints")
    
    @property
    def _joint_features(self) -> dict[str, type]:
        """Get joint position features."""
        return {f"{joint}.pos": float for joint in self.all_joints}
    
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Get camera features."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) 
            for cam in self.config.cameras
        }
    
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features from joints and cameras."""
        return {**self._joint_features, **self._cameras_ft}
    
    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features for all joints."""
        return self._joint_features


class QnbotW(QnbotWBase):
    """
    ROS2-based QnbotW robot with elevator support.
    
    This robot communicates with ROS2 controllers to control a bimanual robot system with:
    - Left arm: 7 joints
    - Right arm: 7 joints  
    - Left gripper: 1 joint
    - Right gripper: 1 joint
    - Elevator: 1 joint (using trajectory control)
    Total: 17 degrees of freedom
    
    The robot reads state from /joint_states topic and publishes commands to 
    individual controller topics.
    """
    
    def __init__(self, config: QnbotWConfig):
        super().__init__(config)
        
        # 如果在训练环境中（没有ROS2），创建mock实现
        if not ROS2_AVAILABLE:
            logger.warning("ROS2 not available. Creating mock QnbotW for training purposes.")
            self._create_mock_implementation()
            return
        
        # ROS2 node (will be initialized on connect)
        self.ros_node: Optional[QnbotWROS2Node] = None
        self.ros_executor = None
        self.ros_thread = None
        
        # Cameras
        self.cameras = make_cameras_from_configs(config.cameras)
    
    def _create_mock_implementation(self):
        """Create mock implementation for training without ROS2."""
        self.ros_node = None
        self.ros_executor = None
        self.ros_thread = None
        self.cameras = {}
        
        # 创建默认的观测数据
        self._mock_observation = {}
        for joint in self.all_joints:
            self._mock_observation[f"{joint}.pos"] = 0.0
    
    @property
    def is_connected(self) -> bool:
        """Check if ROS2 node is connected and cameras are connected."""
        if not ROS2_AVAILABLE:
            return False
        
        ros_connected = (self.ros_node is not None and 
                        self.ros_node.last_joint_state_time is not None and
                        (time.time() - self.ros_node.last_joint_state_time) < 1.0)  # 1 second timeout
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        return ros_connected and cameras_connected
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect to ROS2 and cameras."""
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2 is required for QnbotW robot operation. Install ROS2 to use QnbotW robot.")
        
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        # Initialize ROS2 if not already done
        if not rclpy.ok():
            rclpy.init()
        
        # Create and start ROS2 node
        self.ros_node = QnbotWROS2Node(self.config)
        
        # Start ROS2 spinning in separate thread
        self.ros_executor = rclpy.executors.SingleThreadedExecutor()
        self.ros_executor.add_node(self.ros_node)
        
        def spin_ros():
            try:
                self.ros_executor.spin()
            except Exception as e:
                logger.error(f"ROS2 executor error: {e}")
        
        self.ros_thread = threading.Thread(target=spin_ros, daemon=True)
        self.ros_thread.start()
        
        # Connect cameras
        for name, camera in self.cameras.items():
            logger.info(f"Connecting camera {name}")
            camera.connect()
        
        # Wait for initial joint states
        start_time = time.time()
        while not self.is_connected and (time.time() - start_time) < 5.0:
            time.sleep(0.1)
            
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Failed to receive joint states from {self}")
        
        if calibrate:
            self.calibrate()
        
        logger.info(f"{self} connected successfully")
    
    @property
    def is_calibrated(self) -> bool:
        """ROS2 robots don't require calibration."""
        return True
    
    def calibrate(self) -> None:
        """ROS2 robots don't require calibration."""
        pass
    
    def configure(self) -> None:
        """Configure robot (no-op for ROS2 robots)."""
        pass
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current robot observation from ROS2 and cameras."""
        if not ROS2_AVAILABLE:
            # Return mock observation for training
            return self._mock_observation.copy()
        
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Get joint states from ROS2
        joint_states = self.ros_node.get_joint_states()
        
        # Build observation dictionary
        observation = {}
        
        # Add joint positions
        for joint in self.all_joints:
            key = f"{joint}.pos"
            if joint in joint_states:
                observation[key] = joint_states[joint]
            else:
                logger.warning(f"Joint {joint} not found in joint states")
                observation[key] = 0.0  # Default value
        
        # Add camera observations
        for name, camera in self.cameras.items():
            observation[name] = camera.read()
        
        return observation
    
    def generate_elevator_trajectory(self, start_position: float, target_position: float) -> JointTrajectory:
        """Generate smooth trajectory for elevator movement."""
        if not ROS2_AVAILABLE:
            return None
        
        distance = abs(target_position - start_position)
        
        # For very small movements, use a simple trajectory
        if distance < 0.01:
            total_time = 0.5
            point = JointTrajectoryPoint()
            point.positions = [target_position]
            point.velocities = [0.0]
            point.time_from_start = Duration(
                sec=0,
                nanosec=int(total_time * 1e9)
            )
            
            traj = JointTrajectory()
            traj.header.stamp = self.ros_node.get_clock().now().to_msg()
            traj.joint_names = ['elevator_joint']
            traj.points = [point]
            return traj
        
        # Calculate trajectory phases
        direction = 1 if target_position > start_position else -1
        max_vel = self.config.max_elevator_velocity
        max_accel = self.config.max_elevator_acceleration
        
        # Calculate acceleration time and distance
        accel_time = max_vel / max_accel
        accel_distance = 0.5 * max_accel * accel_time * accel_time
        
        if 2 * accel_distance <= distance:
            # Can reach maximum velocity
            constant_velocity_distance = distance - 2 * accel_distance
            constant_velocity_time = constant_velocity_distance / max_vel
            decel_time = accel_time
            total_time = accel_time + constant_velocity_time + decel_time
        else:
            # Cannot reach maximum velocity
            accel_time = math.sqrt(distance / max_accel)
            decel_time = accel_time
            constant_velocity_time = 0
            total_time = accel_time + decel_time
        
        # Generate trajectory points
        num_points = min(5, max(2, int(total_time * 10)))  # 5 points max, 2 points min
        times = np.linspace(0, total_time, num_points)
        
        points = []
        for i, t in enumerate(times):
            point = JointTrajectoryPoint()
            
            # Calculate position and velocity based on time
            if t <= accel_time:
                # Acceleration phase
                s = 0.5 * max_accel * t * t
                v = max_accel * t
            elif t <= accel_time + constant_velocity_time:
                # Constant velocity phase
                s = accel_distance + max_vel * (t - accel_time)
                v = max_vel
            else:
                # Deceleration phase
                t_decel = t - (accel_time + constant_velocity_time)
                s = distance - 0.5 * max_accel * (decel_time - t_decel) * (decel_time - t_decel)
                v = max_vel - max_accel * t_decel
            
            position = start_position + direction * s
            velocity = direction * v if i < num_points - 1 else 0.0  # Final point has zero velocity
            
            # Clamp position to limits
            position = max(self.config.min_elevator_position, 
                          min(self.config.max_elevator_position, position))
            
            point.positions = [position]
            point.velocities = [velocity]
            point.time_from_start = Duration(
                sec=int(t),
                nanosec=int((t - int(t)) * 1e9)
            )
            points.append(point)
        
        # Create trajectory message
        traj = JointTrajectory()
        traj.header.stamp = self.ros_node.get_clock().now().to_msg()
        traj.joint_names = ['elevator_joint']
        traj.points = points
        
        return traj
    
    def send_action(self, action: Dict[str, float]) -> Dict[str, float]:
        """Send action commands to ROS2 controllers."""
        if not ROS2_AVAILABLE:
            # In training mode, just return the action
            return action
        
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Extract joint positions from action
        left_arm_positions = []
        right_arm_positions = []
        left_gripper_positions = []
        right_gripper_positions = []
        
        # Left arm
        for joint in self.config.left_arm_joints:
            key = f"{joint}.pos"
            if key in action:
                left_arm_positions.append(action[key])
            else:
                logger.warning(f"Action missing for {key}")
                left_arm_positions.append(0.0)
        
        # Right arm
        for joint in self.config.right_arm_joints:
            key = f"{joint}.pos"
            if key in action:
                right_arm_positions.append(action[key])
            else:
                logger.warning(f"Action missing for {key}")
                right_arm_positions.append(0.0)
        
        # Left gripper
        for joint in self.config.left_gripper_joints:
            key = f"{joint}.pos"
            if key in action:
                left_gripper_positions.append(action[key])
            else:
                logger.warning(f"Action missing for {key}")
                left_gripper_positions.append(0.0)
        
        # Right gripper
        for joint in self.config.right_gripper_joints:
            key = f"{joint}.pos"
            if key in action:
                right_gripper_positions.append(action[key])
            else:
                logger.warning(f"Action missing for {key}")
                right_gripper_positions.append(0.0)
        
        # Elevator (trajectory control)
        for joint in self.config.elevator_joints:
            key = f"{joint}.pos"
            if key in action:
                current_pos = self.ros_node.get_joint_states().get(joint, 0.0)
                target_pos = action[key]
                
                # Clamp target position to limits
                target_pos = max(self.config.min_elevator_position, 
                               min(self.config.max_elevator_position, target_pos))
                
                # Generate and publish trajectory
                trajectory = self.generate_elevator_trajectory(current_pos, target_pos)
                if trajectory:
                    self.ros_node.publish_elevator_trajectory(trajectory)
                
                logger.debug(f"Elevator trajectory: {current_pos:.3f} → {target_pos:.3f}")
            else:
                logger.warning(f"Action missing for {key}")
        
        # Publish commands for arms and grippers
        if left_arm_positions:
            self.ros_node.publish_left_arm_command(left_arm_positions)
        if right_arm_positions:
            self.ros_node.publish_right_arm_command(right_arm_positions)
        if left_gripper_positions:
            self.ros_node.publish_left_gripper_command(left_gripper_positions)
        if right_gripper_positions:
            self.ros_node.publish_right_gripper_command(right_gripper_positions)
        
        return action  # Return the action as sent
    
    def disconnect(self) -> None:
        """Disconnect from ROS2 and cameras."""
        if not ROS2_AVAILABLE:
            return
        
        if not self.is_connected:
            logger.warning(f"{self} already disconnected or not connected")
            return
        
        # Disconnect cameras
        for name, camera in self.cameras.items():
            logger.info(f"Disconnecting camera {name}")
            camera.disconnect()
        
        # Stop ROS2 executor
        if self.ros_executor:
            self.ros_executor.shutdown()
            
        # Stop ROS2 thread
        if self.ros_thread and self.ros_thread.is_alive():
            self.ros_thread.join(timeout=1.0)
        
        # Destroy ROS2 node
        if self.ros_node:
            self.ros_node.destroy_node()
            self.ros_node = None
        
        logger.info(f"{self} disconnected") 