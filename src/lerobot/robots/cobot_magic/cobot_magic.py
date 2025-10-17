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
import cv2
import numpy as np

# 检查ROS1依赖是否可用
# try:
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

ROS1_AVAILABLE = True
# except ImportError:
#     ROS1_AVAILABLE = False
#     logging.warning("ROS1 not available. CobotMagic robot will not be functional but can be imported for training.")

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_cobot_magic import CobotMagicConfig  # 导入你的配置类

logger = logging.getLogger(__name__)


# 移动底座速度控制映射
COBOTMAGIC_VEL = {
    "mobile_base.vx": "linear.x",
    "mobile_base.vy": "linear.y",
    "mobile_base.vtheta": "angular.z",
}

# 右臂关节映射
COBOTMAGIC_R_ARM_JOINTS = {
    "right_waist.pos": "right_joint0",
    "right_shoulder.pos": "right_joint1",
    "right_elbow.pos": "right_joint2",
    "right_forearm_roll.pos": "right_joint3",
    "right_wrist_angle.pos": "right_joint4",
    "right_wrist_rotate.pos": "right_joint5",
    "right_gripper.pos": "right_joint6",
}

# 左臂关节映射
COBOTMAGIC_L_ARM_JOINTS = {
    "left_waist.pos": "left_joint0",
    "left_shoulder.pos": "left_joint1",
    "left_elbow.pos": "left_joint2",
    "left_forearm_roll.pos": "left_joint3",
    "left_wrist_angle.pos": "left_joint4",
    "left_wrist_rotate.pos": "left_joint5",
    "left_gripper.pos": "left_joint6",
}

class CobotMagicROS1Node:
    """ROS1 node for CobotMagic robot communication."""
    
    def __init__(self, config: CobotMagicConfig):
        if not ROS1_AVAILABLE:
            raise ImportError("ROS1 is required for CobotMagic robot operation.")
        
        # 初始化ROS节点
        rospy.init_node(config.node_name, anonymous=True)
        self.config = config
        
        # Joint state storage
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}
        self.joint_state_lock = threading.Lock()
        self.last_joint_state_time = None
        
        # Joint state subscribers (根据配置使用左右臂独立的话题)
        self.left_joint_state_sub = rospy.Subscriber(
            config.left_joint_states_topic,  # 例如: "/puppet/joint_left"
            JointState,
            self.left_joint_state_callback,
            queue_size=10
        )
        self.right_joint_state_sub = rospy.Subscriber(
            config.right_joint_states_topic,  # 例如: "/puppet/joint_right"
            JointState,
            self.right_joint_state_callback,
            queue_size=10
        )
        
        # rospy.Subscriber(
        #     config.endpose_left_topic,
        #     PoseStamped,
        #     self.endpose_left_callback,
        #     queue_size=1000,
        #     tcp_nodelay=True,
        # )
        # rospy.Subscriber(
        #     config.endpose_right_topic,
        #     PoseStamped,
        #     self.endpose_right_callback,
        #     queue_size=1000,
        #     tcp_nodelay=True,
        # )
        
        # Command publishers (根据配置使用独立的话题)
        self.left_arm_pub = rospy.Publisher(
            config.left_arm_command_topic,  # 例如: "/master/joint_left"
            JointState,  # 使用JointState消息类型
            queue_size=10
        )
        self.right_arm_pub = rospy.Publisher(
            config.right_arm_command_topic,  # 例如: "/master/joint_right"
            JointState,  # 使用JointState消息类型
            queue_size=10
        )
        
        
        # 添加相机订阅者
        self.camera_subs = {}
        self.camera_images = {}
        self.camera_lock = threading.Lock()
        self.cv_bridge = CvBridge()
        

        # 根据配置创建相机订阅者
        # if hasattr(config, 'cam_high_topic'):
        #     rospy.Subscriber(config.cam_high_topic, Image, self.img_high_img_callback, queue_size=1000, tcp_nodelay=True)
        # if hasattr(config, 'cam_left_topic'):
        #     rospy.Subscriber(config.cam_left_topic, Image, self.img_left_img_callback, queue_size=1000, tcp_nodelay=True)
        # if hasattr(config, 'cam_right_topic'):
        #     rospy.Subscriber(config.cam_right_topic, Image, self.img_right_img_callback, queue_size=1000, tcp_nodelay=True)

        if hasattr(config, 'cam_left_topic'):
            self._create_camera_subscriber('left', config.cam_left_topic)
        if hasattr(config, 'cam_high_topic'):
            self._create_camera_subscriber('high', config.cam_high_topic)
        if hasattr(config, 'cam_right_topic'):
            self._create_camera_subscriber('right', config.cam_right_topic)
        logger.info(f"CobotMagicROS1Node initialized with {len(self.camera_subs)} cameras")
        logger.info(f"CobotMagicROS1Node initialized with node name: {config.node_name}")
    
    
    
    def img_high_img_callback(self, msg):
        try:
            # 将ROS Image消息转换为OpenCV格式
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # print("11111")
            # 安全地更新图像数据
            # with self.camera_lock:
            self.camera_images["high"] = cv_image
                
        except Exception as e:
            logger.error(f"Error processing image from camera high: {e}")
    
    
    def img_left_img_callback(self, msg):
        try:
            # 将ROS Image消息转换为OpenCV格式
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # print("2222")
            # 安全地更新图像数据
            # with self.camera_lock:
            self.camera_images["left"] = cv_image
                
        except Exception as e:
            logger.error(f"Error processing image from camera left: {e}")
    
    
    def img_right_img_callback(self, msg):
        try:
            # 将ROS Image消息转换为OpenCV格式
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # print("3333")
            # 安全地更新图像数据
            # with self.camera_lock:
            self.camera_images["right"] = cv_image
                
        except Exception as e:
            logger.error(f"Error processing image from camera right: {e}") 
        
        
    def left_joint_state_callback(self, msg: JointState):
        """Callback for left arm joint state messages."""
        self._process_joint_states(msg, "left_")
    
    def right_joint_state_callback(self, msg: JointState):
        """Callback for right arm joint state messages."""
        self._process_joint_states(msg, "right_")
    
    def _process_joint_states(self, msg: JointState, prefix: str):
        """通用处理关节状态回调"""
        with self.joint_state_lock:
            for i, name in enumerate(msg.name):
                full_name = f"{prefix}{name}"  # 添加前缀以区分左右臂关节
                if i < len(msg.position):
                    self.joint_positions[full_name] = msg.position[i]
                if i < len(msg.velocity):
                    self.joint_velocities[full_name] = msg.velocity[i]
                if i < len(msg.effort):
                    self.joint_efforts[full_name] = msg.effort[i]
            self.last_joint_state_time = time.time()
    
    def get_joint_states(self) -> Dict[str, float]:
        """Get current joint positions."""
        with self.joint_state_lock:
            return self.joint_positions.copy()
    
    def publish_left_arm_command(self, positions: list[float]):
        """Publish command to left arm controller."""
        if len(positions) != 7:
            logger.error(f"Left arm command requires 7 values, got {len(positions)}")
            return
        
    
        msg= JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()  # 设置时间戳
        msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        msg.position = positions

        self.left_arm_pub.publish(msg)
    
    def publish_right_arm_command(self, positions: list[float]):
        """Publish command to right arm controller."""
        if len(positions) != 7:
            logger.error(f"Right arm command requires 7 values, got {len(positions)}")
            return
        
        msg= JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()  # 设置时间戳
        msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        msg.position = positions
        
        self.right_arm_pub.publish(msg)
        
    def _create_camera_subscriber(self, camera_name: str, topic: str):
        """创建相机订阅者"""
        try:
            sub = rospy.Subscriber(
                topic,
                Image,
                lambda msg, cam=camera_name: self._camera_callback(msg, cam),
                queue_size=10  # 只需要最新的图像
            )
            self.camera_subs[camera_name] = sub
            logger.info(f"Subscribed to camera topic: {topic} for camera: {camera_name}")
        except Exception as e:
            logger.error(f"Failed to subscribe to camera topic {topic}: {e}")
    
    def _camera_callback(self, msg: Image, camera_name: str):
        """相机话题的回调函数"""
        try:
            # 将ROS Image消息转换为OpenCV格式
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # 安全地更新图像数据
            with self.camera_lock:
                self.camera_images[camera_name] = cv_image
                
        except Exception as e:
            logger.error(f"Error processing image from camera {camera_name}: {e}")
    
    def get_camera_images(self) -> Dict[str, np.ndarray]:
        """获取所有相机的当前图像"""
        # with self.camera_lock:
        return self.camera_images.copy()


class CobotMagicBase(Robot):
    """Base class for CobotMagic robot with common functionality."""
    
    config_class = CobotMagicConfig
    name = "cobot_magic"
    
    def __init__(self, config: CobotMagicConfig):
        super().__init__(config)
        self.config = config
        
        # 根据配置构建所有关节名称列表
        self.all_joints = (
            config.left_arm_joints + 
            config.right_arm_joints
        )
        
        logger.info(f"CobotMagic robot initialized with {len(self.all_joints)} joints")
    
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
        print("Combined observation features from joints and cameras.", {**self._joint_features, **self._cameras_ft})
        return {**self._joint_features, **self._cameras_ft}
    
    
    # @cached_property
    # def observation_features(self) -> dict[str, type | tuple]:
    #     """Combined observation features from joints and cameras."""
    #     features = self._joint_features.copy()
        
    #     # 添加相机特征 (根据配置中的相机话题)
    #     if hasattr(self.config, 'cam_high_topic'):
    #         features['camera_high'] = (480, 640, 3)  # 假设分辨率
    #     if hasattr(self.config, 'cam_left_topic'):
    #         features['camera_left'] = (480, 640, 3)
    #     if hasattr(self.config, 'cam_right_topic'):
    #         features['camera_right'] = (480, 640, 3)  
        
    #     return features
    
    @property
    def action_features(self) -> Dict[str, type]:
        """定义动作空间的特征。"""
        # 示例：返回一个字典，键是关节名，值是数据类型
        return {f"{joint}.pos": float for joint in self.all_joints}

    def calibrate(self) -> None:
        """执行机器人校准程序。"""
        # 示例：实现你的校准逻辑，例如归零、寻找参考点等
        logger.info("Starting calibration procedure...")
        # 你的校准代码 here
        logger.info("Calibration completed.")

    def configure(self, config_dict: Dict[str, Any]) -> None:
        """配置机器人参数。"""
        # 示例：根据提供的配置字典设置机器人参数
        logger.info(f"Configuring robot with: {config_dict}")
        # 你的配置代码 here

    @property
    def is_calibrated(self) -> bool:
        """检查机器人是否已完成校准。"""
        # 示例：返回一个布尔值，表示校准状态
        # 你可能需要设置一个内部状态变量，并在校准成功后更新它
        return self._calibration_status  # 确保在类中初始化和管理这个状态


class CobotMagic(CobotMagicBase):
    """
    ROS1-based CobotMagic robot with elevator support.
    """
    
    def __init__(self, config: CobotMagicConfig):
        super().__init__(config)
        
        # ROS1 node (will be initialized on connect)
        self.ros_node: Optional[CobotMagicROS1Node] = None
        self.ros_thread = None
        
        # 初始化相机 (根据配置)
        self.cameras = make_cameras_from_configs(config.cameras)

    
    @property
    def is_connected(self) -> bool:
        """Check if ROS1 node is connected."""
        if not ROS1_AVAILABLE:
            return False
        
        camera_images_ready = False
        if self.ros_node:
            # with self.ros_node.camera_lock:
            camera_images_ready = len(self.ros_node.camera_images) >= len(self.cameras)
                
        joint_states_ready =  (self.ros_node is not None and 
                self.ros_node.last_joint_state_time is not None and
                (time.time() - self.ros_node.last_joint_state_time) < 3.0)
        
        return joint_states_ready and camera_images_ready
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect to ROS1 and cameras."""
        if not ROS1_AVAILABLE:
            raise ImportError("ROS1 is required for CobotMagic robot operation.")
        
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        # Create and start ROS1 node
        self.ros_node = CobotMagicROS1Node(self.config)
        
        # Start ROS1 spinning in separate thread
        def spin_ros():
            try:
                rospy.spin()
            except Exception as e:
                logger.error(f"ROS1 spinner error: {e}")
        
        self.ros_thread = threading.Thread(target=spin_ros, daemon=True)
        self.ros_thread.start()
        
        # Wait for initial joint states
        start_time = time.time()
        while not self.is_connected and (time.time() - start_time) < 10.0:
            time.sleep(0.1)
            
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Failed to receive joint states from {self}")
        
        logger.info(f"{self} connected successfully")
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current robot observation from ROS1."""
        if not ROS1_AVAILABLE:
            # 返回用于训练的模拟观测数据
            observation = {f"{joint}.pos": 0.0 for joint in self.all_joints}
            # 添加模拟相机数据
            for cam_name in ['high', 'left', 'right']:
                if hasattr(self.config, f'cam_{cam_name}_topic'):
                    observation[f'camera_{cam_name}'] = np.zeros((480, 640, 3), dtype=np.uint8)
            return observation
        
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Get joint states from ROS1
        joint_states = self.ros_node.get_joint_states()
        
        # 获取相机图像
        camera_images = self.ros_node.get_camera_images()
        
        # Build observation dictionary
        observation = {}
        
        # Add joint positions
        for joint in self.all_joints:
            key = f"{joint}.pos"
            observation[key] = joint_states.get(joint, 0.0)  # Default to 0.0 if not found
        
        # 添加相机图像
        for cam_name, _ in self.cameras.items():
            # print(f"++++++++{cam_name}+++++++")
            observation[cam_name] = camera_images[cam_name]
        
        return observation
    
    def send_action(self, action: Dict[str, float]) -> Dict[str, float]:
        """Send action commands to ROS1 controllers."""
        if not ROS1_AVAILABLE:
            return action
        
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Extract joint positions from action for left arm
        left_arm_positions = []
        for joint in self.config.left_arm_joints:
            key = f"{joint}.pos"
            left_arm_positions.append(action.get(key, 0.0))
        
        # Extract joint positions from action for right arm
        right_arm_positions = []
        for joint in self.config.right_arm_joints:
            key = f"{joint}.pos"
            right_arm_positions.append(action.get(key, 0.0))
        
        # Publish commands
        self.ros_node.publish_left_arm_command(left_arm_positions)
        self.ros_node.publish_right_arm_command(right_arm_positions)
        
        return action
    
    def disconnect(self) -> None:
        """断开与ROS1的连接"""
        if not ROS1_AVAILABLE:
            return
        
        # if self.ros_node is not None:
        #     # 取消所有相机订阅
        #     for sub in self.ros_node.camera_subs.values():
        #         sub.unregister()
            
        #     rospy.signal_shutdown("CobotMagic disconnecting")
        #     self.ros_node = None
        
        logger.info(f"{self} disconnected")