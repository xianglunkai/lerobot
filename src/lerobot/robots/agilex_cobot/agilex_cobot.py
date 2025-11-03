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
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry   
import tf.transformations as tf_trans


ROS1_AVAILABLE = True
# except ImportError:
#     ROS1_AVAILABLE = False
#     logging.warning("ROS1 not available. CobotMagic robot will not be functional but can be imported for training.")

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_agilex_cobot import AgilexCobotConfig  # 导入你的配置类

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class ArmState:
    positions: Dict[str, float]
    velocities: Dict[str, float]
    efforts: Dict[str, float]
    last_update: float

    
@dataclass
class BaseState:
    """移动底座状态数据结构"""
    position: Dict[str, float]  # x, y, z 位置
    orientation: Dict[str, float]  # 四元数 (x, y, z, w) 或 欧拉角 (roll, pitch, yaw)
    linear_velocity: Dict[str, float]  # vx, vy, vz
    angular_velocity: Dict[str, float]  # vroll, vpitch, vyaw
    last_update: float
    

SUB_TOPIC_QUEUE_SIZE = 1000  # 订阅话题队列大小
SUB_MSG_QUEUE_SIZE = SUB_TOPIC_QUEUE_SIZE * 2
PUB_TOPIC_QUEUE_SIZE = 10  # 发布话题队列大小

def jpeg_mapping(img):
    img = cv2.imencode(".jpg", img)[1].tobytes()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    return img

from scipy.spatial.transform import Rotation as R
def quat_to_RPY(quat):
    """
    Convert quaternion in (x, y, z, w) order to roll, pitch, yaw (intrinsic).

    Args:
        quat: quaternion as [x, y, z, w]

    Returns:
        tuple: (roll, pitch, yaw) in the specified units
    """

    # Create rotation object from quaternion (scipy expects [x, y, z, w] format)
    rot = R.from_quat([quat.x, quat.y, quat.z, quat.w])

    # Convert to Euler angles (intrinsic R→P→Y = 'xyz')
    euler_angles = rot.as_euler("xyz", degrees=False)

    return euler_angles

from collections import deque

class AgilexCobotROS1Node:
    """ROS1 node for CobotMagic robot communication."""
    
    def __init__(self, config: AgilexCobotConfig):
        if not ROS1_AVAILABLE:
            raise ImportError("ROS1 is required for Agilex Cobot robot.")
        
        # 初始化ROS节点
        rospy.init_node(config.node_name, anonymous=True)
        self.config = config
        
        # Joint state storage
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}
        self.joint_state_timestamp = {}
        
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()
        
        
        # Joint state subscribers (根据配置使用左右臂独立的话题)
        self.joint_state_lock = threading.Lock()
        
        if config.with_l_arm:
            self.puppet_arm_left_deque = deque()
            self.left_joint_state_sub = rospy.Subscriber(
                config.left_joint_states_topic,  # 例如: "/puppet/joint_left"
                JointState,
                self.left_joint_state_callback,
                queue_size=SUB_TOPIC_QUEUE_SIZE,
                tcp_nodelay=True,
            )
        
        if config.with_r_arm:
            self.puppet_arm_right_deque = deque()
            self.right_joint_state_sub = rospy.Subscriber(
                config.right_joint_states_topic,  # 例如: "/puppet/joint_right"
                JointState,
                self.right_joint_state_callback,
                queue_size=SUB_TOPIC_QUEUE_SIZE,
                tcp_nodelay=True,
            )
    
        
        # Command publishers (根据配置使用独立的话题)
        if config.with_l_arm:
            self.left_arm_pub = rospy.Publisher(
                config.left_arm_command_topic,  # 例如: "/master/joint_left"
                JointState,  # 使用JointState消息类型
                queue_size=PUB_TOPIC_QUEUE_SIZE
            )
        
        if config.with_r_arm:
            self.right_arm_pub = rospy.Publisher(
                config.right_arm_command_topic,  # 例如: "/master/joint_right"
                JointState,  # 使用JointState消息类型
                queue_size=PUB_TOPIC_QUEUE_SIZE
            )
        
        self.cv_bridge = CvBridge()
        self.camera_images_timestamp = {}
        

        # 根据配置创建相机订阅者
     
        if hasattr(config, 'cam_high_topic') and config.with_front_camera:
            self.img_front_deque = deque()
            rospy.Subscriber(config.cam_high_topic, Image, self.img_high_img_callback, queue_size=SUB_TOPIC_QUEUE_SIZE, tcp_nodelay=True)
        if hasattr(config, 'cam_left_topic') and config.with_left_camera:
            self.img_left_deque = deque()
            rospy.Subscriber(config.cam_left_topic, Image, self.img_left_img_callback, queue_size=SUB_TOPIC_QUEUE_SIZE, tcp_nodelay=True)
        if hasattr(config, 'cam_right_topic') and config.with_right_camera:
            self.img_right_deque = deque()
            rospy.Subscriber(config.cam_right_topic, Image, self.img_right_img_callback, queue_size=SUB_TOPIC_QUEUE_SIZE, tcp_nodelay=True)

        
        if self.config.use_depth_image:
            self.img_left_depth_deque = deque()
            rospy.Subscriber(
                self.config.img_left_depth_topic,
                Image,
                self.img_left_depth_callback,
                queue_size=SUB_TOPIC_QUEUE_SIZE,
                tcp_nodelay=True,
            )
            
            self.img_right_depth_deque = deque()
            rospy.Subscriber(
                self.config.img_right_depth_topic,
                Image,
                self.img_right_depth_callback,
                queue_size=SUB_TOPIC_QUEUE_SIZE,
                tcp_nodelay=True,
            )
            
            self.img_front_depth_deque = deque()
            rospy.Subscriber(
                self.config.img_front_depth_topic,
                Image,
                self.img_front_depth_callback,
                queue_size=SUB_TOPIC_QUEUE_SIZE,
                tcp_nodelay=True,
            )
            
        # 其他订阅者（如果需要）
        if config.with_mobile_base:
            self.robot_base_deque = deque()
            rospy.Subscriber(
                config.mobile_base_state_topic,
                Odometry,
                self.robot_base_callback,
                queue_size=SUB_TOPIC_QUEUE_SIZE,
                tcp_nodelay=True,
            )
              
        # todo: add support EEF control mode  
        if config.with_endpose:
            self.endpose_left_deque = deque()
            self.endpose_right_deque = deque()
            rospy.Subscriber(
                config.endpose_left_topic,
                PoseStamped,
                self.endpose_left_callback,
                queue_size=SUB_TOPIC_QUEUE_SIZE,
                tcp_nodelay=True,
            )
            rospy.Subscriber(
                config.endpose_right_topic,
                PoseStamped,
                self.endpose_right_callback,
                queue_size=SUB_TOPIC_QUEUE_SIZE,
                tcp_nodelay=True,
        )
        
      
        # self.endpose_left_publisher = rospy.Publisher(config.endpose_left_cmd_topic, PosCmd, queue_size=10)
        # self.endpose_right_publisher = rospy.Publisher(config.endpose_right_cmd_topic, PosCmd, queue_size=10)
        
        if config.with_mobile_base:
            self.robot_base_publisher = rospy.Publisher(config.mobile_command_topic, 
                                                    Twist, 
                                                    queue_size=PUB_TOPIC_QUEUE_SIZE)
            
        logger.info(f"CobotMagicROS1Node initialized with {len(self.camera_subs)} cameras")
        
        logger.info(f"CobotMagicROS1Node initialized with node name: {config.node_name}")
    
    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= SUB_MSG_QUEUE_SIZE:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= SUB_MSG_QUEUE_SIZE:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= SUB_MSG_QUEUE_SIZE:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)
    
    def robot_base_callback(self, msg: Odometry):
        """移动底座状态回调 - 处理 Odometry 消息"""
        if len(self.robot_base_deque) >= SUB_MSG_QUEUE_SIZE:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)
    
    def endpose_left_callback(self, msg):
        if len(self.endpose_left_deque) >= SUB_MSG_QUEUE_SIZE:
            self.endpose_left_deque.popleft()
        self.endpose_left_deque.append(msg)

    def endpose_right_callback(self, msg):
        if len(self.endpose_right_deque) >= SUB_MSG_QUEUE_SIZE:
            self.endpose_right_deque.popleft()
        self.endpose_right_deque.append(msg)
    
            
    def img_high_img_callback(self, msg):
        try:
            if len(self.img_front_deque) >= SUB_MSG_QUEUE_SIZE:
                self.img_front_deque.popleft()
            self.img_front_deque.append(msg)
            self.camera_images_timestamp["high"] = time.time()
                
        except Exception as e:
            logger.error(f"Error processing image from camera high: {e}")
    
    
    def img_left_img_callback(self, msg):
        try:      
            if len(self.img_left_deque) >= SUB_MSG_QUEUE_SIZE:
                self.img_left_deque.popleft()
            self.img_left_deque.append(msg)
            self.camera_images_timestamp["left"] = time.time()
                
        except Exception as e:
            logger.error(f"Error processing image from camera left: {e}")
    
    
    def img_right_img_callback(self, msg):
        try:
            if len(self.img_right_deque) >= SUB_MSG_QUEUE_SIZE:
                self.img_right_deque.popleft()
            self.img_right_deque.append(msg)
            self.camera_images_timestamp["right"] = time.time()
                
        except Exception as e:
            logger.error(f"Error processing image from camera right: {e}") 
        
        
    def left_joint_state_callback(self, msg: JointState):
        """Callback for left arm joint state messages."""
        self._process_joint_states(msg, "left_")
        if len(self.puppet_arm_left_deque) >= SUB_MSG_QUEUE_SIZE:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)
        self.joint_state_timestamp["left_joint"] = time.time()
    
    def right_joint_state_callback(self, msg: JointState):
        """Callback for right arm joint state messages."""
        self._process_joint_states(msg, "right_")
        if len(self.puppet_arm_right_deque) >= SUB_MSG_QUEUE_SIZE:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)
        self.joint_state_timestamp["right_joint"] = time.time()
    
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
         
    
    def get_joint_states(self) -> Dict[str, float]:
        """Get current joint positions."""
        with self.joint_state_lock:
            return self.joint_positions.copy()
        
    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = vel[1]
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[2]
        self.robot_base_publisher.publish(vel_msg)
    
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
        
        
    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.config.control_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.config.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.config.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.config.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.config.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # Set the timestep
            joint_state_msg.name = [
                "joint0",
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ]
            joint_state_msg.position = left_arm
            self.left_arm_pub.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.right_arm_pub.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()    
        
    
    def get_frame(self):
        if (
            len(self.img_left_deque) == 0
            or len(self.img_right_deque) == 0
            or len(self.img_front_deque) == 0
            or (
                self.config.use_depth_image
                and (
                    len(self.img_left_depth_deque) == 0
                    or len(self.img_right_depth_deque) == 0
                    or len(self.img_front_depth_deque) == 0
                )
            )
        ):
            return False
        if self.config.use_depth_image:
            frame_time = min(
                [
                    self.img_left_deque[-1].header.stamp.to_sec(),
                    self.img_right_deque[-1].header.stamp.to_sec(),
                    self.img_front_deque[-1].header.stamp.to_sec(),
                    self.img_left_depth_deque[-1].header.stamp.to_sec(),
                    self.img_right_depth_deque[-1].header.stamp.to_sec(),
                    self.img_front_depth_deque[-1].header.stamp.to_sec(),
                ]
            )
        else:
            frame_time = min(
                [
                    self.img_left_deque[-1].header.stamp.to_sec(),
                    self.img_right_deque[-1].header.stamp.to_sec(),
                    self.img_front_deque[-1].header.stamp.to_sec(),
                ]
            )

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.config.use_depth_image and (
            len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time
        ):
            return False
        if self.config.use_depth_image and (
            len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time
        ):
            return False
        if self.config.use_depth_image and (
            len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time
        ):
            return False
        if self.config.with_mobile_base and (
            len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time
        ):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.cv_bridge.imgmsg_to_cv2(self.img_left_deque[0], 'passthrough')
        self.img_left_deque.popleft()

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.cv_bridge.imgmsg_to_cv2(self.img_right_deque[0], 'passthrough')
        self.img_right_deque.popleft()

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.cv_bridge.imgmsg_to_cv2(self.img_front_deque[0], 'passthrough')
        self.img_front_deque.popleft()

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()
        
        while self.endpose_left_deque[0].header.stamp.to_sec() < frame_time:
            self.endpose_left_deque.popleft()
        endpose_left = self.endpose_left_deque.popleft()

        while self.endpose_right_deque[0].header.stamp.to_sec() < frame_time:
            self.endpose_right_deque.popleft()
        endpose_right = self.endpose_right_deque.popleft()


        img_left_depth = None
        if self.config.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.cv_bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), "passthrough")

        img_right_depth = None
        if self.config.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.cv_bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), "passthrough")

        img_front_depth = None
        if self.config.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.cv_bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), "passthrough")

        robot_base = None
        if self.config.with_mobile_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (
            img_front,
            img_left,
            img_right,
            img_front_depth,
            img_left_depth,
            img_right_depth,
            puppet_arm_left,
            puppet_arm_right,
            endpose_left,
            endpose_right,
            robot_base,
        )
        
    def observation(self):
        motor_position = {}
        cameras = {}
        endpose = {}
        if self.config.cam_use_deque:
        
            while True and not rospy.is_shutdown():
                result = self.get_frame()
                if not result:
                    time.sleep(0.01)
                    print("syn fail when get_ros_observation")
                    continue
                (
                    img_front,
                    img_left,
                    img_right,
                    img_front_depth,
                    img_left_depth,
                    img_right_depth,
                    puppet_arm_left,
                    puppet_arm_right,
                    endpose_left,
                    endpose_right,
                    robot_base,
                ) = result
                
                #["left_joint0:",....]
                for i, name in enumerate(puppet_arm_left.name):
                    full_name = f"left_{name}"  
                    if i < len(puppet_arm_left.position):
                        motor_position[full_name] = puppet_arm_left.position[i]

                for i, name in enumerate(puppet_arm_right.name):
                    full_name = f"right_{name}"  
                    if i < len(puppet_arm_right.position):
                        motor_position[full_name] = puppet_arm_right.position[i]
                        
                
                if self.config.with_mobile_base:
                    motor_position["vx"] = robot_base.twist.linear.x
                    motor_position["vy"] = robot_base.twist.linear.y
                    motor_position["vtheta"] = robot_base.twist.angular.z
                
                cameras["high"] = jpeg_mapping(img_front)
                cameras["left"] = jpeg_mapping(img_left)
                cameras["right"] = jpeg_mapping(img_right)
                
                
                
                left_pos = endpose_left.pose.position
                left_rpy = quat_to_RPY(endpose_left.pose.orientation)
                left_gripper = puppet_arm_left.position[-1]
                endpose_left = np.array([left_pos.x, left_pos.y, left_pos.z, left_rpy[0], left_rpy[1], left_rpy[2], left_gripper])

                right_pos = endpose_right.pose.position
                right_rpy = quat_to_RPY(endpose_right.pose.orientation)
                right_gripper = puppet_arm_right.position[-1]
                endpose_right = np.array(
                    [right_pos.x, right_pos.y, right_pos.z, right_rpy[0], right_rpy[1], right_rpy[2], right_gripper]
                )

                endpose["left"] = endpose_left
                endpose["right"] = endpose_right
    
                return motor_position, cameras, endpose
 
        
class AgilexCobotBase(Robot):
    """Base class for AgilexCobt robot with common functionality."""
    
    config_class = AgilexCobotConfig
    name = "agilex_cobot"
    
    def __init__(self, config: AgilexCobotConfig):
        super().__init__(config)
        self.config = config
        
        # 根据配置构建所有关节名称列表
        self.all_joints = (
            config.left_arm_joints + 
            config.right_arm_joints
        )
        
        self.robot_base = (
            config.mobile_base_joints if config.with_mobile_base else []
        )
        
        logger.info(f"AgilexCobt robot initialized with {len(self.all_joints)} joints")
    
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
        if self.config.with_mobile_base:
            return {
                **dict.fromkeys(
                    self._joint_features.keys(),
                    float,
                ),
                **dict.fromkeys(
                    self._robot_base_features.keys(),
                    float,
                ),
            }
        else:
            return dict.fromkeys(self._joint_features.keys(), float)
    
    @property
    def camera_features(self) -> dict[str, tuple]:
        """Get camera features."""
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


class AgilexCobot(AgilexCobotBase):
    """
    ROS1-based CobotMagic robot with elevator support.
    """
    
    def __init__(self, config: AgilexCobotConfig):
        super().__init__(config)
        
        # ROS1 node (will be initialized on connect)
        self.ros_node: Optional[AgilexCobotROS1Node] = None
        self.ros_thread = None
        
        # 初始化相机 (根据配置)
        self.cameras = make_cameras_from_configs(config.cameras)

    
    @property
    def is_connected(self) -> bool:
        """Check if ROS1 node is connected."""
        if not ROS1_AVAILABLE:
            return False
        now = time.time()
        
        camera_images_ready = False
        joint_states_ready = False
        if self.ros_node:
            cameras_time = np.array(list(self.ros_node.camera_images_timestamp.values()))
            if len(cameras_time) == 3 and abs(max(cameras_time - now)) < self.config.connection_timeout:
                camera_images_ready = True
                    
            joints_time = np.array(list(self.ros_node.joint_state_timestamp.values()))
            if len(joints_time) == 2 and abs(max(joints_time - now)) < self.config.connection_timeout:
                joint_states_ready = True
                
            return joint_states_ready and camera_images_ready
        
        return False
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect to ROS1 and cameras."""
        if not ROS1_AVAILABLE:
            raise ImportError("ROS1 is required for CobotMagic robot operation.")
        
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        # Create and start ROS1 node
        self.ros_node = AgilexCobotROS1Node(self.config)
        
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
            raise DeviceNotConnectedError(f"Failed to receive states from {self}")
        
        logger.info(f"{self} connected successfully")
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current robot observation from ROS1."""
        if not ROS1_AVAILABLE:
            # 返回用于训练的模拟观测数据
            observation = {f"{joint}.pos": 0.0 for joint in self.all_joints}
            
            # 添加移动底座数据（如果适用）
            if self.config.with_mobile_base:
                observation.update({
                    "vx.pos": 0.0,
                    "vy.pos": 0.0,
                    "vtheta.pos": 0.0,
                })
            
            # 添加模拟相机数据
            for cam_name in ['high', 'left', 'right']:
                if hasattr(self.config, f'cam_{cam_name}_topic'):
                    observation[f'camera_{cam_name}'] = np.zeros((480, 640, 3), dtype=np.uint8)
            return observation
        
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        joint_states, camera_images, endpose = self.ros_node.observation()
        
        # Build observation dictionary
        observation = {}
        
        # Add joint positions
        for joint in self.all_joints:
            key = f"{joint}.pos"
            observation[key] = joint_states.get(joint, 0.0)  # Default to 0.0 if not found
            
        # Add mobile base data (if applicable)
        if self.config.with_mobile_base:
            for joint in self.robot_base:
                key = f"{joint}.pos"
                observation[key] = joint_states.get(joint, 0.0)
            
        # 添加相机图像
        for cam_name, _ in self.cameras.items():
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
        
        # Publish mobile base commands (if applicable)
        vel_cmd = []
        if self.config.with_mobile_base:
            for joint in self.robot_base:
                key = f"{joint}.pos"
                vel_cmd.append(action.get(key, 0.0))
            self.ros_node.robot_base_publish(vel_cmd)
          
        return action
    
    def disconnect(self) -> None:
        """断开与ROS1的连接"""
        if not ROS1_AVAILABLE:
            return
        
        logger.info(f"{self} disconnected")
    
    def reset_to_default_positions(self) -> None:
        """
        Reset the robot to its default positions.
        This method should move all joints and components to their predefined default states.
        """
        if not ROS1_AVAILABLE:
            return
        
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # 定义默认位置（根据实际需求调整）
        self._reset_position_left= [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 0.07]
        self._reset_position_right = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 0.07]

        # 发布默认位置命令
        self.ros_node.puppet_arm_publish_continuous(self._reset_position_left, self._reset_position_right)
        
        logger.info(f"{self} reset to default positions")