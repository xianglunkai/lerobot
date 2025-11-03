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

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import logging
from functools import cached_property
from typing import Any, Dict, Optional, List
import time
import threading
from dataclasses import dataclass
import tf.transformations as tf_trans

from ..teleoperator import Teleoperator
from ..utils import make_teleoperator_from_config
from .config_agilex_cobot_teleop import AgilexCobotTeleopConfig

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

# 状态数据结构
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

class AgilexCobotTeleop(Teleoperator):
    """
    CobotMagic teleoperator that wraps two single-arm teleoperators into a bimanual system.
    Uses ROS1 subscriptions for state monitoring and command reception.
    """
    
    config_class = AgilexCobotTeleopConfig
    name = "cobot_magic_teleop"
    
    def __init__(self, config: AgilexCobotTeleopConfig):
        super().__init__(config)
        self.config = config
        
        # ROS状态存储
        self._left_arm_state: Optional[ArmState] = None
        self._right_arm_state: Optional[ArmState] = None
        self._mobile_base_state: Optional[BaseState] = None
        
        # 动作命令存储
        self._left_arm_command: Optional[Dict[str, float]] = None
        self._right_arm_command: Optional[Dict[str, float]] = None
        self._mobile_base_command: Optional[Dict[str, float]] = None
        
        # 连接状态和锁
        self._connected = False
        self._connection_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._command_lock = threading.Lock()
        
        # 生成关节映射字典
        self.joints_dict: dict[str, str] = self._generate_joints_dict()
        
        # 创建反向映射用于状态处理
        self._joint_name_to_key = {v: k for k, v in self.joints_dict.items()}
    
        # 创建动作特征列表（用于delta_actions_mask映射）
        self._action_features_list = self._create_action_features_list()
        
        # 创建delta_actions_mask映射字典
        self._delta_mask_dict = self._create_delta_mask_dict()
        logger.info(f"Delta actions mask: {self._delta_mask_dict}")
        
        # ROS节点初始化（延迟到connect方法中）
        self._node_initialized = False
        
        logger.info(f"{self.name} initialized with config: {config}")

    def _generate_joints_dict(self) -> dict[str, str]:
        """生成关节映射字典"""
        joints = {}
        if self.config.with_l_arm:
            joints.update(COBOTMAGIC_L_ARM_JOINTS)
        if self.config.with_r_arm:
            joints.update(COBOTMAGIC_R_ARM_JOINTS)
        if self.config.with_mobile_base:
            joints.update(COBOTMAGIC_VEL)
        return joints

    def _create_action_features_list(self) -> List[str]:
        """创建动作特征列表（用于delta_actions_mask映射）"""
        features = []
        
        # 添加左臂关节
        if self.config.with_l_arm:
            features.extend(COBOTMAGIC_L_ARM_JOINTS.keys())
        
        # 添加右臂关节
        if self.config.with_r_arm:
            features.extend(COBOTMAGIC_R_ARM_JOINTS.keys())
        
        # 添加移动底座速度
        if self.config.with_mobile_base:
            features.extend(COBOTMAGIC_VEL.keys())
        
        return features
    
    def _create_delta_mask_dict(self) -> Dict[str, bool]:
        """创建delta_actions_mask映射字典"""
        mask_dict = {}
        
        for i, feature_name in enumerate(self._action_features_list):
            if i < len(self.config.delta_actions_mask):
                mask_dict[feature_name] = self.config.delta_actions_mask[i]
            else:
                # 如果mask长度不够，默认使用相对动作
                mask_dict[feature_name] = True
                logger.warning(f"Delta actions mask not specified for {feature_name}, using default (True)")
        
        return mask_dict

    def _init_ros_node(self):
        """初始化ROS节点和订阅器"""
        if not self._node_initialized:
            try:
                # 初始化ROS节点
                rospy.init_node(self.config.node_name, anonymous=True)
                
                # 状态订阅器
                if self.config.with_l_arm:
                    rospy.Subscriber(
                        self.config.left_joint_states_topic, 
                        JointState,
                        self._left_joint_state_callback,
                        queue_size=10
                    )
                
                if self.config.with_r_arm:
                    rospy.Subscriber(
                        self.config.right_joint_states_topic, 
                        JointState,
                        self._right_joint_state_callback,
                        queue_size=10
                    )
                
                if self.config.with_mobile_base:
                    rospy.Subscriber(
                        self.config.mobile_base_state_topic,
                        Odometry,
                        self._mobile_base_state_callback,
                        queue_size=10
                    )
                
                # 命令订阅器
                if self.config.with_l_arm:
                    rospy.Subscriber(
                        self.config.left_arm_command_topic, 
                        JointState,
                        self._left_arm_command_callback,
                        queue_size=10
                    )
                
                if self.config.with_r_arm:
                    rospy.Subscriber(
                        self.config.right_arm_command_topic, 
                        JointState,
                        self._right_arm_command_callback,
                        queue_size=10
                    )
                
                if self.config.with_mobile_base:
                    rospy.Subscriber(
                        self.config.mobile_command_topic,
                        Twist,
                        self._mobile_base_command_callback,
                        queue_size=10
                    )
                
                self._node_initialized = True
                logger.info("ROS node and subscribers initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize ROS node: {e}")
                raise

    # ROS回调函数
    def _left_joint_state_callback(self, msg: JointState):
        """左臂关节状态回调"""
        with self._state_lock:
            positions = {}
            velocity = {}
            for i, name in enumerate(msg.name):
                full_name = f"{"left_"}{name}"  # 添加前缀以区分左右臂关节
                if full_name in self._joint_name_to_key:
                    key = self._joint_name_to_key[full_name]
                    positions[key] = msg.position[i] if i < len(msg.position) else 0.0
                    velocity[key] = msg.velocity[i] if i < len(msg.velocity) else 0.0
            
            self._left_arm_state = ArmState(
                positions=positions,
                velocities=velocity,  # 可根据需要添加
                efforts={},     # 可根据需要添加
                last_update=time.time()
            )

    def _right_joint_state_callback(self, msg: JointState):
        """右臂关节状态回调"""
        with self._state_lock:
            positions = {}
            velocity = {}
            for i, name in enumerate(msg.name):
                full_name = f"{"right_"}{name}"  # 添加前缀以区分左右臂关节
                if full_name in self._joint_name_to_key:
                    key = self._joint_name_to_key[full_name]
                    positions[key] = msg.position[i] if i < len(msg.position) else 0.0
                    velocity[key] = msg.velocity[i] if i < len(msg.velocity) else 0.0
            
            self._right_arm_state = ArmState(
                positions=positions,
                velocities=velocity,
                efforts={},
                last_update=time.time()
            )

    def _mobile_base_state_callback(self, msg: Odometry):
        """移动底座状态回调 - 处理 Odometry 消息"""
        try:
            with self._state_lock:
                # 提取位置信息
                position = {
                    "x": msg.pose.pose.position.x,
                    "y": msg.pose.pose.position.y,
                    "z": msg.pose.pose.position.z
                }
                
                # 提取方向信息（四元数）
                orientation_quat = {
                    "x": msg.pose.pose.orientation.x,
                    "y": msg.pose.pose.orientation.y,
                    "z": msg.pose.pose.orientation.z,
                    "w": msg.pose.pose.orientation.w
                }
                
                # 将四元数转换为欧拉角（可选，便于理解）
                euler_angles = self._quaternion_to_euler(orientation_quat)
                
                # 提取线速度和角速度
                linear_velocity = {
                    "x": msg.twist.twist.linear.x,
                    "y": msg.twist.twist.linear.y,
                    "z": msg.twist.twist.linear.z
                }
                
                angular_velocity = {
                    "x": msg.twist.twist.angular.x,
                    "y": msg.twist.twist.angular.y,
                    "z": msg.twist.twist.angular.z
                }
                
                self._mobile_base_state = BaseState(
                    position=position,
                    orientation={
                        "quaternion": orientation_quat,
                        "euler": euler_angles  # 添加欧拉角表示
                    },
                    linear_velocity=linear_velocity,
                    angular_velocity=angular_velocity,
                    last_update=time.time()
                )
                
                # 调试日志（可选）
                if self.config.debug_mode:
                    logger.debug(f"Mobile base state updated - "
                               f"Position: ({position['x']:.3f}, {position['y']:.3f}, {position['z']:.3f}), "
                               f"Linear vel: ({linear_velocity['x']:.3f}, {linear_velocity['y']:.3f}), "
                               f"Angular vel: {angular_velocity['z']:.3f}")
                
        except Exception as e:
            logger.error(f"Error processing mobile base odometry: {e}")
    def _quaternion_to_euler(self, quat: Dict[str, float]) -> Dict[str, float]:
        """将四元数转换为欧拉角"""
        try:
            # 从四元数获取欧拉角 (roll, pitch, yaw)
            euler = tf_trans.euler_from_quaternion([
                quat["w"],
                quat["x"],
                quat["y"], 
                quat["z"],
            ])
            
            return {
                "roll": euler[0],
                "pitch": euler[1],
                "yaw": euler[2]
            }
        except Exception as e:
            logger.warning(f"Failed to convert quaternion to Euler angles: {e}")
            return {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}


    def _left_arm_command_callback(self, msg: JointState):
        """左臂命令回调"""
        with self._command_lock:
            command = {}
            for i, name in enumerate(msg.name):
                full_name = f"{"left_"}{name}"  # 添加前缀以区分左右臂关节
                if full_name in self._joint_name_to_key:
                    key = self._joint_name_to_key[full_name]
                    command[key] = msg.position[i] if i < len(msg.position) else 0.0
            
            self._left_arm_command = command

    def _right_arm_command_callback(self, msg: JointState):
        """右臂命令回调"""
        with self._command_lock:
            command = {}
            for i, name in enumerate(msg.name):
                full_name = f"{"right_"}{name}"  # 添加前缀以区分左右臂关节
                if full_name in self._joint_name_to_key:
                    key = self._joint_name_to_key[full_name]
                    command[key] = msg.position[i] if i < len(msg.position) else 0.0
            
            self._right_arm_command = command

    def _mobile_base_command_callback(self, msg: Twist):
        """移动底座命令回调"""
        with self._command_lock:
            self._mobile_base_command = {
                "mobile_base.vx": msg.linear.x,
                "mobile_base.vy": msg.linear.y,
                "mobile_base.vtheta": msg.angular.z
            }

    @property
    def action_features(self) -> dict[str, type]:
        """返回动作特征字典"""
        if self.config.with_mobile_base:
            return {
                **dict.fromkeys(self.joints_dict.keys(), float),
                **dict.fromkeys(COBOTMAGIC_VEL.keys(), float),
            }
        else:
            return dict.fromkeys(self.joints_dict.keys(), float)

    @property
    def feedback_features(self) -> dict[str, type]:
        """返回反馈特征字典"""
        return {}

    @property
    def is_connected(self) -> bool:
        """检查是否连接到ROS并收到状态更新"""
        if not self._connected:
            return False
        
        # 检查最近是否收到状态更新
        current_time = time.time()
        timeout = self.config.connection_timeout  # 默认5秒
        
        with self._state_lock:
            # 检查左臂
            if self.config.with_l_arm:
                if (self._left_arm_state is None or 
                    current_time - self._left_arm_state.last_update > timeout):
                    return False
            
            # 检查右臂
            if self.config.with_r_arm:
                if (self._right_arm_state is None or 
                    current_time - self._right_arm_state.last_update > timeout):
                    return False
            
            # 检查移动底座
            if self.config.with_mobile_base:
                if (self._mobile_base_state is None or 
                    current_time - self._mobile_base_state.last_update > timeout):
                    return False
        
        return True

    def connect(self, calibrate: bool = True) -> None:
        """连接到ROS并初始化订阅器"""
        with self._connection_lock:
            if self._connected:
                logger.warning("Already connected")
                return
            
            try:
                # 初始化ROS节点
                self._init_ros_node()
                
                # 等待首次状态更新
                logger.info("Waiting for initial state updates...")
                start_time = time.time()
                timeout = self.config.connection_timeout
                
                while time.time() - start_time < timeout:
                    if self.is_connected:
                        break
                    time.sleep(0.1)
                
                if not self.is_connected:
                    raise ConnectionError("Failed to receive initial state updates within timeout")
                
                self._connected = True
                logger.info(f"{self.name} connected successfully")
                
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                self._connected = False
                raise

    @property
    def is_calibrated(self) -> bool:
        """检查是否已校准"""
        # 这里可以添加实际的校准检查逻辑
        return True

    def calibrate(self) -> None:
        """执行校准程序"""
        # 这里可以添加实际的校准逻辑
        logger.info("Calibration procedure would be implemented here")
        pass

    def configure(self) -> None:
        """配置机器人参数"""
        # 这里可以添加配置逻辑
        logger.info("Configuration procedure would be implemented here")
        pass

    def get_action(self) -> dict[str, float]:
        """获取当前动作状态，包括关节位置和移动底座速度"""
        start = time.perf_counter()
        
        # 检查连接状态
        if not self.is_connected:
            logger.warning("Not connected, cannot get action state")
            return {}
        
        joint_action = {}
        vel_action = {}
        
        try:
            # 获取关节动作（位置）
            if self.config.use_present_position:
                # 使用当前位置（从状态订阅获取）
                with self._state_lock:
                    # 左臂当前位置
                    if self.config.with_l_arm and self._left_arm_state:
                        joint_action.update(self._left_arm_state.positions)
                    
                    # 右臂当前位置
                    if self.config.with_r_arm and self._right_arm_state:
                        joint_action.update(self._right_arm_state.positions)
            else:
                # 使用目标位置（从命令订阅获取）
                with self._command_lock:
                    # 左臂目标位置
                    if self.config.with_l_arm and self._left_arm_command:
                        joint_action.update(self._left_arm_command)
                    
                    # 右臂目标位置
                    if self.config.with_r_arm and self._right_arm_command:
                        joint_action.update(self._right_arm_command)
            
            # 如果没有移动底座，直接返回关节动作
            if not self.config.with_mobile_base:
                dt_ms = (time.perf_counter() - start) * 1e3
                if dt_ms > 50.0:
                    logger.debug(f"{self.name} read joint action: {dt_ms:.1f}ms")
                return joint_action
            
            # 获取移动底座速度动作
            if self.config.use_present_position:
                # 使用当前实际速度（从状态订阅获取）
                with self._state_lock:
                    if self._mobile_base_state:
                        # 映射到标准的速度键
                        vel_action = {
                            "mobile_base.vx": self._mobile_base_state.linear.get("x", 0.0),
                            "mobile_base.vy": self._mobile_base_state.linear.get("y", 0.0),
                            "mobile_base.vtheta": self._mobile_base_state.angular.get("z", 0.0)
                        }
            else:
                # 使用最后命令速度（从命令订阅获取）
                with self._command_lock:
                    if self._mobile_base_command:
                        vel_action = self._mobile_base_command.copy()
            
            # 合并关节动作和速度动作
            result = {**joint_action, **vel_action}
            
            dt_ms = (time.perf_counter() - start) * 1e3
            if dt_ms > 50.0:
                logger.debug(f"{self.name} read action: {dt_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting action state: {e}")
            return {}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """发送反馈信息（未实现）"""
        raise NotImplementedError("Feedback sending not implemented for ROS-based teleoperator")

    def disconnect(self) -> None:
        """断开连接"""
        with self._connection_lock:
            if not self._connected:
                return
            
            # 关闭ROS节点
            try:
                rospy.signal_shutdown("Disconnecting teleoperator")
                self._node_initialized = False
            except:
                pass
            
            # 重置状态
            with self._state_lock:
                self._left_arm_state = None
                self._right_arm_state = None
                self._mobile_base_state = None
            
            with self._command_lock:
                self._left_arm_command = None
                self._right_arm_command = None
                self._mobile_base_command = None
            
            self._connected = False
            logger.info(f"{self.name} disconnected")

    def __del__(self):
        """析构函数，确保正确断开连接"""
        self.disconnect()
    
    @property
    def mobile_base_pose(self) -> Dict[str, float]:
        """获取移动底座的完整位姿信息（位置和方向）"""
        if not self._mobile_base_state:
            return {}
        
        with self._state_lock:
            return {
                "position_x": self._mobile_base_state.position.get("x", 0.0),
                "position_y": self._mobile_base_state.position.get("y", 0.0),
                "position_z": self._mobile_base_state.position.get("z", 0.0),
                "orientation_yaw": self._mobile_base_state.orientation.get("euler", {}).get("yaw", 0.0),
                "orientation_roll": self._mobile_base_state.orientation.get("euler", {}).get("roll", 0.0),
                "orientation_pitch": self._mobile_base_state.orientation.get("euler", {}).get("pitch", 0.0)
            }

    @property
    def mobile_base_velocity(self) -> Dict[str, float]:
        """获取移动底座的完整速度信息"""
        if not self._mobile_base_state:
            return {}
        
        with self._state_lock:
            return {
                "linear_x": self._mobile_base_state.linear_velocity.get("x", 0.0),
                "linear_y": self._mobile_base_state.linear_velocity.get("y", 0.0),
                "linear_z": self._mobile_base_state.linear_velocity.get("z", 0.0),
                "angular_x": self._mobile_base_state.angular_velocity.get("x", 0.0),
                "angular_y": self._mobile_base_state.angular_velocity.get("y", 0.0),
                "angular_z": self._mobile_base_state.angular_velocity.get("z", 0.0)
            }