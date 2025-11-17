# agilex_cobot_ros_manager.py
import logging
import threading
import time
import math
from collections import deque
from typing import Dict, Optional, Deque, Any

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

from scipy.spatial.transform import Rotation as R

from .config_agilex_cobot_ros_manager import AgilexCobotROSManagerConfig

logger = logging.getLogger(__name__)

# 配置常量
SUB_TOPIC_QUEUE_SIZE = 1000
SUB_MSG_QUEUE_SIZE = SUB_TOPIC_QUEUE_SIZE * 2
PUB_TOPIC_QUEUE_SIZE = 10


def quat_to_RPY(quat):
    """Convert quaternion to roll, pitch, yaw."""
    rot = R.from_quat([quat.x, quat.y, quat.z, quat.w])
    euler_angles = rot.as_euler("xyz", degrees=False)
    return euler_angles


def jpeg_mapping(img):
    """Compress and decompress image using JPEG."""
    img = cv2.imencode(".jpg", img)[1].tobytes()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    return img


class AgilexCobotROSManager:
    """
    Singleton ROS manager for Agilex Cobot robot.
    Manages all ROS communication in a thread-safe manner.
    """
    config_class = AgilexCobotROSManagerConfig
    _instance = None
    _lock = threading.Lock()
    name = "agilex_cobot_ros_manager"
    
    def __new__(cls, config: AgilexCobotROSManagerConfig | None):
        with cls._lock:
            if cls._instance is None:
                if config is None:
                    raise ValueError("Config must be provided for first initialization")
                cls._instance = super(AgilexCobotROSManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: AgilexCobotROSManagerConfig | None):
        if self._initialized:
            return
            
        self.config = config
        self._initialized = True
        self._ros_initialized = False
        self._init_ros_components()
        
    def _init_ros_components(self):
        """Initialize all ROS components."""
        if not rospy.get_node_uri():
            rospy.init_node(self.config.node_name, anonymous=True)
        
        self.cv_bridge = CvBridge()
        
        # Data storage
        self.slave_joint_state_lock = threading.Lock()
        self.slave_joint_positions = {}
        self.master_joint_state_lock = threading.Lock()
        self.master_joint_positions = {}
        # todo: add end effector pose storage if needed
        self.endpose_state_lock = threading.Lock()
        self.endpose_positions = {}
        # todo: add mobile base state storage if needed
        self.mobile_base_state_lock = threading.Lock()
        self.mobile_base_positions = {}
        
        self.camera_images = {}
        self.joint_state_timestamp = {}
        self.camera_images_timestamp = {}
        
        # Threading locks
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()  # Start locked
        
        # Deques for message buffering
        self._init_deques()
        
        # ROS publishers and subscribers
        self._init_subscribers()
        self._init_publishers()
        
        # Start ROS spinner in background thread
        self._start_ros_spinner()
        
        self._ros_initialized = True
        logger.info("AgilexCobot ROS Manager initialized")
    
    def _init_deques(self):
        """Initialize all message deques."""
      
        self.puppet_arm_left_deque = deque(maxlen=SUB_MSG_QUEUE_SIZE)
     
        self.puppet_arm_right_deque = deque(maxlen=SUB_MSG_QUEUE_SIZE)
          
        if hasattr(self.config, 'cam_high_topic') :
            self.img_front_deque = deque(maxlen=SUB_MSG_QUEUE_SIZE)
        if hasattr(self.config, 'cam_left_topic'):
            self.img_left_deque = deque(maxlen=SUB_MSG_QUEUE_SIZE)
        if hasattr(self.config, 'cam_right_topic'):
            self.img_right_deque = deque(maxlen=SUB_MSG_QUEUE_SIZE)
            
        if self.config.use_depth_image:
            self.img_left_depth_deque = deque(maxlen=SUB_MSG_QUEUE_SIZE)
            self.img_right_depth_deque = deque(maxlen=SUB_MSG_QUEUE_SIZE)
            self.img_front_depth_deque = deque(maxlen=SUB_MSG_QUEUE_SIZE)
            
        if self.config.with_mobile_base:
            self.robot_base_deque = deque(maxlen=SUB_MSG_QUEUE_SIZE)
            
        self.endpose_left_deque = deque(maxlen=SUB_MSG_QUEUE_SIZE)
        self.endpose_right_deque = deque(maxlen=SUB_MSG_QUEUE_SIZE)
    
    def _init_subscribers(self):
        """Initialize all ROS subscribers."""
        # Joint state subscribers
     
        self.left_joint_state_sub = rospy.Subscriber(
            self.config.left_joint_states_topic,
            JointState,
            self.left_joint_state_callback,
            queue_size=SUB_TOPIC_QUEUE_SIZE,
            tcp_nodelay=True,
        )
        
        self.left_joint_cmd_sub = rospy.Subscriber(
            self.config.left_arm_command_topic,
            JointState,
            self.left_joint_cmd_callback,
            queue_size=SUB_TOPIC_QUEUE_SIZE,
            tcp_nodelay=True,
        )
        
     
        self.right_joint_state_sub = rospy.Subscriber(
            self.config.right_joint_states_topic,
            JointState,
            self.right_joint_state_callback,
            queue_size=SUB_TOPIC_QUEUE_SIZE,
            tcp_nodelay=True,
        )
        
        self.right_joint_cmd_sub = rospy.Subscriber(
            self.config.right_arm_command_topic,
            JointState,
            self.right_joint_cmd_callback,
            queue_size=SUB_TOPIC_QUEUE_SIZE,
            tcp_nodelay=True,
        )
        
        # Camera subscribers
        if hasattr(self.config, 'cam_high_topic') :
            rospy.Subscriber(self.config.cam_high_topic, Image, self.img_high_img_callback, 
                           queue_size=SUB_TOPIC_QUEUE_SIZE, tcp_nodelay=True)
        if hasattr(self.config, 'cam_left_topic') :
            rospy.Subscriber(self.config.cam_left_topic, Image, self.img_left_img_callback, 
                           queue_size=SUB_TOPIC_QUEUE_SIZE, tcp_nodelay=True)
        if hasattr(self.config, 'cam_right_topic') :
            rospy.Subscriber(self.config.cam_right_topic, Image, self.img_right_img_callback, 
                           queue_size=SUB_TOPIC_QUEUE_SIZE, tcp_nodelay=True)
        
        # Depth image subscribers
        if self.config.use_depth_image:
            rospy.Subscriber(self.config.img_left_depth_topic, Image, self.img_left_depth_callback,
                           queue_size=SUB_TOPIC_QUEUE_SIZE, tcp_nodelay=True)
            rospy.Subscriber(self.config.img_right_depth_topic, Image, self.img_right_depth_callback,
                           queue_size=SUB_TOPIC_QUEUE_SIZE, tcp_nodelay=True)
            rospy.Subscriber(self.config.img_front_depth_topic, Image, self.img_front_depth_callback,
                           queue_size=SUB_TOPIC_QUEUE_SIZE, tcp_nodelay=True)
        
        # Other subscribers
        if self.config.with_mobile_base:
            rospy.Subscriber(self.config.mobile_base_state_topic, Odometry, 
                           self.robot_base_callback, queue_size=SUB_TOPIC_QUEUE_SIZE, tcp_nodelay=True)
            
        
        rospy.Subscriber(self.config.endpose_left_topic, PoseStamped, 
                           self.endpose_left_callback, queue_size=SUB_TOPIC_QUEUE_SIZE, tcp_nodelay=True)
        rospy.Subscriber(self.config.endpose_right_topic, PoseStamped, 
                           self.endpose_right_callback, queue_size=SUB_TOPIC_QUEUE_SIZE, tcp_nodelay=True)
    
    def _init_publishers(self):
        """Initialize all ROS publishers."""
    
        self.left_arm_pub = rospy.Publisher(
                self.config.left_arm_command_topic,
                JointState,
                queue_size=PUB_TOPIC_QUEUE_SIZE
            )
        
  
        self.right_arm_pub = rospy.Publisher(
                self.config.right_arm_command_topic,
                JointState,
                queue_size=PUB_TOPIC_QUEUE_SIZE
            )
            
        if self.config.with_mobile_base:
            self.robot_base_publisher = rospy.Publisher(
                self.config.mobile_command_topic, 
                Twist, 
                queue_size=PUB_TOPIC_QUEUE_SIZE
            )
    
    def _start_ros_spinner(self):
        """Start ROS spinner in background thread."""
        def spin_ros():
            try:
                rospy.spin()
            except Exception as e:
                logger.error(f"ROS spinner error: {e}")
        
        self.ros_thread = threading.Thread(target=spin_ros, daemon=True)
        self.ros_thread.start()
    
    # Callback methods
    def left_joint_state_callback(self, msg: JointState):
        """Callback for left arm joint state messages."""
        self.puppet_arm_left_deque.append(msg)
        self.joint_state_timestamp["left_joint"] = time.time()
        self._process_slave_joint_states(msg, "left_")
    
    def right_joint_state_callback(self, msg: JointState):
        """Callback for right arm joint state messages."""
        self.puppet_arm_right_deque.append(msg)
        self.joint_state_timestamp["right_joint"] = time.time()
        self._process_slave_joint_states(msg, "right_")
        
    def _process_slave_joint_states(self, msg: JointState, prefix: str):
        """通用处理关节状态回调"""
        with self.slave_joint_state_lock:
            for i, name in enumerate(msg.name):
                full_name = f"{prefix}{name}"  # 添加前缀以区分左右臂关节
                if i < len(msg.position):
                    self.slave_joint_positions[full_name] = msg.position[i]
          
    def get_slave_joint_states(self) -> Dict[str, float]:
        """Get current joint positions."""
        with self.slave_joint_state_lock:
            return self.slave_joint_positions.copy()   
        
    def left_joint_cmd_callback(self, msg: JointState):
        """Callback for left arm joint state messages."""
        self._process_master_joint_states(msg, "left_")
   
    
    def right_joint_cmd_callback(self, msg: JointState):
        """Callback for right arm joint state messages."""
        self._process_master_joint_states(msg, "right_")
    
    def _process_master_joint_states(self, msg: JointState, prefix: str):
        """通用处理关节状态回调"""
        with self.master_joint_state_lock:
            for i, name in enumerate(msg.name):
                full_name = f"{prefix}{name}"  # 添加前缀以区分左右臂关节
                if i < len(msg.position):
                    self.master_joint_positions[full_name] = msg.position[i]

    def get_master_joint_states(self) -> Dict[str, float]:
        """Get current joint positions."""
        with self.master_joint_state_lock:
            return self.master_joint_positions.copy()   
        
    
    def img_high_img_callback(self, msg):
        """Callback for front camera images."""
        self.img_front_deque.append(msg)
        self.camera_images_timestamp["high"] = time.time()
    
    def img_left_img_callback(self, msg):
        """Callback for left camera images."""
        self.img_left_deque.append(msg)
        self.camera_images_timestamp["left"] = time.time()
    
    def img_right_img_callback(self, msg):
        """Callback for right camera images."""
        self.img_right_deque.append(msg)
        self.camera_images_timestamp["right"] = time.time()
    
    def img_left_depth_callback(self, msg):
        """Callback for left depth images."""
        self.img_left_depth_deque.append(msg)
    
    def img_right_depth_callback(self, msg):
        """Callback for right depth images."""
        self.img_right_depth_deque.append(msg)
    
    def img_front_depth_callback(self, msg):
        """Callback for front depth images."""
        self.img_front_depth_deque.append(msg)
    
    def robot_base_callback(self, msg: Odometry):
        """Callback for mobile base state."""
        self.robot_base_deque.append(msg)
        with self.mobile_base_state_lock:
            self.mobile_base_positions['vx'] = msg.twist.twist.linear.x
            self.mobile_base_positions['vy'] = msg.twist.twist.linear.y
            self.mobile_base_positions['vtheta'] = msg.twist.twist.angular.z
    
    def get_robot_base_state(self) -> Dict[str, float]:
        """Get current mobile base state."""
        with self.mobile_base_state_lock:
            return self.mobile_base_positions.copy()
    
    def endpose_left_callback(self, msg):
        """Callback for left end effector pose."""
        self.endpose_left_deque.append(msg)
        with self.endpose_state_lock:
            pos = msg.pose.position
            ori = msg.pose.orientation
            rpy = quat_to_RPY(ori)
            self.endpose_positions['left'] = np.array([pos.x, pos.y, pos.z, rpy[0], rpy[1], rpy[2]])
  
    
    def endpose_right_callback(self, msg):
        """Callback for right end effector pose."""
        self.endpose_right_deque.append(msg)
        with self.endpose_state_lock:
            pos = msg.pose.position
            ori = msg.pose.orientation
            rpy = quat_to_RPY(ori)
            self.endpose_positions['right'] = np.array([pos.x, pos.y, pos.z, rpy[0], rpy[1], rpy[2]])
            
            
    def get_end_effector_poses(self) -> Dict[str, np.array]:
        """Get current end effector poses."""
        with self.endpose_state_lock:
            return self.endpose_positions.copy()

    
    def is_connected(self) -> bool:
        """Check if ROS connection is healthy."""
        if not self._ros_initialized:
            return False
            
        now = time.time()
        camera_images_ready = False
        joint_states_ready = False
        
        # Check timestamps
        cameras_time = np.array(list(self.camera_images_timestamp.values()))
        # Three cameras: high, left, right
        if len(cameras_time) == 3 and abs(max(cameras_time - now)) < self.config.connection_timeout:
            camera_images_ready = True

        # Two joint states: left and right
        joints_time = np.array(list(self.joint_state_timestamp.values()))
        # Two joints: left_joint, right_joint
        if len(joints_time) == 2 and abs(max(joints_time - now)) < self.config.connection_timeout:
            joint_states_ready = True
            
        return joint_states_ready and camera_images_ready
    
    def get_synchronized_observation(self) -> tuple[Dict[str, float], Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Get synchronized observation from all sensors.
        Returns motor positions, camera images, and end effector poses.
        """ 
        while True:
            result = self._get_synchronized_frame()
            if result:
                return self._process_observation_frame(*result)
            time.sleep(0.01)
    
    def _get_synchronized_frame(self):
        """Get synchronized frame from all sensors."""
        # Check if all required topics have data
        required_queues = [
            self.img_left_deque, self.img_right_deque, self.img_front_deque,
            self.puppet_arm_left_deque, self.puppet_arm_right_deque
        ]
        
        if any(len(q) == 0 for q in required_queues):
            return None
            
        if self.config.use_depth_image:
            depth_queues = [self.img_left_depth_deque, self.img_right_depth_deque, self.img_front_depth_deque]
            if any(len(q) == 0 for q in depth_queues):
                return None
                
        if self.config.with_mobile_base and len(self.robot_base_deque) == 0:
            return None
            
        # Find the minimum timestamp across all sensors
        frame_time = self._find_synchronized_frame_time()
        if frame_time is None:
            return None
            
        # Extract synchronized data
        return self._extract_synchronized_data(frame_time)
    
    def _find_synchronized_frame_time(self):
        """Find synchronized frame time across all sensors."""
        timestamps = [
            self.img_left_deque[-1].header.stamp.to_sec(),
            self.img_right_deque[-1].header.stamp.to_sec(),
            self.img_front_deque[-1].header.stamp.to_sec(),
            self.puppet_arm_left_deque[-1].header.stamp.to_sec(),
            self.puppet_arm_right_deque[-1].header.stamp.to_sec(),
        ]
        
        if self.config.use_depth_image:
            timestamps.extend([
                self.img_left_depth_deque[-1].header.stamp.to_sec(),
                self.img_right_depth_deque[-1].header.stamp.to_sec(),
                self.img_front_depth_deque[-1].header.stamp.to_sec(),
            ])
            
        if self.config.with_mobile_base:
            timestamps.append(self.robot_base_deque[-1].header.stamp.to_sec())
            
        return min(timestamps)
    
    def _extract_synchronized_data(self, frame_time):
        """Extract synchronized data for given frame time."""
        # Get images
        img_left = self._get_image_from_deque(self.img_left_deque, frame_time)
        img_right = self._get_image_from_deque(self.img_right_deque, frame_time)
        img_front = self._get_image_from_deque(self.img_front_deque, frame_time)
        
        # Get joint states
        puppet_arm_left = self._get_joint_state_from_deque(self.puppet_arm_left_deque, frame_time)
        puppet_arm_right = self._get_joint_state_from_deque(self.puppet_arm_right_deque, frame_time)
  
        
        # Get end effector poses
        endpose_left = self._get_endpose_from_deque(self.endpose_left_deque, frame_time)
        endpose_right = self._get_endpose_from_deque(self.endpose_right_deque, frame_time)
        
        # Get depth images
        img_left_depth, img_right_depth, img_front_depth = None, None, None
        if self.config.use_depth_image:
            img_left_depth = self._get_image_from_deque(self.img_left_depth_deque, frame_time)
            img_right_depth = self._get_image_from_deque(self.img_right_depth_deque, frame_time)
            img_front_depth = self._get_image_from_deque(self.img_front_depth_deque, frame_time)
        
        # Get mobile base state
        robot_base = None
        if self.config.with_mobile_base:
            robot_base = self._get_robot_base_from_deque(self.robot_base_deque, frame_time)
        
        return (
            img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
            puppet_arm_left, puppet_arm_right, endpose_left, endpose_right, robot_base
        )
    
    def _get_image_from_deque(self, deque_obj, frame_time):
        """Get image from deque synchronized to frame time."""
        while len(deque_obj) > 1 and deque_obj[0].header.stamp.to_sec() < frame_time:
            deque_obj.popleft()
        msg = deque_obj[0]
        deque_obj.popleft()
        return self.cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
    
    def _get_joint_state_from_deque(self, deque_obj, frame_time):
        """Get joint state from deque synchronized to frame time."""
        while len(deque_obj) > 1 and deque_obj[0].header.stamp.to_sec() < frame_time:
            deque_obj.popleft()
        msg = deque_obj[0]
        deque_obj.popleft()
        return msg
    
    def _get_endpose_from_deque(self, deque_obj, frame_time):
        """Get end effector pose from deque synchronized to frame time."""
        if len(deque_obj) == 0:
            return None
        while len(deque_obj) > 1 and deque_obj[0].header.stamp.to_sec() < frame_time:
            deque_obj.popleft()
        msg = deque_obj[0]
        deque_obj.popleft()
        return msg
    
    def _get_robot_base_from_deque(self, deque_obj, frame_time):
        """Get robot base state from deque synchronized to frame time."""
        while len(deque_obj) > 1 and deque_obj[0].header.stamp.to_sec() < frame_time:
            deque_obj.popleft()
        msg = deque_obj[0]
        deque_obj.popleft()
        return msg
    
    def _process_observation_frame(self, img_front, img_left, img_right, img_front_depth, 
                                 img_left_depth, img_right_depth, puppet_arm_left, 
                                 puppet_arm_right, endpose_left, endpose_right, robot_base):
        """Process observation frame into structured data."""
        motor_position = {}
        cameras = {}
        endpose = {}
        
        # Process joint positions
        for i, name in enumerate(puppet_arm_left.name):
            full_name = f"left_{name}"
            if i < len(puppet_arm_left.position):
                motor_position[full_name] = puppet_arm_left.position[i]
        
        for i, name in enumerate(puppet_arm_right.name):
            full_name = f"right_{name}"
            if i < len(puppet_arm_right.position):
                motor_position[full_name] = puppet_arm_right.position[i]
        
        # Process mobile base
        if self.config.with_mobile_base and robot_base:
            motor_position["vx"] = robot_base.twist.twist.linear.x
            motor_position["vy"] = robot_base.twist.twist.linear.y
            motor_position["vtheta"] = robot_base.twist.twist.angular.z
        
        # Process camera images
        cameras["high"] = jpeg_mapping(img_front)
        cameras["left"] = jpeg_mapping(img_left)
        cameras["right"] = jpeg_mapping(img_right)
        
        # Process end effector poses
        if endpose_left:
            left_pos = endpose_left.pose.position
            left_rpy = quat_to_RPY(endpose_left.pose.orientation)
            left_gripper = puppet_arm_left.position[-1] if puppet_arm_left.position else 0.0
            endpose["left"] = np.array([left_pos.x, left_pos.y, left_pos.z, left_rpy[0], 
                                      left_rpy[1], left_rpy[2], left_gripper])
        
        if endpose_right:
            right_pos = endpose_right.pose.position
            right_rpy = quat_to_RPY(endpose_right.pose.orientation)
            right_gripper = puppet_arm_right.position[-1] if puppet_arm_right.position else 0.0
            endpose["right"] = np.array([right_pos.x, right_pos.y, right_pos.z, right_rpy[0], 
                                       right_rpy[1], right_rpy[2], right_gripper])
        
        return motor_position, cameras, endpose
    
    # Command methods
    def publish_left_arm_command(self, positions: list[float]):
        """Publish command to left arm."""
        if len(positions) != 7:
            logger.error(f"Left arm command requires 7 values, got {len(positions)}")
            return
        
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        msg.position = positions
        self.left_arm_pub.publish(msg)
    
    def publish_right_arm_command(self, positions: list[float]):
        """Publish command to right arm."""
        if len(positions) != 7:
            logger.error(f"Right arm command requires 7 values, got {len(positions)}")
            return
        
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        msg.position = positions
        self.right_arm_pub.publish(msg)
    
    def publish_mobile_base_command(self, vel: list[float]):
        """Publish command to mobile base."""
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = vel[1]
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[2]
        self.robot_base_publisher.publish(vel_msg)
    
    def publish_continuous_arm_commands(self, left_target, right_target):
        """Publish smooth continuous arm commands with interpolation."""
        rate = rospy.Rate(self.config.control_rate)
        
        # Get current positions
        left_current = self._get_current_arm_position('left')
        right_current = self._get_current_arm_position('right')
        enable_left = True
        if left_target is None:
            left_target = left_current
            enable_left = False
        
        enable_right = True
        if right_target is None:
            right_target = right_current
            enable_right = False
        
        if left_current is None or right_current is None:
            logger.error("Cannot get current arm positions")
            return
        
        # Calculate movement directions
        left_directions = [1 if left_target[i] - left_current[i] > 0 else -1 
                          for i in range(len(left_target))]
        right_directions = [1 if right_target[i] - right_current[i] > 0 else -1 
                           for i in range(len(right_target))]
        
        # Interpolate to target positions
        step = 0
        while not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            
            if enable_left:
                left_finished = self._interpolate_arm_position(left_current, left_target, 
                                                         left_directions, 'left')
            else:
                left_finished = True
            
            if enable_right:
                right_finished = self._interpolate_arm_position(right_current, right_target, 
                                                          right_directions, 'right')
            else:
                right_finished = True
            
            if left_finished and right_finished:
                break
                
            step += 1
            logger.debug(f"Continuous arm command step: {step}")
            rate.sleep()
    
    def _get_current_arm_position(self, arm_side):
        """Get current arm position from deque."""
        deque_obj = getattr(self, f'puppet_arm_{arm_side}_deque')
        if len(deque_obj) == 0:
            return None
        return list(deque_obj[-1].position)
    
    def _interpolate_arm_position(self, current, target, directions, arm_side):
        """Interpolate arm position towards target."""
        finished = True
        for i in range(len(current)):
            diff = abs(target[i] - current[i])
            if diff < self.config.arm_steps_length[i]:
                current[i] = target[i]
            else:
                current[i] += directions[i] * self.config.arm_steps_length[i]
                finished = False
        
        # Publish command
        publisher = getattr(self, f'{arm_side}_arm_pub')
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        msg.position = current
        publisher.publish(msg)
        
        return finished
    
    def shutdown(self):
        """Shutdown ROS manager."""
        if self._ros_initialized:
            rospy.signal_shutdown("AgilexCobot ROS Manager shutdown")
            self._ros_initialized = False
            logger.info("AgilexCobot ROS Manager shutdown")