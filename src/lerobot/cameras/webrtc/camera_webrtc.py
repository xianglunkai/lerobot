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

"""
提供 WebRTCCamera 类，用于从 WebRTC 服务器接收视频流。
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List

import aiohttp
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from .configuration_webrtc import WebRTCCameraConfig

logger = logging.getLogger(__name__)


class WebRTCVideoReceiver:
    """WebRTC 视频接收器，处理视频流的接收和缓冲"""

    def __init__(self, buffer_size=5):
        self.frame_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_event = threading.Event()
        self.frame_count = 0

    def on_frame_received(self, frame):
        """当接收到新帧时调用"""
        with self.lock:
            # 将 aiortc VideoFrame 转换为 numpy array
            if hasattr(frame, 'to_ndarray'):
                # aiortc VideoFrame
                np_frame = frame.to_ndarray(format='bgr24')
            else:
                # 如果已经是 numpy array
                np_frame = frame

            self.frame_buffer.append(np_frame)
            self.latest_frame = np_frame
            self.frame_count += 1

        # 通知等待的线程有新帧可用
        self.frame_event.set()

    def get_latest_frame(self):
        """获取最新的帧"""
        with self.lock:
            if self.frame_buffer:
                return self.frame_buffer[-1].copy()
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def wait_for_frame(self, timeout_ms=200):
        """等待新帧，返回是否有新帧可用"""
        return self.frame_event.wait(timeout_ms / 1000.0)

    def clear_event(self):
        """清除帧事件"""
        self.frame_event.clear()


class WebRTCCamera(Camera):
    """
    从 WebRTC 服务器接收视频流的相机类。

    这个类连接到 qnbot_w_webrtc 服务器，接收指定相机的视频流。
    支持三个相机：left_wrist, head, right_wrist。

    示例用法：
    ```python
    from lerobot.cameras.webrtc import WebRTCCamera
    from lerobot.cameras.webrtc.configuration_webrtc import WebRTCCameraConfig

    # 配置 WebRTC 相机
    config = WebRTCCameraConfig(
        server_url="http://192.168.1.102:8080",
        camera_name="left_wrist", 
        fps=30,
        width=640,
        height=480
    )
    
    camera = WebRTCCamera(config)
    camera.connect()
    
    # 读取帧
    frame = camera.read()
    print(f"接收到帧尺寸: {frame.shape}")
    
    # 异步读取
    async_frame = camera.async_read()
    
    # 断开连接
    camera.disconnect()
    ```
    """

    def __init__(self, config: WebRTCCameraConfig):
        """
        初始化 WebRTC 相机。

        Args:
            config: WebRTC 相机配置
        """
        super().__init__(config)
        
        self.config = config
        self.server_url = config.server_url.rstrip('/')
        self.camera_name = config.camera_name
        self.color_mode = config.color_mode
        self.timeout_ms = config.timeout_ms
        self.auto_reconnect = config.auto_reconnect

        # WebRTC 连接组件
        self.pc: RTCPeerConnection | None = None
        self.video_receiver = WebRTCVideoReceiver(config.buffer_size)
        self._connected = False
        self._connecting = False

        # 异步事件循环
        self.loop: asyncio.AbstractEventLoop | None = None
        self.loop_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        logger.info(f"WebRTC相机初始化: {self.camera_name} @ {self.server_url}")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.camera_name}@{self.server_url})"

    @property
    def is_connected(self) -> bool:
        """检查相机是否已连接"""
        # 详细的连接状态检查
        if not self._connected:
            logger.debug(f"相机 {self.camera_name} 内部连接标志为 False")
            return False
        if self.pc is None:
            logger.debug(f"相机 {self.camera_name} RTCPeerConnection 为 None")
            return False
        logger.debug(f"相机 {self.camera_name} WebRTC状态: {self.pc.connectionState}")
        return self._connected and self.pc is not None and self.pc.connectionState in ['connected', 'connecting']

    @staticmethod
    def find_cameras() -> List[Dict[str, Any]]:
        """
        发现可用的 WebRTC 相机流。
        
        返回标准的三个相机配置示例。
        """
        return [
            {
                "index": "left_wrist",
                "name": "Left Wrist WebRTC Camera",
                "config": {
                    "server_url": "http://192.168.1.102:8080",
                    "camera_name": "left_wrist"
                }
            },
            {
                "index": "head", 
                "name": "Head WebRTC Camera",
                "config": {
                    "server_url": "http://192.168.1.102:8080",
                    "camera_name": "head"
                }
            },
            {
                "index": "right_wrist",
                "name": "Right Wrist WebRTC Camera", 
                "config": {
                    "server_url": "http://192.168.1.102:8080",
                    "camera_name": "right_wrist"
                }
            }
        ]

    def connect(self, warmup: bool = True) -> None:
        """
        连接到 WebRTC 服务器。

        Args:
            warmup: 是否在连接后等待第一帧，用于确保连接稳定

        Raises:
            DeviceAlreadyConnectedError: 如果相机已经连接
            ConnectionError: 如果连接失败
        """
        if self._connected or self._connecting:
            raise DeviceAlreadyConnectedError(f"{self} 已经连接或正在连接中")

        self._connecting = True
        logger.info(f"正在连接到 WebRTC 服务器: {self}")

        try:
            # 启动异步事件循环
            self._start_event_loop()
            
            # 建立 WebRTC 连接
            future = asyncio.run_coroutine_threadsafe(self._establish_connection(), self.loop)
            future.result(timeout=self.timeout_ms / 1000.0)
            
            self._connected = True
            logger.info(f"WebRTC 连接建立成功: {self}")

            # Warmup: 等待第一帧
            if warmup:
                start_time = time.time()
                while time.time() - start_time < 5.0:  # 最多等待5秒
                    if self.video_receiver.latest_frame is not None:
                        logger.info(f"Warmup 完成，收到第一帧: {self}")
                        break
                    time.sleep(0.1)
                else:
                    logger.warning(f"Warmup 超时，未收到第一帧: {self}")

        except Exception as e:
            self._connecting = False
            self._connected = False
            self._stop_event_loop()
            raise ConnectionError(f"WebRTC 连接失败: {e}")
        finally:
            self._connecting = False

    def _start_event_loop(self):
        """启动异步事件循环线程"""
        if self.loop_thread is not None:
            return

        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_forever()
            finally:
                self.loop.close()

        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        
        # 等待事件循环启动
        while self.loop is None:
            time.sleep(0.001)

    def _stop_event_loop(self):
        """停止异步事件循环"""
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.loop_thread is not None:
                self.loop_thread.join(timeout=2.0)
            self.loop = None
            self.loop_thread = None

    async def _establish_connection(self):
        """建立 WebRTC 连接"""
        # 创建 RTCPeerConnection
        self.pc = RTCPeerConnection()

        @self.pc.on("track")
        def on_track(track):
            logger.info(f"收到轨道: {track.kind}")
            if track.kind == "video":
                # 创建帧接收任务
                asyncio.create_task(self._receive_frames(track))

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"WebRTC 连接状态变化: {self.pc.connectionState}")

        # 创建 offer - 使用兼容服务器端的方式
        # 注意：aiortc 的 createOffer 虽然不直接支持 offerToReceiveVideo 参数，
        # 但我们可以先添加 transceiver 然后创建 offer 来实现相同效果
        self.pc.addTransceiver("video", direction="recvonly")
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # 发送 offer 到服务器
        offer_data = {
            "sdp": self.pc.localDescription.sdp,
            "type": self.pc.localDescription.type
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_url}/offer/{self.camera_name}",
                json=offer_data,
                timeout=aiohttp.ClientTimeout(total=self.timeout_ms / 1000.0)
            ) as response:
                if response.status != 200:
                    raise ConnectionError(f"服务器返回错误: {response.status}")
                
                answer_data = await response.json()
                answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
                await self.pc.setRemoteDescription(answer)

    async def _receive_frames(self, track):
        """接收视频帧"""
        logger.info(f"开始接收视频帧: {self.camera_name}")
        try:
            while True:
                frame = await track.recv()
                self.video_receiver.on_frame_received(frame)
        except Exception as e:
            logger.error(f"接收帧时出错: {e}")

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        同步读取一帧。

        Args:
            color_mode: 颜色模式，如果为 None 则使用配置的默认模式

        Returns:
            np.ndarray: 捕获的帧

        Raises:
            DeviceNotConnectedError: 如果相机未连接
            RuntimeError: 如果无法获取帧
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        frame = self.video_receiver.get_latest_frame()
        if frame is None:
            # 等待新帧
            if self.video_receiver.wait_for_frame(self.timeout_ms):
                self.video_receiver.clear_event()
                frame = self.video_receiver.get_latest_frame()
            
            if frame is None:
                raise RuntimeError(f"无法从 {self} 获取帧")

        return self._postprocess_frame(frame, color_mode)

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """
        异步读取一帧。

        Args:
            timeout_ms: 超时时间（毫秒）

        Returns:
            np.ndarray: 捕获的帧

        Raises:
            DeviceNotConnectedError: 如果相机未连接
            RuntimeError: 如果超时或无法获取帧
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        if self.video_receiver.wait_for_frame(timeout_ms):
            self.video_receiver.clear_event()
            frame = self.video_receiver.get_latest_frame()
            if frame is not None:
                return self._postprocess_frame(frame)

        raise RuntimeError(f"异步读取超时: {self}")

    def _postprocess_frame(self, frame: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        后处理帧：颜色模式转换和尺寸调整

        Args:
            frame: 输入帧 (BGR格式)
            color_mode: 目标颜色模式

        Returns:
            np.ndarray: 处理后的帧
        """
        if color_mode is None:
            color_mode = self.color_mode

        # 调整尺寸（如果需要）
        if self.height and self.width and frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # 颜色模式转换
        if color_mode == ColorMode.RGB:
            # 从 BGR 转换为 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 如果是 BGR，不需要转换，因为默认就是 BGR

        return frame

    def disconnect(self) -> None:
        """
        断开 WebRTC 连接并释放资源。
        """
        if not self._connected:
            logger.warning(f"{self} 已经断开或未连接")
            return

        logger.info(f"正在断开 WebRTC 连接: {self}")

        # 关闭 WebRTC 连接
        if self.pc is not None:
            if self.loop is not None:
                future = asyncio.run_coroutine_threadsafe(self._close_connection(), self.loop)
                try:
                    future.result(timeout=2.0)
                except:
                    logger.warning("关闭 WebRTC 连接超时")

        # 停止事件循环
        self._stop_event_loop()

        self._connected = False
        self.pc = None
        logger.info(f"WebRTC 连接已断开: {self}")

    async def _close_connection(self):
        """异步关闭 WebRTC 连接"""
        if self.pc is not None:
            try:
                # 通知服务器关闭连接
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"{self.server_url}/close/{self.camera_name}",
                        timeout=aiohttp.ClientTimeout(total=1.0)
                    )
            except:
                pass  # 忽略关闭通知的错误

            await self.pc.close() 