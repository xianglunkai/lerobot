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

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode


@CameraConfig.register_subclass("webrtc")
@dataclass
class WebRTCCameraConfig(CameraConfig):
    """Configuration class for WebRTC-based camera streams.

    This class provides configuration options for cameras that receive video streams
    from a WebRTC server. It's designed to work with the qnbot_w_webrtc streaming
    server that provides camera feeds over WebRTC.

    Example configurations:
    ```python
    # Basic configuration for left wrist camera
    WebRTCCameraConfig(
        server_url="http://192.168.1.102:8080",
        camera_name="left_wrist",
        fps=30,
        width=640,
        height=480
    )

    # Configuration for head camera with different server
    WebRTCCameraConfig(
        server_url="http://192.168.1.100:8080", 
        camera_name="head",
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.BGR,
        timeout_ms=5000
    )
    ```

    Attributes:
        server_url: URL of the WebRTC streaming server (e.g., "http://192.168.1.102:8080")
        camera_name: Name of the camera stream to connect to (e.g., "left_wrist", "head", "right_wrist")
        fps: Requested frames per second (should match server configuration)
        width: Expected frame width in pixels
        height: Expected frame height in pixels
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        timeout_ms: Connection timeout in milliseconds. Defaults to 3000ms.
        buffer_size: Client-side frame buffer size. Defaults to 5.
        auto_reconnect: Whether to automatically reconnect on connection loss. Defaults to True.
    """

    server_url: str
    camera_name: str
    color_mode: ColorMode = ColorMode.RGB
    timeout_ms: int = 3000
    buffer_size: int = 5
    auto_reconnect: bool = True

    def __post_init__(self):
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` 应该是 {ColorMode.RGB.value} 或 {ColorMode.BGR.value}，但提供的是 {self.color_mode}"
            )

        if not self.server_url.startswith(("http://", "https://")):
            raise ValueError(f"`server_url` 必须以 http:// 或 https:// 开头，但提供的是: {self.server_url}")

        if self.camera_name not in ["left_wrist", "head", "right_wrist"]:
            print(f"警告: camera_name '{self.camera_name}' 不在标准相机列表中 ['left_wrist', 'head', 'right_wrist']")

        if self.timeout_ms <= 0:
            raise ValueError(f"`timeout_ms` 必须大于0，但提供的是: {self.timeout_ms}")

        if self.buffer_size <= 0:
            raise ValueError(f"`buffer_size` 必须大于0，但提供的是: {self.buffer_size}") 