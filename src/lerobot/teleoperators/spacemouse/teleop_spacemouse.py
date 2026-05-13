# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import sys
from enum import IntEnum
from turtle import home
from typing import Any

import numpy as np
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from .configuration_spacemouse import SpacemouseTeleopConfig

import pyspacemouse
import threading
import time 
import math

class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


class SpacemouseTeleop(Teleoperator):
    """
    Teleop class to use spacemouse inputs for control.
    """

    config_class = SpacemouseTeleopConfig

    name = "spacemouse"

    def __init__(self, config: SpacemouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self._connected = False
        # Underlying pyspacemouse device object (returned by pyspacemouse.open())
        self._device = None
        # Gripper toggle state: assume starts OPEN
        self._gripper_state: int = GripperAction.STAY.value
        self._prev_button_state: int = 0

        # Background reader thread vars (used to keep only the latest state)
        self._latest_state = None  # will store the most recent raw state coming from the driver
        self._reader_thread = None
        self._stop_reader = False
        self.state_lock = threading.Lock()
        self._last_call_ms = None
    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (7,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "delta_roll": 3, "delta_pitch": 4, "delta_yaw": 5, "gripper": 6},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (6,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "delta_roll": 3, "delta_pitch": 4, "delta_yaw": 5},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        """Connect to the SpaceMouse device (real or mock)."""
        # pyspacemouse.open() returns a device object (which may also be used
        # as a context manager). Preserve the object so we can call its
        # read()/close() methods instead of relying on any module-level helpers.
        try:
            self._device = pyspacemouse.open()
        except Exception:
            # open failed; ensure device stays None
            self._device = None

        self._connected = self._device is not None

        # Start background reader to avoid piling up driver messages (reduces perceived latency)
        if self._connected:

            def _reader_loop():
                """Continuously poll the driver so its internal queue stays empty.

                We only keep the most recent state which `get_action` then consumes. This
                prevents buildup when the driver polls faster than the main control loop
                (e.g. teleoperate.py's ≈60 Hz loop vs. SpaceMouse ≈125 Hz updates).
                """
                while not self._stop_reader:
                    try:
                        # Prefer the device instance's read() method.
                        if self._device is None:
                            break
                        with self.state_lock:
                            self._latest_state = self._device.read()
                    except Exception:
                        # In case device is unplugged mid-run; exit thread gracefully
                        break
                    time.sleep(0.005)  # Sleep a little: driver typically updates ~100-200Hz; reduce CPU by a short sleep

            self._stop_reader = False
            self._reader_thread = threading.Thread(target=_reader_loop, daemon=True)
            self._reader_thread.start()
    @check_if_not_connected
    def get_action(self) -> RobotAction:
        if not self.is_connected():
            raise RuntimeError("SpaceMouse is not connected. Call connect() first.")

        # Prefer the state produced by the background reader (most recent),
        # fall back to direct read if thread hasn't produced anything yet.
        with self.state_lock:
            if self._latest_state is not None:
                state = self._latest_state
            else:
                # Fallback to device.read() when available. Avoid using module-level
                # read() which some pyspacemouse distributions do not expose.
                if self._device is not None and hasattr(self._device, "read"):
                    state = self._device.read()
                else:
                    raise RuntimeError("SpaceMouse device not ready and no read() method available")

        deltas = [
            state.y,
            -state.x,
            state.z,
            state.roll,
            state.pitch,
            -state.yaw,
        ]
        
        # clamp deltas to [-1, 1] just in case (pyspacemouse docs say values are already normalized but we add this as a safety measure)
        deltas = np.array([max(-1.0, min(1.0, d)) for d in deltas])

        # convert deltas to spacemouse_action
        spacemouse_action = np.array(deltas, dtype=np.float32)
        
        # apply cutoff frequency (simple low-pass filter); alpha uses measured interval when available
        rc = 1.0 / (2 * math.pi * self.config.eef_cutoff_freq)
        now = time.perf_counter()
        if not hasattr(self, "_prev_action"):
            self._prev_action = np.zeros_like(spacemouse_action)
        if self._last_call_ms is not None:
            take_time = now - self._last_call_ms
            if take_time > 1.0:
                self._last_call_ms = None
            else:
                dt_lp = min(max(take_time, 1e-3), 2./self.config.fps) # 2/fps is the maximum time step for consistent behaviour when using fixed-dt helpers
                # dt_lp = 1/self.config.fps
                alpha = dt_lp / (dt_lp + rc)
                spacemouse_action = alpha * spacemouse_action.copy() + (1 - alpha) * self._prev_action.copy()
        
        self._prev_action = spacemouse_action.copy() 
        
        # apply deadzone & scaling
        for i, axis in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
            step_size = self.config.end_effector_step_sizes.get(axis, 0.01)  # Default to 0.01 if not specified
            delta_scaled   = spacemouse_action[i] * step_size
            
            if abs(delta_scaled) < self.config.deadzone:
                delta_scaled = 0

            spacemouse_action[i] = delta_scaled

        action_dict = {
            "delta_x": spacemouse_action[0],
            "delta_y": spacemouse_action[1],
            "delta_z": spacemouse_action[2],
            "delta_roll": spacemouse_action[3],
            "delta_pitch": spacemouse_action[4],
            "delta_yaw": spacemouse_action[5],
            "home": False,
        }

        # Simple gripper control: right button (index 1) toggles open/close.
        # Assumption: the physical gripper starts in the OPEN state when the teleop script boots.
        # Each button press switches the command between OPEN and CLOSE accordingly.
       
        if self.config.use_gripper and hasattr(state, "buttons") and len(state.buttons) >= 2:

            if self._last_call_ms is None:
                self._gripper_state=  GripperAction.STAY.value
            else:
                # Normalize button value to boolean (some drivers return 0/1, others may return truthy values)
                try:
                    current_btn = bool(state.buttons[1])
                except Exception:
                    current_btn = False

                # Toggle on rising edge: pressed now but was not pressed previously
                if current_btn and not bool(self._prev_button_state):
                    self._gripper_state = (
                        GripperAction.CLOSE.value
                        if self._gripper_state == GripperAction.OPEN.value
                        else GripperAction.OPEN.value
                    )

                # Always update prev state to the normalized boolean
                self._prev_button_state = current_btn
                
            action_dict["gripper"] = self._gripper_state

        # Home (left) button: independent of use_gripper so reset still works without gripper mapping.
        if hasattr(state, "buttons") and len(state.buttons) >= 2:
            try:
                home_btn = bool(state.buttons[0])
            except Exception:
                home_btn = False
            action_dict["home"] = home_btn

        self._last_call_ms = now
      
        return action_dict

    def disconnect(self) -> None:
        """Disconnect from the spacemouse."""
        # pyspacemouse does not expose an explicit close API but we reset connection flag
        self._connected = False

        # Stop reader thread if running
        self._stop_reader = True
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=0.1)
            self._reader_thread = None

    def is_connected(self) -> bool:
        """Check if spacemouse is connected."""
        return self._connected

    def calibrate(self) -> None:
        """Calibrate the spacemouse."""
        # No calibration needed for spacemouse
        pass

    def is_calibrated(self) -> bool:
        """Check if spacemouse is calibrated."""
        # Spacemouse doesn't require calibration
        return True

    def configure(self) -> None:
        """Configure the spacemouse."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the spacemouse."""
        # Spacemouse doesn't support feedback
        pass