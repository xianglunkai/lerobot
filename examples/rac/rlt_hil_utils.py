# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""RLT-oriented helpers for AgileX HIL data collection (schema + keyboard)."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.io_utils import write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.control_utils import is_headless
from lerobot.utils.recording_annotations import (
    EPISODE_FAILURE,
    EPISODE_SUCCESS,
    infer_collector_policy_version,
)

logger = logging.getLogger(__name__)

# RLT phase / source codes (compatible with Evo-RLT recording schema).
PHASE_PREFIX = 0.0
PHASE_CRITICAL = 1.0

SOURCE_VLA = 0.0
SOURCE_RL = 1.0
SOURCE_HUMAN = 2.0

INTERVENTION_STATE_POLICY = 0.0
INTERVENTION_STATE_ACTIVE = 1.0
INTERVENTION_STATE_RELEASE = 2.0

COLLECTOR_HUMAN = 0
COLLECTOR_POLICY = 1
COLLECTOR_RLT_ACTOR = 2

RLT_COLLECTOR_POLICY_ID_TO_NAME = {
    COLLECTOR_HUMAN: "human",
    COLLECTOR_POLICY: "policy",
    COLLECTOR_RLT_ACTOR: "rlt_actor",
}


def ensure_rlt_dataset_features(
    dataset_features: dict[str, dict],
    *,
    action_feature_names: list[str],
) -> None:
    """Add Evo-RLT-compatible complementary_info columns to dataset features."""
    names = list(action_feature_names)
    dataset_features["complementary_info.policy_action"] = {
        "dtype": "float32",
        "shape": (len(names),),
        "names": names,
    }
    dataset_features["complementary_info.is_intervention"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["is_intervention"],
    }
    dataset_features["complementary_info.state"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["state"],
    }
    dataset_features["complementary_info.phase"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["phase"],
    }
    dataset_features["complementary_info.collector_policy_id"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": ["collector_policy_id"],
    }


def write_rlt_schema_metadata(
    dataset: LeRobotDataset,
    *,
    policy_cfg: Any | None,
) -> None:
    """Persist collector codebook + schema version into dataset meta/info.json."""
    codebook = {
        str(COLLECTOR_HUMAN): "human",
        str(COLLECTOR_POLICY): infer_collector_policy_version(policy_cfg) if policy_cfg else "policy",
        str(COLLECTOR_RLT_ACTOR): "rlt_actor",
    }
    collector_info = dataset.meta.info["features"].get("complementary_info.collector_policy_id")
    if collector_info is not None:
        collector_info["info"] = {"codebook": codebook}
    dataset.meta.info["recording_schema_version"] = 2
    dataset.meta.info["rlt_recording"] = {
        "phase": {"0": "prefix", "1": "critical"},
        "collector_policy_id": codebook,
        "intervention_state": {
            "0": "policy",
            "1": "active",
            "2": "release",
        },
    }
    write_info(dataset.meta.info, Path(dataset.root))


def resolve_rlt_collector_policy_id(*, is_intervention: bool, source_type: float) -> int:
    if is_intervention or source_type == SOURCE_HUMAN:
        return COLLECTOR_HUMAN
    if source_type == SOURCE_RL:
        return COLLECTOR_RLT_ACTOR
    return COLLECTOR_POLICY


def policy_action_vector(
    policy_action: dict[str, Any] | None,
    action_names: list[str],
) -> np.ndarray:
    if policy_action is None:
        return np.zeros((len(action_names),), dtype=np.float32)
    return np.array([float(policy_action.get(name, 0.0)) for name in action_names], dtype=np.float32)


def annotate_rlt_frame(
    frame: dict[str, Any],
    *,
    dataset_features: dict[str, dict],
    is_intervention: bool,
    intervention_state: float,
    phase: float,
    collector_policy_id: int,
    policy_action: dict[str, Any] | None,
    action_names: list[str],
) -> None:
    """Mutate frame with RLT complementary_info fields when present in schema."""
    if "complementary_info.is_intervention" in dataset_features:
        frame["complementary_info.is_intervention"] = np.array(
            [1.0 if is_intervention else 0.0], dtype=np.float32
        )
    if "complementary_info.state" in dataset_features:
        frame["complementary_info.state"] = np.array([intervention_state], dtype=np.float32)
    if "complementary_info.phase" in dataset_features:
        frame["complementary_info.phase"] = np.array([phase], dtype=np.float32)
    if "complementary_info.collector_policy_id" in dataset_features:
        frame["complementary_info.collector_policy_id"] = np.array(
            [int(collector_policy_id)], dtype=np.int64
        )
    if "complementary_info.policy_action" in dataset_features:
        frame["complementary_info.policy_action"] = policy_action_vector(policy_action, action_names)


class CriticalPhaseTracker:
    """Track critical-segment recording window + outcome for --only-critical."""

    def __init__(self) -> None:
        self.is_active = False
        self.start_frame: int | None = None
        self.end_frame: int | None = None
        self.outcome: str | None = None

    def reset(self) -> None:
        self.is_active = False
        self.start_frame = None
        self.end_frame = None
        self.outcome = None

    def enter(self, frame_idx: int) -> None:
        self.is_active = True
        self.start_frame = frame_idx
        self.end_frame = None
        self.outcome = None
        logger.info("[RLT] Entered critical phase at frame %d", frame_idx)

    def mark_success(self, frame_idx: int) -> None:
        self.is_active = False
        self.end_frame = frame_idx
        self.outcome = EPISODE_SUCCESS
        logger.info("[RLT] Critical phase success at frame %d", frame_idx)

    def mark_failure(self, frame_idx: int) -> None:
        self.is_active = False
        self.end_frame = frame_idx
        self.outcome = EPISODE_FAILURE
        logger.info("[RLT] Critical phase failure at frame %d", frame_idx)


def init_rlt_keyboard_listener(
    *,
    teleop_toggle_key: str = "space",
    rlt_toggle_key: str = "r",
    double_tap_window_s: float = 0.6,
    only_critical: bool = False,
):
    """Keyboard controls aligned with Evo-RLT collect defaults (AgileX HIL)."""
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "intervention_active": False,
        "toggle_intervention": False,
        "toggle_critical_phase": False,
        "in_reset": False,
        "start_next_episode": False,
        "episode_outcome": None,
        "resume_policy": False,  # unused, kept for hil_utils.reset_loop compatibility
        "policy_paused": False,
        "correction_active": False,
    }

    if is_headless():
        logger.warning("Headless environment - keyboard controls unavailable")
        return None, events

    from pynput import keyboard

    teleop_name = teleop_toggle_key.lower()
    rlt_name = rlt_toggle_key.lower()
    last_rlt_tap_t = 0.0
    pending_single_tap = False
    pending_lock = threading.Lock()

    def _flush_single_tap_success():
        nonlocal pending_single_tap
        time.sleep(double_tap_window_s)
        with pending_lock:
            if not pending_single_tap:
                return
            pending_single_tap = False
        _handle_rlt_single_tap()

    def _handle_rlt_single_tap():
        if events["in_reset"]:
            events["rerecord_episode"] = False
            events["start_next_episode"] = True
            return
        if only_critical:
            events["toggle_critical_phase"] = True
        else:
            events["episode_outcome"] = EPISODE_SUCCESS
            events["exit_early"] = True
            logger.info("[RLT] Marked episode success")

    def _handle_rlt_double_tap():
        if events["in_reset"]:
            events["rerecord_episode"] = True
            events["start_next_episode"] = True
            return
        events["episode_outcome"] = EPISODE_FAILURE
        events["exit_early"] = True
        if only_critical:
            # Ensure critical tracker gets a failure mark on exit.
            events["toggle_critical_phase"] = False
        logger.info("[RLT] Marked episode failure (double tap)")

    def _on_rlt_key():
        nonlocal last_rlt_tap_t, pending_single_tap
        now = time.perf_counter()
        with pending_lock:
            if pending_single_tap and (now - last_rlt_tap_t) <= double_tap_window_s:
                pending_single_tap = False
                last_rlt_tap_t = now
                _handle_rlt_double_tap()
                return
            pending_single_tap = True
            last_rlt_tap_t = now
        threading.Thread(target=_flush_single_tap_success, daemon=True).start()

    def _key_name(key) -> str | None:
        if key == keyboard.Key.space:
            return "space"
        if key == keyboard.Key.enter:
            return "enter"
        if hasattr(key, "char") and key.char:
            return key.char.lower()
        return None

    def on_press(key):
        try:
            name = _key_name(key)
            if events["in_reset"]:
                if key in (keyboard.Key.space, keyboard.Key.right) or (name and name == teleop_name):
                    events["rerecord_episode"] = False
                    events["start_next_episode"] = True
                elif key == keyboard.Key.left:
                    events["rerecord_episode"] = True
                    events["start_next_episode"] = True
                elif name == rlt_name:
                    _on_rlt_key()
                elif key == keyboard.Key.esc:
                    events["stop_recording"] = True
                    events["start_next_episode"] = True
                return

            if name == teleop_name or (teleop_name == "space" and key == keyboard.Key.space):
                events["toggle_intervention"] = True
                return
            if name == rlt_name:
                _on_rlt_key()
                return
            if key == keyboard.Key.right:
                events["exit_early"] = True
                return
            if key == keyboard.Key.left:
                events["rerecord_episode"] = True
                events["exit_early"] = True
                return
            if key == keyboard.Key.esc:
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as exc:
            logger.info("Key error: %s", exc)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def start_rlt_pedal_listener(events: dict, *, double_tap_window_s: float = 0.6):
    """Optional foot-pedal outcome labels: single=success, double=failure."""
    try:
        from evdev import InputDevice, categorize, ecodes
    except ImportError:
        logger.warning("[Pedal] evdev not installed - pedal support disabled")
        return None

    pedal_device = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
    key_codes = {"KEY_A", "KEY_C"}
    last_tap_t = 0.0
    pending_single = False
    lock = threading.Lock()

    def _flush_success():
        nonlocal pending_single
        time.sleep(double_tap_window_s)
        with lock:
            if not pending_single:
                return
            pending_single = False
        if events.get("in_reset"):
            events["start_next_episode"] = True
            return
        events["episode_outcome"] = EPISODE_SUCCESS
        events["exit_early"] = True
        logger.info("[Pedal] Episode success")

    def pedal_reader():
        nonlocal last_tap_t, pending_single
        try:
            dev = InputDevice(pedal_device)
            logger.info("[Pedal] Connected: %s", dev.name)
            for ev in dev.read_loop():
                if ev.type != ecodes.EV_KEY:
                    continue
                key = categorize(ev)
                code = key.keycode
                if isinstance(code, (list, tuple)):
                    code = code[0]
                if key.keystate != 1 or code not in key_codes:
                    continue
                now = time.perf_counter()
                with lock:
                    if pending_single and (now - last_tap_t) <= double_tap_window_s:
                        pending_single = False
                        last_tap_t = now
                        events["episode_outcome"] = EPISODE_FAILURE
                        events["exit_early"] = True
                        logger.info("[Pedal] Episode failure (double tap)")
                        continue
                    pending_single = True
                    last_tap_t = now
                threading.Thread(target=_flush_success, daemon=True).start()
        except FileNotFoundError:
            logger.info("[Pedal] Device not found: %s", pedal_device)
        except PermissionError:
            logger.warning("[Pedal] Permission denied for %s", pedal_device)
        except Exception as exc:
            logger.warning("[Pedal] Error: %s", exc)

    thread = threading.Thread(target=pedal_reader, daemon=True)
    thread.start()
    return thread


def print_rlt_controls(*, only_critical: bool, rtc: bool) -> None:
    mode = "RLT AgileX HIL Collection" + (" (RTC)" if rtc else "")
    if only_critical:
        body = (
            f"{mode}\n"
            "  Controls (--only-critical):\n"
            "    space  - Toggle teleop intervention\n"
            "    r      - Enter critical / start recording; press again = success + end\n"
            "    r+r    - Failure + end (while in critical or as episode outcome)\n"
            "    ←      - Re-record current episode\n"
            "    →      - End episode early\n"
            "    Esc    - Stop collection"
        )
    else:
        body = (
            f"{mode}\n"
            "  Controls (full trajectory):\n"
            "    space  - Toggle teleop intervention (policy <-> human)\n"
            "    r      - Mark success and end episode\n"
            "    r+r    - Mark failure and end episode\n"
            "    ←      - Re-record current episode\n"
            "    →      - End episode early\n"
            "    Esc    - Stop collection"
        )
    logger.info("%s", body)
