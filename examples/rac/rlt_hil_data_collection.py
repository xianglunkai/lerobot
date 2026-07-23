#!/usr/bin/env python

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

"""
RLT-schema AgileX HIL data collection.

New entrypoint (does not modify examples/rac/hil_data_collection.py).

Records Evo-RLT-compatible complementary_info fields and uses RLT-style controls:
  - space: toggle teleop intervention
  - r / r+r: success / failure (or critical enter/exit with --only-critical)
  - ←: rerecord, Esc: stop

Usage (AgileX + RTC example):
    python examples/rac/rlt_hil_data_collection.py \\
        --policy.path=/path/to/pretrained_model \\
        --robot.type=agilex_cobot \\
        --teleop.type=spacemouse \\
        --dataset.repo_id=lerobot-data-collection/rlt_hil_$(date +%Y%m%d_%H%M%S) \\
        --dataset.single_task="Please sort and return the silver screws..." \\
        --enable_episode_outcome_labeling=true \\
        --rtc.enabled=true
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pprint import pformat
from threading import Event, Lock, Thread
from typing import Any

import torch
from hil_data_collection import (
    ACPInferenceConfig,
    ThreadSafeRobot,
    _resolve_action_key_order,
    _resolve_state_joint_order,
    _rtc_inference_thread,
    _set_openarm_max_relative_target_if_missing,
)
from hil_utils import (
    HILDatasetConfig,
    make_identity_processors,
    reset_loop,
    teleop_disable_torque,
    teleop_smooth_move_to,
)
from rlt_hil_utils import (
    CriticalPhaseTracker,
    INTERVENTION_STATE_ACTIVE,
    INTERVENTION_STATE_POLICY,
    PHASE_CRITICAL,
    PHASE_PREFIX,
    SOURCE_HUMAN,
    SOURCE_VLA,
    annotate_rlt_frame,
    ensure_rlt_dataset_features,
    init_rlt_keyboard_listener,
    print_rlt_controls,
    resolve_rlt_collector_policy_id,
    start_rlt_pedal_listener,
    write_rlt_schema_metadata,
)

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import get_policy_class, make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.action_interpolator import ActionInterpolator
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import make_robot_action
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.robots.agilex_cobot.config_agilex_cobot import AgilexCobotROSManagerConfig  # noqa: F401
from lerobot.robots.bi_openarm_follower.config_bi_openarm_follower import BiOpenArmFollowerConfig  # noqa: F401
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig  # noqa: F401
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig  # noqa: F401
from lerobot.teleoperators.openarm_mini.config_openarm_mini import OpenArmMiniConfig  # noqa: F401
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig  # noqa: F401
from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpacemouseTeleopConfig  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import is_headless, predict_action
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.recording_annotations import (
    EPISODE_FAILURE,
    EPISODE_SUCCESS,
    normalize_episode_success_label,
    resolve_episode_success_label,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)


@dataclass
class RLTHILConfig:
    """HIL config with RLT recording schema / interaction options for AgileX."""

    robot: RobotConfig
    teleop: TeleoperatorConfig
    dataset: HILDatasetConfig
    policy: PreTrainedConfig | None = None
    rtc: RTCConfig = field(default_factory=RTCConfig)
    interpolation_multiplier: int = 2
    record_interpolated_actions: bool = False
    display_data: bool = True
    play_sounds: bool = True
    resume: bool = False
    device: str = "cuda"
    use_torch_compile: bool = False
    compile_warmup_inferences: int = 2
    calibrate: bool = False
    log_hz: bool = True
    hz_log_interval_s: float = 2.0
    action_queue_size_to_get_new_actions: int = 30

    enable_episode_outcome_labeling: bool = True
    default_episode_success: str | None = None
    require_episode_success_label: bool = True

    # RLT interaction
    teleop_toggle_key: str = "space"
    rlt_toggle_key: str = "r"
    double_tap_window_s: float = 0.6
    only_critical: bool = False
    start_with_teleop: bool = False
    enable_pedal_outcome: bool = True

    acp_inference: ACPInferenceConfig = field(default_factory=ACPInferenceConfig)

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        if self.policy is None:
            raise ValueError("policy.path is required")

        if not self.teleop_toggle_key:
            raise ValueError("`teleop_toggle_key` must be non-empty.")
        if not self.rlt_toggle_key:
            raise ValueError("`rlt_toggle_key` must be non-empty.")
        if self.teleop_toggle_key.lower() == self.rlt_toggle_key.lower():
            raise ValueError("`teleop_toggle_key` and `rlt_toggle_key` must be distinct.")
        if self.double_tap_window_s <= 0:
            raise ValueError("`double_tap_window_s` must be > 0.")

        if self.default_episode_success is not None:
            self.default_episode_success = normalize_episode_success_label(self.default_episode_success)

        if self.acp_inference.use_cfg and not self.acp_inference.enable:
            raise ValueError("`acp_inference.use_cfg=true` requires `acp_inference.enable=true`.")
        if self.acp_inference.cfg_beta < 0:
            raise ValueError("`acp_inference.cfg_beta` must be >= 0.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _handle_intervention_toggle(events: dict) -> None:
    if not events.get("toggle_intervention"):
        return
    events["toggle_intervention"] = False
    events["intervention_active"] = not events["intervention_active"]
    state = "TELEOP" if events["intervention_active"] else "POLICY"
    logger.info("[RLT] Intervention -> %s", state)


def _handle_critical_events(
    events: dict,
    tracker: CriticalPhaseTracker,
    *,
    only_critical: bool,
    frame_idx: int,
) -> None:
    if not only_critical:
        return
    if events.get("episode_outcome") is not None and tracker.is_active:
        if events["episode_outcome"] == EPISODE_FAILURE:
            tracker.mark_failure(frame_idx)
        else:
            tracker.mark_success(frame_idx)
        return
    if events.get("toggle_critical_phase"):
        events["toggle_critical_phase"] = False
        if not tracker.is_active:
            tracker.enter(frame_idx)
        else:
            tracker.mark_success(frame_idx)
            events["episode_outcome"] = EPISODE_SUCCESS
            events["exit_early"] = True


def _should_record_frame(*, only_critical: bool, tracker: CriticalPhaseTracker) -> bool:
    if not only_critical:
        return True
    return tracker.is_active


def _current_phase(*, only_critical: bool, tracker: CriticalPhaseTracker) -> float:
    if not only_critical:
        return PHASE_CRITICAL
    return PHASE_CRITICAL if tracker.is_active else PHASE_PREFIX


@safe_stop_image_writer
def _rollout_sync_rlt(
    robot: Robot,
    teleop: Teleoperator,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset: LeRobotDataset,
    events: dict,
    cfg: RLTHILConfig,
    tracker: CriticalPhaseTracker,
):
    fps = cfg.dataset.fps
    device = get_safe_torch_device(cfg.device)
    stream_online = bool(cfg.dataset.streaming_encoding)
    record_stride = 1 if cfg.record_interpolated_actions else max(1, cfg.interpolation_multiplier)

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    frame_buffer: list[dict] = []
    teleop_disable_torque(teleop)

    events["intervention_active"] = bool(cfg.start_with_teleop)
    last_action: dict[str, Any] | None = None
    last_policy_action: dict[str, Any] | None = None
    robot_action: dict[str, Any] = {}
    action_keys = list(dataset.features[ACTION]["names"])
    obs_state_names = list(dataset.features[f"{OBS_STR}.state"]["names"])
    obs_image_names = [
        key.removeprefix(f"{OBS_STR}.images.")
        for key in dataset.features
        if key.startswith(f"{OBS_STR}.images.")
    ]

    interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
    control_interval = interpolator.get_control_interval(fps)

    timestamp = 0.0
    record_tick = 0
    recorded_frames = 0
    start_t = time.perf_counter()
    stats_window_start = start_t
    policy_inference_count = 0
    robot_command_count = 0

    while timestamp < cfg.dataset.episode_time_s:
        loop_start = time.perf_counter()
        _handle_intervention_toggle(events)
        _handle_critical_events(
            events, tracker, only_critical=cfg.only_critical, frame_idx=recorded_frames
        )

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        obs_filtered = {k: obs[k] for k in obs_state_names if k in obs}
        obs_filtered.update({k: obs[k] for k in obs_image_names if k in obs})
        obs_frame = build_dataset_frame(dataset.features, obs_filtered, prefix=OBS_STR)

        is_intervention = bool(events["intervention_active"])
        intervention_state = INTERVENTION_STATE_ACTIVE if is_intervention else INTERVENTION_STATE_POLICY
        phase = _current_phase(only_critical=cfg.only_critical, tracker=tracker)

        if is_intervention:
            if teleop.name not in ["spacemouse", "gamepad"] and last_action is None:
                robot_pos = {
                    k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features
                }
                teleop_smooth_move_to(teleop, robot_pos, duration_s=0.5, fps=50)
            robot_action = teleop.get_action()
            robot_action = robot.send_action(robot_action)
            robot_command_count += 1
            source = SOURCE_HUMAN
        else:
            if interpolator.needs_new_action():
                action_values = predict_action(
                    observation=obs_frame,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=cfg.dataset.single_task,
                    robot_type=robot.robot_type,
                )
                policy_inference_count += 1
                robot_action = make_robot_action(action_values, dataset.features)
                last_policy_action = dict(robot_action)
                action_tensor = torch.tensor([robot_action[k] for k in action_keys])
                interpolator.add(action_tensor)

            interp_action = interpolator.get()
            if interp_action is not None:
                robot_action = {k: interp_action[i].item() for i, k in enumerate(action_keys)}
                robot.send_action(robot_action)
                robot_command_count += 1
                last_action = robot_action
            source = SOURCE_VLA

        if (
            _should_record_frame(only_critical=cfg.only_critical, tracker=tracker)
            and robot_action
            and (record_tick % record_stride == 0)
        ):
            action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
            frame = {**obs_frame, **action_frame, "task": cfg.dataset.single_task}
            annotate_rlt_frame(
                frame,
                dataset_features=dataset.features,
                is_intervention=is_intervention,
                intervention_state=intervention_state,
                phase=phase,
                collector_policy_id=resolve_rlt_collector_policy_id(
                    is_intervention=is_intervention, source_type=source
                ),
                policy_action=last_policy_action,
                action_names=action_keys,
            )
            if stream_online:
                dataset.add_frame(frame)
            else:
                frame_buffer.append(frame)
            recorded_frames += 1
        record_tick += 1

        if cfg.display_data and robot_action:
            log_rerun_data(observation=obs_filtered, action=robot_action)

        dt = time.perf_counter() - loop_start
        if (sleep_time := control_interval - dt) > 0:
            precise_sleep(sleep_time)
        now = time.perf_counter()
        timestamp = now - start_t

        if cfg.log_hz and (window_elapsed := now - stats_window_start) >= cfg.hz_log_interval_s:
            logger.info(
                "[RLT HIL rates] policy=%.1f Hz | robot=%.1f Hz | phase=%s | intervention=%s",
                policy_inference_count / window_elapsed,
                robot_command_count / window_elapsed,
                "critical" if phase == PHASE_CRITICAL else "prefix",
                is_intervention,
            )
            stats_window_start = now
            policy_inference_count = 0
            robot_command_count = 0

    teleop_disable_torque(teleop)
    if not stream_online:
        for frame in frame_buffer:
            dataset.add_frame(frame)


@safe_stop_image_writer
def _rollout_rtc_rlt(
    robot,
    teleop: Teleoperator,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset: LeRobotDataset,
    events: dict,
    cfg: RLTHILConfig,
    tracker: CriticalPhaseTracker,
    queue_holder: dict,
    obs_holder: dict,
    obs_lock: Lock,
    policy_active: Event,
    compile_warmup_done: Event,
):
    fps = cfg.dataset.fps
    stream_online = bool(cfg.dataset.streaming_encoding)
    record_stride = 1 if cfg.record_interpolated_actions else max(1, cfg.interpolation_multiplier)

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    frame_buffer: list[dict] = []
    teleop_disable_torque(teleop)

    events["intervention_active"] = bool(cfg.start_with_teleop)
    last_action: dict[str, Any] | None = None
    last_policy_action: dict[str, Any] | None = None
    dataset_action_keys = list(dataset.features[ACTION]["names"])
    action_keys = _resolve_action_key_order(cfg, dataset_action_keys)
    obs_state_names = list(dataset.features[f"{OBS_STR}.state"]["names"])
    obs_image_names = [
        key.removeprefix(f"{OBS_STR}.images.")
        for key in dataset.features
        if key.startswith(f"{OBS_STR}.images.")
    ]

    interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
    control_interval = interpolator.get_control_interval(fps)

    robot_action: dict[str, Any] = {}
    timestamp = 0.0
    start_t = time.perf_counter()
    stats_window_start = start_t
    robot_command_count = 0
    record_tick = 0
    recorded_frames = 0
    obs_poll_interval = 1.0 / fps
    last_obs_poll_t = 0.0
    obs_filtered: dict[str, Any] = {}
    obs_frame: dict[str, Any] = {}
    warmup_wait_logged = False
    warmup_queue_flushed = False

    while timestamp < cfg.dataset.episode_time_s:
        loop_start = time.perf_counter()
        _handle_intervention_toggle(events)
        _handle_critical_events(
            events, tracker, only_critical=cfg.only_critical, frame_idx=recorded_frames
        )

        if events["exit_early"]:
            events["exit_early"] = False
            break

        is_intervention = bool(events["intervention_active"])
        if is_intervention:
            policy_active.clear()
        elif not policy_active.is_set():
            policy_active.set()

        now_for_obs = time.perf_counter()
        should_poll_obs = (
            not obs_filtered
            or (now_for_obs - last_obs_poll_t) >= obs_poll_interval
            or is_intervention
        )
        if should_poll_obs:
            obs = robot.get_observation()
            obs_filtered = {k: obs[k] for k in obs_state_names if k in obs}
            obs_filtered.update({k: obs[k] for k in obs_image_names if k in obs})
            obs_frame = build_dataset_frame(dataset.features, obs_filtered, prefix=OBS_STR)
            with obs_lock:
                obs_holder["obs"] = obs_filtered
            last_obs_poll_t = now_for_obs

        intervention_state = INTERVENTION_STATE_ACTIVE if is_intervention else INTERVENTION_STATE_POLICY
        phase = _current_phase(only_critical=cfg.only_critical, tracker=tracker)

        if is_intervention:
            robot_action = teleop.get_action()
            robot_action = robot.send_action(robot_action)
            robot_command_count += 1
            source = SOURCE_HUMAN
            queue_holder["queue"] = ActionQueue(cfg.rtc)
            interpolator.reset()
        else:
            if cfg.use_torch_compile and not compile_warmup_done.is_set():
                if not warmup_wait_logged:
                    logger.info(
                        "[RTC] Waiting for compile warmup (%d inferences)",
                        max(1, int(cfg.compile_warmup_inferences)),
                    )
                    warmup_wait_logged = True
                source = SOURCE_VLA
            else:
                if cfg.use_torch_compile and not warmup_queue_flushed:
                    queue_holder["queue"] = ActionQueue(cfg.rtc)
                    interpolator.reset()
                    warmup_queue_flushed = True

                queue = queue_holder["queue"]
                if interpolator.needs_new_action():
                    new_action = queue.get() if queue else None
                    if new_action is not None:
                        interpolator.add(new_action.cpu())

                action_tensor = interpolator.get()
                if action_tensor is not None:
                    robot_action = {
                        k: action_tensor[i].item()
                        for i, k in enumerate(action_keys)
                        if i < len(action_tensor)
                    }
                    robot.send_action(robot_action)
                    robot_command_count += 1
                    last_action = robot_action
                    last_policy_action = dict(robot_action)
                source = SOURCE_VLA

        if (
            _should_record_frame(only_critical=cfg.only_critical, tracker=tracker)
            and robot_action
            and (record_tick % record_stride == 0)
        ):
            action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
            frame = {**obs_frame, **action_frame, "task": cfg.dataset.single_task}
            annotate_rlt_frame(
                frame,
                dataset_features=dataset.features,
                is_intervention=is_intervention,
                intervention_state=intervention_state,
                phase=phase,
                collector_policy_id=resolve_rlt_collector_policy_id(
                    is_intervention=is_intervention, source_type=source
                ),
                policy_action=last_policy_action,
                action_names=list(dataset.features[ACTION]["names"]),
            )
            if stream_online:
                dataset.add_frame(frame)
            else:
                frame_buffer.append(frame)
            recorded_frames += 1
        record_tick += 1

        if cfg.display_data and robot_action:
            log_rerun_data(observation=obs_filtered, action=robot_action)

        dt = time.perf_counter() - loop_start
        if (sleep_time := control_interval - dt) > 0:
            precise_sleep(sleep_time)

        now = time.perf_counter()
        timestamp = now - start_t
        if cfg.log_hz and (window_elapsed := now - stats_window_start) >= cfg.hz_log_interval_s:
            logger.info(
                "[RLT HIL RTC rates] robot=%.1f Hz | phase=%s | intervention=%s | queue=%d",
                robot_command_count / window_elapsed,
                "critical" if phase == PHASE_CRITICAL else "prefix",
                is_intervention,
                queue_holder["queue"].qsize() if queue_holder.get("queue") else 0,
            )
            stats_window_start = now
            robot_command_count = 0

    policy_active.clear()
    teleop_disable_torque(teleop)
    if not stream_online:
        for frame in frame_buffer:
            dataset.add_frame(frame)


@parser.wrap()
def rlt_hil_collect(cfg: RLTHILConfig) -> LeRobotDataset:
    init_logging()
    if cfg.require_episode_success_label and not cfg.enable_episode_outcome_labeling:
        raise ValueError(
            "`require_episode_success_label=true` requires `enable_episode_outcome_labeling=true`."
        )
    logger.info(pformat(cfg.__dict__))

    use_rtc = cfg.rtc.enabled
    if use_rtc:
        _set_openarm_max_relative_target_if_missing(cfg.robot, max_relative_target=8.0)

    if cfg.display_data:
        init_rerun(session_name="rlt_hil_collection")

    robot_raw = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop_proc, obs_proc = make_identity_processors()

    action_features_hw = {k: v for k, v in robot_raw.action_features.items() if k.endswith(".pos")}
    all_observation_features = robot_raw.observation_features
    available_joint_names = [
        key for key, value in all_observation_features.items() if key.endswith(".pos") and value is float
    ]
    ordered_joint_names = _resolve_state_joint_order(
        getattr(cfg.policy, "action_feature_names", None),
        available_joint_names,
    )
    observation_features_hw = {
        joint_name: all_observation_features[joint_name] for joint_name in ordered_joint_names
    }
    for key, value in all_observation_features.items():
        if isinstance(value, tuple):
            observation_features_hw[key] = value

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_proc,
            initial_features=create_initial_features(action=action_features_hw),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=obs_proc,
            initial_features=create_initial_features(observation=observation_features_hw),
            use_videos=cfg.dataset.video,
        ),
    )
    action_names = list(action_features_hw)
    names_from_features = dataset_features.get(ACTION, {}).get("names")
    if names_from_features is not None:
        action_names = list(names_from_features)
    ensure_rlt_dataset_features(dataset_features, action_feature_names=action_names)

    dataset = None
    listener = None
    shutdown_event = Event()
    policy_active = Event()
    compile_warmup_done = Event()
    if not cfg.use_torch_compile:
        compile_warmup_done.set()
    rtc_thread = None
    tracker = CriticalPhaseTracker()

    try:
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )
            if hasattr(robot_raw, "cameras") and robot_raw.cameras:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot_raw.cameras),
                )
        else:
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot_raw.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot_raw.cameras if hasattr(robot_raw, "cameras") else []),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )
            write_rlt_schema_metadata(dataset, policy_cfg=cfg.policy)

        if use_rtc:
            policy_class = get_policy_class(cfg.policy.type)
            policy_config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)
            if hasattr(policy_config, "compile_model"):
                policy_config.compile_model = cfg.use_torch_compile
            policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=policy_config)
            policy.config.rtc_config = cfg.rtc
            if hasattr(policy, "init_rtc_processor"):
                policy.init_rtc_processor()
            policy = policy.to(cfg.device)
            policy.eval()
        else:
            policy = make_policy(cfg.policy, ds_meta=dataset.meta)

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

        if use_rtc:
            logger.info("Connecting robot (calibrate=%s)", cfg.calibrate)
            robot_raw.connect(calibrate=False)
            if cfg.calibrate and hasattr(robot_raw, "calibrate"):
                robot_raw.calibrate()
                robot_raw.disconnect()
                robot_raw.connect(calibrate=False)
        else:
            robot_raw.connect()

        robot = ThreadSafeRobot(robot_raw) if use_rtc else robot_raw
        teleop.connect()
        listener, events = init_rlt_keyboard_listener(
            teleop_toggle_key=cfg.teleop_toggle_key,
            rlt_toggle_key=cfg.rlt_toggle_key,
            double_tap_window_s=cfg.double_tap_window_s,
            only_critical=cfg.only_critical,
        )
        if cfg.enable_pedal_outcome:
            start_rlt_pedal_listener(events, double_tap_window_s=cfg.double_tap_window_s)

        queue_holder = None
        obs_holder = None
        obs_lock = Lock()
        hw_features = None
        if use_rtc:
            queue_holder = {"queue": ActionQueue(cfg.rtc)}
            obs_holder = {
                "obs": None,
                "robot_type": robot.robot_type,
                "action_feature_names": [key for key in robot.action_features if key.endswith(".pos")],
            }
            hw_features = hw_to_dataset_features(observation_features_hw, "observation")
            rtc_thread = Thread(
                target=_rtc_inference_thread,
                args=(
                    policy,
                    obs_holder,
                    obs_lock,
                    hw_features,
                    preprocessor,
                    postprocessor,
                    queue_holder,
                    shutdown_event,
                    policy_active,
                    compile_warmup_done,
                    cfg,
                ),
                daemon=True,
            )
            rtc_thread.start()

        print_rlt_controls(only_critical=cfg.only_critical, rtc=use_rtc)
        logger.info("  Policy: %s", cfg.policy.pretrained_path)
        logger.info("  Task: %s", cfg.dataset.single_task)
        logger.info("  only_critical=%s start_with_teleop=%s", cfg.only_critical, cfg.start_with_teleop)
        if use_rtc:
            logger.info("  RTC execution_horizon=%s", cfg.rtc.execution_horizon)

        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                events["episode_outcome"] = None
                events["rerecord_episode"] = False
                events["toggle_intervention"] = False
                events["toggle_critical_phase"] = False
                events["intervention_active"] = bool(cfg.start_with_teleop)
                tracker.reset()
                log_say(f"Episode {dataset.num_episodes}", cfg.play_sounds)

                if policy is not None:
                    robot_raw.reset_to_default_positions()

                if use_rtc:
                    queue_holder["queue"] = ActionQueue(cfg.rtc)
                    _rollout_rtc_rlt(
                        robot=robot,
                        teleop=teleop,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        dataset=dataset,
                        events=events,
                        cfg=cfg,
                        tracker=tracker,
                        queue_holder=queue_holder,
                        obs_holder=obs_holder,
                        obs_lock=obs_lock,
                        policy_active=policy_active,
                        compile_warmup_done=compile_warmup_done,
                    )
                else:
                    _rollout_sync_rlt(
                        robot=robot,
                        teleop=teleop,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        dataset=dataset,
                        events=events,
                        cfg=cfg,
                        tracker=tracker,
                    )

                episode_success = None
                if cfg.enable_episode_outcome_labeling:
                    explicit = events.get("episode_outcome")
                    if explicit is None and tracker.outcome is not None:
                        explicit = tracker.outcome
                    episode_success = resolve_episode_success_label(
                        explicit_label=explicit,
                        default_label=cfg.default_episode_success,
                        require_label=cfg.require_episode_success_label,
                    )
                    if explicit is None and episode_success is not None:
                        logging.warning(
                            "Episode %s has no explicit success/failure label, defaulting to '%s'.",
                            dataset.num_episodes,
                            episode_success,
                        )

                if not events["stop_recording"] and recorded < cfg.dataset.num_episodes:
                    log_say("Reset the environment", cfg.play_sounds)
                    reset_loop(robot_raw if not use_rtc else robot_raw, teleop, events, cfg.dataset.fps)

                if events["rerecord_episode"]:
                    log_say("Re-recording", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    events["episode_outcome"] = None
                    dataset.clear_episode_buffer()
                    continue

                if dataset.has_pending_frames():
                    extra_episode_metadata = (
                        {"episode_success": episode_success} if cfg.enable_episode_outcome_labeling else None
                    )
                    if cfg.only_critical and tracker.start_frame is not None:
                        if extra_episode_metadata is None:
                            extra_episode_metadata = {}
                        extra_episode_metadata["rl_intervals"] = [
                            {
                                "start_frame": tracker.start_frame,
                                "end_frame": tracker.end_frame
                                if tracker.end_frame is not None
                                else max(tracker.start_frame, 0),
                                "outcome": episode_success,
                            }
                        ]
                    dataset.save_episode(extra_episode_metadata=extra_episode_metadata)
                    recorded += 1
                else:
                    log_say("Episode buffer is empty, skipping save_episode().")
                    dataset.clear_episode_buffer()

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        shutdown_event.set()
        policy_active.clear()
        if rtc_thread and rtc_thread.is_alive():
            rtc_thread.join(timeout=2.0)
        if dataset:
            dataset.finalize()
        if robot_raw.is_connected:
            if policy is not None:
                robot_raw.reset_to_default_positions()
            robot_raw.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if not is_headless() and listener:
            listener.stop()
        if cfg.dataset.push_to_hub and dataset is not None:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    return dataset


def main():
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    rlt_hil_collect()


if __name__ == "__main__":
    main()
