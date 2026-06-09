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
Latent PI0.5 + RTR (Reuse-then-Refine) inference on real robots.

Pipeline (RTR paper + ``pi0_5_latent_rdp_vae_model_wrapper.py``):
  1. Policy predicts latent chunk (plain ``predict_action_chunk``)
  2. VAE decode to high-frequency action chunk
  3. RTR refine: concat executed actions during inference delay + new chunk suffix → VAE encode→decode
  4. Robot executes decoded actions at ``fps`` (default 60 Hz)

Usage:
    uv run examples/rtc/eval_latent_rtr_with_robot.py \\
        --policy.path=outputs/pi05_latent_.../pretrained_model \\
        --vae_checkpoint_path=outputs/vae_train/.../latest.ckpt \\
        --latent_dataset_statistics=outputs/vae_eval/.../dataset_stats.json \\
        --rtr.enabled=true \\
        --robot.type=agilex_cobot \\
        --task="sort the screws"
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Thread

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    agilex_cobot,
    bi_so_follower,
    koch_follower,
    so_follower,
    unitree_g1,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging
from lerobot.vae.configuration import ActionVAEConfig
from lerobot.vae.latent_inference import LatentActionPipeline, LatentInferenceConfig
from lerobot.vae.latent_rtc_queue import LatentRTRActionQueue
from lerobot.vae.rtr_refiner import RTRConfig

_rtc_eval_path = Path(__file__).with_name("eval_with_real_robot.py")
_rtc_spec = importlib.util.spec_from_file_location("_rtc_eval_real_robot", _rtc_eval_path)
_rtc_mod = importlib.util.module_from_spec(_rtc_spec)
assert _rtc_spec.loader is not None
_rtc_spec.loader.exec_module(_rtc_mod)

RobotWrapper = _rtc_mod.RobotWrapper
_apply_torch_compile = _rtc_mod._apply_torch_compile
actor_control = _rtc_mod.actor_control
save_visualization = _rtc_mod.save_visualization
update_visualization = _rtc_mod.update_visualization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LatentRTRDemoConfig(HubMixin):
    """RTR demo for latent PI0.5 policies on real robots."""

    policy: PreTrainedConfig | None = None
    robot: RobotConfig | None = None

    vae: ActionVAEConfig = field(default_factory=ActionVAEConfig)
    vae_checkpoint_path: str | None = None
    latent_dataset_statistics: str | None = None
    latent_normalization_type: str = "NORMAL"
    temporal_downsample_ratio: int = 4

    rtr: RTRConfig = field(default_factory=RTRConfig)

    duration: float = 60.0
    fps: float = 60.0
    interpolation_multiplier: int = 1
    device: str | None = None
    action_queue_size_to_get_new_actions: int = 24

    task: str = field(default="", metadata={"help": "Task instruction for the policy"})

    use_torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "max-autotune"
    torch_compile_disable_cudagraphs: bool = True

    enable_visualization: bool = False
    viz_history_size: int = 2400

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required (`--policy.path=...`).")

        if self.robot is None:
            raise ValueError("Robot configuration must be provided.")

        if self.vae_checkpoint_path is None:
            raise ValueError("`vae_checkpoint_path` is required.")
        if self.latent_dataset_statistics is None:
            raise ValueError("`latent_dataset_statistics` is required.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def get_latent_actions(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: LatentRTRActionQueue,
    latent_pipeline: LatentActionPipeline,
    shutdown_event: Event,
    cfg: LatentRTRDemoConfig,
):
    """Request latent chunks, decode + RTR refine, and enqueue robot actions.

    Inference delay handling (aligned with ``eval_with_real_robot.py``):
    - Before inference: snapshot ``queue_leftover = get_processed_left_over()``.
    - Before inference: estimate delay from ``LatencyTracker.max()``.
    - After inference: measure latency, update tracker, refine with queue leftover
      prefix + decoded suffix, then merge skipping consumed steps.
    """
    try:
        logger.info("[GET_ACTIONS] Starting latent+RTR get actions thread")

        latency_tracker = LatencyTracker()
        time_per_action = 1.0 / cfg.fps

        dataset_features = hw_to_dataset_features(robot.observation_features(), "observation")
        policy_device = policy.config.device

        logger.info("[GET_ACTIONS] Loading preprocessor from %s", cfg.policy.pretrained_path)
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=None,
            preprocessor_overrides={"device_processor": {"device": cfg.policy.device}},
        )

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions

        inference_warmup_times = 0

        while not shutdown_event.is_set():
            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                queue_leftover = action_queue.get_processed_left_over()

                max_latency = latency_tracker.max() or 0.0
                estimated_delay = math.ceil(max_latency / time_per_action)

                obs = robot.get_observation()
                obs_processed = robot_observation_processor(obs)
                obs_with_policy_features = build_dataset_frame(
                    dataset_features, obs_processed, prefix="observation"
                )

                for name in obs_with_policy_features:
                    obs_with_policy_features[name] = torch.from_numpy(obs_with_policy_features[name])
                    if "image" in name:
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].type(torch.float32) / 255
                        )
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].permute(2, 0, 1).contiguous()
                        )
                    obs_with_policy_features[name] = obs_with_policy_features[name].unsqueeze(0)
                    obs_with_policy_features[name] = obs_with_policy_features[name].to(policy_device)

                obs_with_policy_features["task"] = [cfg.task]
                obs_with_policy_features["robot_type"] = (
                    robot.robot.name if hasattr(robot.robot, "name") else ""
                )

                preprocessed_obs = preprocessor(obs_with_policy_features)

                with torch.no_grad():
                    latent_actions = policy.predict_action_chunk(preprocessed_obs)

                inference_warmup_times += 1
                if inference_warmup_times < 3:
                    continue

                new_latency = time.perf_counter() - current_time
                latency_delay = math.ceil(new_latency / time_per_action)
                latency_tracker.add(new_latency)

                consumed_delay = action_queue.get_consumed_since(action_index_before_inference)
                # Prefer actual queue consumption; fall back to measured / estimated latency.
                inference_delay = consumed_delay if consumed_delay > 0 else latency_delay
                if inference_delay == 0 and estimated_delay > 0:
                    inference_delay = estimated_delay

                with torch.no_grad():
                    action_chunk = latent_pipeline.latent_to_actions(
                        latent_actions,
                        queue_leftover=queue_leftover,
                        inference_delay=inference_delay,
                    )
                    processed_actions = action_chunk.squeeze(0)

                logger.debug(
                    "[GET_ACTIONS] latency=%.3fs latency_delay=%s consumed_delay=%s "
                    "estimated_delay=%s inference_delay=%s leftover=%s",
                    new_latency,
                    latency_delay,
                    consumed_delay,
                    estimated_delay,
                    inference_delay,
                    None if queue_leftover is None else tuple(queue_leftover.shape),
                )

                action_queue.merge(
                    processed_actions,
                    latency_delay,
                    action_index_before_inference,
                )
            else:
                time.sleep(time_per_action)

        logger.info("[GET_ACTIONS] Latent get actions thread shutting down")
    except Exception as e:
        logger.error("[GET_ACTIONS] Fatal exception: %s", e)
        logger.error(traceback.format_exc())
        sys.exit(1)


@parser.wrap()
def demo_cli(cfg: LatentRTRDemoConfig):
    init_logging()
    logger.info("Using device: %s", cfg.device)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    policy = None
    robot = None

    policy_class = get_policy_class(cfg.policy.type)
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)

    if cfg.policy.type in ("pi05", "pi0"):
        config.compile_model = cfg.use_torch_compile

    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)
    policy.type = cfg.policy.type
    logger.info("RTR mode: latent inference + VAE decode + refine")

    if policy.name not in ("pi05", "pi0"):
        raise ValueError("Latent RTR demo supports pi05 and pi0 only.")

    policy = policy.to(cfg.device)
    policy.eval()

    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)

    latent_cfg = LatentInferenceConfig(
        vae=cfg.vae,
        vae_checkpoint_path=cfg.vae_checkpoint_path,
        latent_dataset_statistics=cfg.latent_dataset_statistics,
        latent_normalization_type=cfg.latent_normalization_type,
        temporal_downsample_ratio=cfg.temporal_downsample_ratio,
        rtr=cfg.rtr,
    )
    latent_pipeline = LatentActionPipeline.from_config(latent_cfg, cfg.device)
    logger.info(
        "Latent pipeline: latent_horizon=%s action_horizon=%s rtr=%s",
        latent_pipeline.latent_horizon,
        latent_pipeline.action_horizon,
        cfg.rtr.enabled,
    )

    logger.info("Initializing robot: %s", cfg.robot.type)
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    robot.reset_to_default_positions()
    time.sleep(3)
    robot_wrapper = RobotWrapper(robot)

    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()

    viz_queue = None
    fig = axes = lines = history_buffer = None
    if cfg.enable_visualization:
        try:
            import matplotlib.pyplot as plt
            from collections import deque

            plt.ion()
            action_features = list(robot_wrapper.action_features())
            num_actions = len(action_features)
            fig, axes = plt.subplots(num_actions, 1, figsize=(10, 2 * num_actions), sharex=True)
            if num_actions == 1:
                axes = [axes]
            fig.suptitle("Executed Actions (Latent+RTR)", fontsize=14)
            lines = []
            for i, ax in enumerate(axes):
                line, = ax.plot([], [], linewidth=1.5, color="red")
                lines.append(line)
                ax.set_ylabel(action_features[i], fontsize=10)
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel("Time Steps")
            plt.tight_layout()
            plt.show(block=False)
            cfg.viz_history_size = max(
                int(cfg.duration * cfg.fps * cfg.interpolation_multiplier),
                cfg.viz_history_size,
            )
            history_buffer = deque(maxlen=cfg.viz_history_size)
            viz_queue = Queue(maxsize=cfg.viz_history_size * 2)
        except ImportError:
            logger.warning("matplotlib not available, visualization disabled")
            cfg.enable_visualization = False

    action_queue = LatentRTRActionQueue()

    get_actions_thread = Thread(
        target=get_latent_actions,
        args=(
            policy,
            robot_wrapper,
            robot_observation_processor,
            action_queue,
            latent_pipeline,
            shutdown_event,
            cfg,
        ),
        daemon=True,
        name="GetLatentActions",
    )
    actor_thread = Thread(
        target=actor_control,
        args=(robot_wrapper, robot_action_processor, action_queue, viz_queue, shutdown_event, cfg),
        daemon=True,
        name="Actor",
    )

    get_actions_thread.start()
    time.sleep(3)
    actor_thread.start()

    logger.info("Running latent+RTR demo for %.0fs...", cfg.duration)
    start_time = time.time()
    while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
        time.sleep(5)
        logger.info("[MAIN] Action queue size: %s", action_queue.qsize())

    shutdown_event.set()
    get_actions_thread.join(timeout=10)
    actor_thread.join(timeout=10)

    if cfg.enable_visualization and fig is not None:
        import matplotlib.pyplot as plt

        if viz_queue is not None:
            while not viz_queue.empty():
                history_buffer.append(viz_queue.get_nowait())
        update_visualization(fig, axes, lines, history_buffer, cfg)
        save_visualization(fig, "latent_rtr_action_history.png")
        plt.pause(2.0)
        plt.close(fig)

    latent_pipeline.reset()
    if robot:
        robot.reset_to_default_positions()
        robot.disconnect()
    logger.info("Cleanup completed")


if __name__ == "__main__":
    demo_cli()
