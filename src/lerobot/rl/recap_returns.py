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

"""RECAP-style return and advantage utilities (aligned with RLinf)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeTargetInfo:
    episode_index: int
    task_index: int
    length: int
    success: bool


@dataclass(frozen=True)
class RecapReturnStats:
    ret_min: float
    ret_max: float

    def normalize_to_minus_one_zero(self, value: float) -> float:
        """Map ``[ret_min, ret_max]`` to ``[-1, 0]`` (RLinf compute_advantages)."""
        ret_range = self.ret_max - self.ret_min
        if ret_range <= 0:
            return -0.5
        return (float(value) - self.ret_min) / ret_range - 1.0


def compute_episode_returns_and_rewards(
    episode_length: int,
    is_success: bool,
    *,
    gamma: float = 1.0,
    failure_reward: float = -300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-step rewards and discounted returns for one episode.

    Reward schedule (RLinf RECAP):
    - ``r_t = -1`` for all non-terminal steps
    - terminal: ``0`` on success, ``failure_reward`` on failure
    - ``G_t = r_t + gamma * G_{t+1}`` (backward iteration)
    """
    if episode_length <= 0:
        raise ValueError(f"'episode_length' must be > 0, got {episode_length}.")
    if gamma < 0:
        raise ValueError(f"'gamma' must be >= 0, got {gamma}.")

    rewards = np.full(episode_length, -1.0, dtype=np.float32)
    rewards[-1] = 0.0 if is_success else float(failure_reward)

    returns = np.zeros(episode_length, dtype=np.float32)
    cumulative = 0.0
    for step in reversed(range(episode_length)):
        cumulative = float(rewards[step]) + float(gamma) * cumulative
        returns[step] = cumulative
    return returns, rewards


def build_episode_return_tables(
    episode_info: dict[int, EpisodeTargetInfo],
    *,
    gamma: float = 1.0,
    failure_reward: float = -300.0,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], RecapReturnStats]:
    """Build per-episode return/reward arrays and global return stats."""
    episode_returns: dict[int, np.ndarray] = {}
    episode_rewards: dict[int, np.ndarray] = {}
    all_returns: list[float] = []

    for ep_idx, ep in episode_info.items():
        returns, rewards = compute_episode_returns_and_rewards(
            episode_length=ep.length,
            is_success=ep.success,
            gamma=gamma,
            failure_reward=failure_reward,
        )
        episode_returns[ep_idx] = returns
        episode_rewards[ep_idx] = rewards
        all_returns.extend(returns.tolist())

    if not all_returns:
        raise ValueError("No episode returns were computed.")

    return (
        episode_returns,
        episode_rewards,
        RecapReturnStats(ret_min=float(np.min(all_returns)), ret_max=float(np.max(all_returns))),
    )


def compute_recap_normalized_value_targets(
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    episode_returns: dict[int, np.ndarray],
    return_stats: RecapReturnStats,
) -> np.ndarray:
    """Map each frame to its normalized return target in ``[-1, 0]``."""
    if episode_indices.shape != frame_indices.shape:
        raise ValueError("episode_indices and frame_indices must have the same shape.")

    targets = np.zeros(episode_indices.shape[0], dtype=np.float32)
    for i in range(episode_indices.shape[0]):
        ep_idx = int(episode_indices[i])
        frame_idx = int(frame_indices[i])
        if ep_idx not in episode_returns:
            raise KeyError(f"Missing return table for episode_index={ep_idx}.")
        ep_returns = episode_returns[ep_idx]
        if frame_idx < 0 or frame_idx >= ep_returns.shape[0]:
            raise IndexError(
                f"frame_index={frame_idx} out of range for episode_index={ep_idx} "
                f"(length={ep_returns.shape[0]})."
            )
        targets[i] = return_stats.normalize_to_minus_one_zero(float(ep_returns[frame_idx]))
    return targets


def compute_recap_n_step_advantages(
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    episode_returns: dict[int, np.ndarray],
    episode_rewards: dict[int, np.ndarray],
    episode_lengths: dict[int, int],
    *,
    n_step: int,
    gamma: float = 1.0,
    return_stats: RecapReturnStats,
    discount_next_value: bool = True,
) -> np.ndarray:
    """Compute RECAP advantages aligned with RLinf.

    ``A_t = normalize(sum_{k=0}^{N-1} gamma^k r_{t+k}) + gamma^N V(o_{t+N}) - V(o_t)``
    """
    if n_step <= 0:
        raise ValueError("'n_step' must be > 0.")
    if values.shape[0] != episode_indices.shape[0]:
        raise ValueError("values and episode_indices must have the same length.")

    n = values.shape[0]
    advantages = np.zeros(n, dtype=np.float32)
    gamma_powers = np.array([gamma**i for i in range(n_step)], dtype=np.float64)

    for i in range(n):
        ep_idx = int(episode_indices[i])
        frame_idx = int(frame_indices[i])
        ep_length = episode_lengths[ep_idx]
        ep_returns = episode_returns[ep_idx]
        ep_rewards = episode_rewards[ep_idx]

        num_valid = min(n_step, ep_length - frame_idx)
        is_next_pad = frame_idx + n_step >= ep_length

        v_curr = float(values[i])
        if is_next_pad:
            v_next = 0.0
            future_return = None
        else:
            future_frame = frame_idx + n_step
            future_return = float(ep_returns[future_frame])
            # Locate V(o_{t+N}) in the flat arrays.
            future_mask = (episode_indices == ep_idx) & (frame_indices == future_frame)
            future_indices = np.flatnonzero(future_mask)
            if future_indices.size != 1:
                raise RuntimeError(
                    f"Expected exactly one frame for episode={ep_idx} frame={future_frame}, "
                    f"found {future_indices.size}."
                )
            v_next = float(values[int(future_indices[0])])

        true_return = float(ep_returns[frame_idx])
        if abs(gamma - 1.0) < 1e-8:
            if is_next_pad:
                reward_sum_raw = true_return
            else:
                reward_sum_raw = true_return - float(future_return)
        else:
            reward_slice = ep_rewards[frame_idx : frame_idx + num_valid]
            if reward_slice.shape[0] != num_valid:
                raise RuntimeError(
                    f"Invalid reward slice for episode={ep_idx} frame={frame_idx}: "
                    f"expected {num_valid}, got {reward_slice.shape[0]}."
                )
            reward_sum_raw = float(np.sum(gamma_powers[:num_valid] * reward_slice))

        reward_sum = return_stats.normalize_to_minus_one_zero(reward_sum_raw)
        gamma_k = gamma**num_valid if discount_next_value else 1.0
        advantages[i] = float(reward_sum + gamma_k * v_next - v_curr)

    return advantages
