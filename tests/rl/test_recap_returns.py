#!/usr/bin/env python

import numpy as np

from lerobot.rl.recap_returns import (
    RecapReturnStats,
    build_episode_return_tables,
    compute_episode_returns_and_rewards,
    compute_recap_n_step_advantages,
    compute_recap_normalized_value_targets,
    EpisodeTargetInfo,
)


def _episode_info(ep_idx: int, length: int, success: bool) -> EpisodeTargetInfo:
    return EpisodeTargetInfo(
        episode_index=ep_idx,
        task_index=0,
        length=length,
        success=success,
    )


def test_success_episode_returns_gamma_one():
    returns, rewards = compute_episode_returns_and_rewards(4, True, gamma=1.0, failure_reward=-300.0)
    assert np.allclose(rewards, [-1.0, -1.0, -1.0, 0.0])
    assert np.allclose(returns, [-3.0, -2.0, -1.0, 0.0])


def test_failure_episode_terminal_penalty():
    returns, rewards = compute_episode_returns_and_rewards(3, False, gamma=1.0, failure_reward=-10.0)
    assert np.allclose(rewards, [-1.0, -1.0, -10.0])
    assert np.allclose(returns, [-12.0, -11.0, -10.0])


def test_recap_advantage_matches_return_difference_when_values_are_perfect():
    episode_info = {
        0: _episode_info(0, 4, True),
        1: _episode_info(1, 3, False),
    }
    episode_returns, episode_rewards, stats = build_episode_return_tables(
        episode_info, gamma=1.0, failure_reward=-10.0
    )
    episode_lengths = {ep_idx: ep.length for ep_idx, ep in episode_info.items()}

    episode_indices = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    frame_indices = np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int64)
    values = compute_recap_normalized_value_targets(
        episode_indices, frame_indices, episode_returns, stats
    )

    advantages = compute_recap_n_step_advantages(
        values=values,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        episode_returns=episode_returns,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        n_step=1,
        gamma=1.0,
        return_stats=stats,
        discount_next_value=True,
    )

    # Perfect value predictions => advantage ~= 0 everywhere.
    assert np.allclose(advantages, 0.0, atol=1e-5)
    # Failure terminal frame should still have lower return than success frames.
    assert episode_returns[1][-1] < episode_returns[0][-1]


def test_normalize_return_range():
    stats = RecapReturnStats(ret_min=-12.0, ret_max=0.0)
    assert np.isclose(stats.normalize_to_minus_one_zero(-12.0), -1.0)
    assert np.isclose(stats.normalize_to_minus_one_zero(0.0), 0.0)
