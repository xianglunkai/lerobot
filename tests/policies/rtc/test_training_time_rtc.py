#!/usr/bin/env python

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

"""Tests for training-time RTC helpers."""

import pytest
import torch

from lerobot.configs.types import RTCTrainingDelayDistribution
from lerobot.policies.rtc.configuration_rtc import RTCTrainingConfig
from lerobot.policies.rtc.training_time import apply_rtc_training_time, masked_mean, sample_rtc_delay


def test_rtc_training_config_defaults():
    config = RTCTrainingConfig()
    assert config.enabled is False
    assert config.min_delay == 0
    assert config.max_delay == 0
    assert config.delay_distribution == RTCTrainingDelayDistribution.UNIFORM
    assert config.exp_decay == 1.0


def test_sample_rtc_delay_uniform_range():
    cfg = RTCTrainingConfig(enabled=True, min_delay=1, max_delay=4)
    delays = sample_rtc_delay(cfg, batch_size=100, device=torch.device("cpu"))
    assert delays.min().item() >= 1
    assert delays.max().item() <= 4


def test_apply_rtc_training_time_prefix_mask():
    time = torch.tensor([0.5])
    delays = torch.tensor([2])
    time_tokens, postfix_mask = apply_rtc_training_time(time, delays, seq_len=4)
    assert time_tokens.shape == (1, 4)
    assert postfix_mask.shape == (1, 4)
    # Delay=2 means the first two steps are prefix (time forced to 0.0) and only the last two are postfix.
    assert torch.allclose(time_tokens[0], torch.tensor([0.0, 0.0, 0.5, 0.5]))
    assert torch.equal(postfix_mask[0], torch.tensor([False, False, True, True]))


def test_apply_training_time_rtc_inference_uses_clean_prefix_time_zero():
    from lerobot.policies.rtc.training_time import apply_training_time_rtc_inference

    x_t = torch.randn(1, 4, 2)
    prev = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
    x_cond, time_tokens, padded = apply_training_time_rtc_inference(
        x_t, time=0.3, inference_delay=2, prev_chunk_left_over=prev, chunk_size=4
    )
    assert padded is not None
    assert torch.allclose(x_cond[0, :2], prev[0, :2])
    assert torch.allclose(time_tokens[0], torch.tensor([0.0, 0.0, 0.3, 0.3]))


def test_apply_training_time_rtc_inference_pads_short_leftover_with_zeros():
    from lerobot.policies.rtc.training_time import apply_training_time_rtc_inference

    x_t = torch.ones(1, 4, 4) * 9.0
    prev = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # shorter in time and action dim
    x_cond, time_tokens, padded = apply_training_time_rtc_inference(
        x_t, time=0.3, inference_delay=2, prev_chunk_left_over=prev, chunk_size=4
    )
    assert padded is not None
    assert padded.shape == x_t.shape
    assert torch.allclose(padded[0, :2, :2], prev[0])
    assert torch.allclose(padded[0, :2, 2:], torch.zeros(2, 2))
    assert torch.allclose(padded[0, 2:], torch.zeros(2, 4))
    assert torch.allclose(x_cond[0, :2, :2], prev[0])
    assert torch.allclose(x_cond[0, :2, 2:], torch.zeros(2, 2))
    assert torch.allclose(time_tokens[0], torch.tensor([0.0, 0.0, 0.3, 0.3]))


def test_pi05_config_rejects_invalid_training_rtc_delay():
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.rtc.configuration_rtc import RTCTrainingConfig

    with pytest.raises(ValueError, match="rtc_training_config.max_delay"):
        PI05Config(
            chunk_size=5,
            n_action_steps=5,
            rtc_training_config=RTCTrainingConfig(enabled=True, max_delay=5),
        )


def test_pi05_flow_matching_clean_prefix_and_postfix_loss():
    """Training-time RTC keeps prefix clean and averages loss only on postfix."""
    actions = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    noise = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
    time = torch.tensor([0.25])
    delay = torch.tensor([2])
    time_tokens, postfix_mask = apply_rtc_training_time(time, delay, seq_len=4)

    time_expanded = time_tokens[:, :, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    assert torch.allclose(time_tokens[0], torch.tensor([0.0, 0.0, 0.25, 0.25]))
    assert torch.equal(x_t[:, :2], actions[:, :2])
    assert torch.allclose(x_t[:, 2:], 0.25 * noise[:, 2:] + 0.75 * actions[:, 2:])

    losses = torch.tensor([[[100.0], [100.0], [2.0], [4.0]]])
    loss = masked_mean(losses, postfix_mask, reduce_dims=(0, 1, 2))
    assert loss.item() == pytest.approx(3.0)


def test_gemma_rmsnorm_per_token_cond_keeps_rank():
    """Per-token AdaRMS cond must stay (B, T, H), not broadcast to (B, T, T, H)."""
    pytest.importorskip("transformers")
    from lerobot.policies.pi05 import modeling_pi05 as pi05_mod

    if pi05_mod.GemmaRMSNorm is None:
        pytest.skip("transformers GemmaRMSNorm unavailable")

    # Ensure the module-level patch ran.
    assert getattr(pi05_mod.GemmaRMSNorm, "_lerobot_per_token_cond_patched", False)

    norm = pi05_mod.GemmaRMSNorm(dim=8, eps=1e-6, cond_dim=8)
    x = torch.randn(2, 4, 8)
    cond = torch.randn(2, 4, 8)
    out, gate = norm(x, cond=cond)
    assert out.shape == (2, 4, 8)
    assert gate is not None and gate.shape == (2, 4, 8)

    # Scalar-per-sample cond still broadcasts over tokens.
    out2, gate2 = norm(x, cond=torch.randn(2, 8))
    assert out2.shape == (2, 4, 8)
    assert gate2 is not None and gate2.shape == (2, 1, 8)
