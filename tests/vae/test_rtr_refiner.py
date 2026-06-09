# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import torch

from lerobot.utils.constants import ACTION
from lerobot.vae.configuration import ActionVAEConfig
from lerobot.vae.factory import make_action_vae
from lerobot.vae.normalizer import LinearNormalizer
from lerobot.vae.rtr_refiner import RTRConfig, RTRRefiner


def _make_vae(horizon: int = 48, action_dim: int = 7) -> torch.nn.Module:
    cfg = ActionVAEConfig(
        horizon=horizon,
        action_dim=action_dim,
        n_embed=10,
        n_latent_dims=8,
        use_conv_encoder=True,
        conv_latent_dims=32,
        conv_layer_num=1,
        device="cpu",
    )
    vae = make_action_vae(cfg)
    normalizer = LinearNormalizer()
    normalizer.fit({ACTION: torch.randn(4, horizon, action_dim)}, last_n_dims=1, mode="limits")
    vae.set_normalizer(normalizer)
    return vae


def test_rtr_refiner_paper_first_chunk_passthrough():
    vae = _make_vae()
    refiner = RTRRefiner(vae, RTRConfig(enabled=True, mode="paper"))
    chunk = torch.randn(1, 48, 7)
    out = refiner.refine(chunk, reuse_actions=None, inference_delay=0)
    assert torch.allclose(out, chunk)


def test_rtr_refiner_paper_concat_with_variable_delay():
    vae = _make_vae()
    refiner = RTRRefiner(vae, RTRConfig(enabled=True, mode="paper"))
    reuse = torch.randn(5, 7)
    new_chunk = torch.randn(1, 48, 7)
    out = refiner.refine(new_chunk, reuse_actions=reuse, inference_delay=5)
    assert out.shape == (1, 48, 7)
    assert not torch.allclose(out, new_chunk)


def test_rtr_refiner_released_mode():
    vae = _make_vae()
    refiner = RTRRefiner(vae, RTRConfig(enabled=True, mode="released", reuse_action_num=24))
    chunk_a = torch.randn(1, 48, 7)
    chunk_b = torch.randn(1, 48, 7)
    refiner.refine(chunk_a)
    out_b = refiner.refine(chunk_b)
    assert out_b.shape == (1, 48, 7)
    assert not torch.allclose(out_b, chunk_b)


def test_rtr_refiner_disabled():
    vae = _make_vae()
    refiner = RTRRefiner(vae, RTRConfig(enabled=False, mode="paper"))
    chunk = torch.randn(1, 48, 7)
    out = refiner.refine(chunk, reuse_actions=torch.randn(3, 7), inference_delay=3)
    assert torch.allclose(out, chunk)
