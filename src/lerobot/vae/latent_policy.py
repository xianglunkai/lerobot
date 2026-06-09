# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""Helpers for training VLA policies in VAE latent action space (RTR-style)."""

from __future__ import annotations

from typing import Any

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import ACTION

from lerobot.vae.configuration import ActionVAEConfig
from lerobot.vae.modeling_action_vae import ActionVAE


def resolve_temporal_downsample_ratio(
    vae_cfg: ActionVAEConfig,
    override: int | None = None,
) -> int:
    ratio = vae_cfg.temporal_downsample_ratio if override is None else override
    if ratio <= 0:
        raise ValueError("`temporal_downsample_ratio` must be positive.")
    return ratio


def configure_policy_for_latent_training(
    policy_cfg: PreTrainedConfig,
    vae_cfg: ActionVAEConfig,
    temporal_downsample_ratio: int,
    *,
    match_rtr_n_action_steps: bool = True,
) -> None:
    """Shrink policy action horizon/dim to match VAE latent space (RTR recipe)."""
    if policy_cfg.chunk_size % temporal_downsample_ratio != 0:
        raise ValueError(
            f"policy.chunk_size={policy_cfg.chunk_size} must be divisible by "
            f"temporal_downsample_ratio={temporal_downsample_ratio}."
        )
    if policy_cfg.chunk_size != vae_cfg.horizon:
        raise ValueError(
            f"policy.chunk_size ({policy_cfg.chunk_size}) must match vae.horizon ({vae_cfg.horizon}) "
            "before latent downsampling."
        )

    policy_cfg.chunk_size = policy_cfg.chunk_size // temporal_downsample_ratio
    if match_rtr_n_action_steps:
        # Matches RTR third_party/lerobot/scripts/codes/train_pi0.5/lerobot_train_latent_rdp_vae.py
        policy_cfg.n_action_steps = policy_cfg.chunk_size // temporal_downsample_ratio
    else:
        policy_cfg.n_action_steps = policy_cfg.chunk_size

    latent_action_dim = vae_cfg.n_embed
    if getattr(policy_cfg, "max_action_dim", None) is not None:
        policy_cfg.max_action_dim = max(policy_cfg.max_action_dim, latent_action_dim)


@torch.no_grad()
def encode_actions_to_latent_batch(
    vae: ActionVAE,
    batch: dict[str, Any],
    *,
    normalization_type: str = "NORMAL",
) -> torch.Tensor:
    """Encode raw action chunks and normalize in latent statistics space."""
    latent_action = vae.encode_to_latent(batch[ACTION])
    return vae.normalize_from_dataset(latent_action, is_latent=True, normalization_type=normalization_type)


@torch.no_grad()
def decode_latent_policy_actions(
    vae: ActionVAE,
    latent_actions: torch.Tensor,
    *,
    normalization_type: str = "NORMAL",
) -> torch.Tensor:
    """Decode policy latent predictions back to executable action chunks."""
    latent = vae.denormalize_from_dataset(latent_actions, is_latent=True, normalization_type=normalization_type)
    return vae.decode_from_latent(latent)


def latent_reconstruction_l1(
    vae: ActionVAE,
    predicted_latent: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    *,
    normalization_type: str = "NORMAL",
) -> float:
    decoded = decode_latent_policy_actions(vae, predicted_latent, normalization_type=normalization_type)
    return torch.mean(torch.abs(decoded - ground_truth_actions)).item()
