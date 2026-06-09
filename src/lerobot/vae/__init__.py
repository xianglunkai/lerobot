# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""RTR action-chunk VAE ported for LeRobot latent-space policies."""

from lerobot.vae.configuration import ActionVAEConfig
from lerobot.vae.factory import (
    compute_latent_action_stats,
    fit_action_normalizer,
    load_action_vae,
    make_action_vae,
    save_vae_checkpoint,
)
from lerobot.vae.latent_inference import LatentActionPipeline, LatentInferenceConfig
from lerobot.vae.latent_policy import (
    configure_policy_for_latent_training,
    decode_latent_policy_actions,
    encode_actions_to_latent_batch,
    latent_reconstruction_l1,
    resolve_temporal_downsample_ratio,
)
from lerobot.vae.latent_rtc_queue import LatentRTRActionQueue
from lerobot.vae.modeling_action_vae import ActionVAE, VAE
from lerobot.vae.rtr_refiner import RTRConfig, RTRRefiner
from lerobot.vae.normalizer import LinearNormalizer

__all__ = [
    "ActionVAE",
    "ActionVAEConfig",
    "LatentActionPipeline",
    "LatentInferenceConfig",
    "LatentRTRActionQueue",
    "LinearNormalizer",
    "RTRConfig",
    "RTRRefiner",
    "VAE",
    "compute_latent_action_stats",
    "configure_policy_for_latent_training",
    "decode_latent_policy_actions",
    "encode_actions_to_latent_batch",
    "fit_action_normalizer",
    "latent_reconstruction_l1",
    "load_action_vae",
    "make_action_vae",
    "resolve_temporal_downsample_ratio",
    "save_vae_checkpoint",
]
