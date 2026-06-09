# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""Latent-policy inference: VAE decode + optional RTR refine."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from lerobot.vae.configuration import ActionVAEConfig
from lerobot.vae.factory import load_action_vae
from lerobot.vae.latent_policy import decode_latent_policy_actions
from lerobot.vae.modeling_action_vae import ActionVAE
from lerobot.vae.rtr_refiner import RTRConfig, RTRRefiner


@dataclass
class LatentInferenceConfig:
    vae: ActionVAEConfig = field(default_factory=ActionVAEConfig)
    vae_checkpoint_path: str | None = None
    latent_dataset_statistics: str | None = None
    latent_normalization_type: str = "NORMAL"
    temporal_downsample_ratio: int = 4
    rtr: RTRConfig = field(default_factory=RTRConfig)

    def validate(self) -> None:
        self.vae.validate()
        if self.vae_checkpoint_path is None:
            raise ValueError("`vae_checkpoint_path` is required.")
        if self.latent_dataset_statistics is None:
            raise ValueError("`latent_dataset_statistics` is required.")


class LatentActionPipeline:
    """Decode latent policy outputs to robot actions with optional RTR refinement."""

    def __init__(
        self,
        vae: ActionVAE,
        *,
        normalization_type: str = "NORMAL",
        rtr_config: RTRConfig | None = None,
    ):
        self.vae = vae
        self.normalization_type = normalization_type
        self.rtr_refiner = RTRRefiner(vae, rtr_config or RTRConfig(enabled=False))

    @classmethod
    def from_config(cls, cfg: LatentInferenceConfig, device: str | torch.device) -> LatentActionPipeline:
        cfg.validate()
        cfg.vae.device = str(device)
        vae = load_action_vae(cfg.vae, cfg.vae_checkpoint_path)
        vae._load_latent_dataset_statistics(cfg.latent_dataset_statistics)
        return cls(vae, normalization_type=cfg.latent_normalization_type, rtr_config=cfg.rtr)

    def reset(self) -> None:
        self.rtr_refiner.reset()

    @torch.no_grad()
    def decode_latent_actions(self, latent_actions: torch.Tensor) -> torch.Tensor:
        """Decode policy latent output [B,T,D] to action chunk [B,H,A] (no RTR refine)."""
        return decode_latent_policy_actions(
            self.vae,
            latent_actions,
            normalization_type=self.normalization_type,
        )

    @torch.no_grad()
    def latent_to_actions(
        self,
        latent_actions: torch.Tensor,
        *,
        queue_leftover: torch.Tensor | None = None,
        inference_delay: int | None = None,
    ) -> torch.Tensor:
        """Decode latent predictions and RTR-refine using queue leftover + new suffix.

        Args:
            latent_actions: Policy output after inference completes.
            queue_leftover: Remaining decoded actions in the queue **before** inference
                started (``get_processed_left_over()``). The first ``inference_delay``
                steps are reused and concatenated with ``decoded[:, inference_delay:]``.
            inference_delay: Steps consumed during inference (from ``LatencyTracker`` /
                queue index diff).
        """
        action_chunk = self.decode_latent_actions(latent_actions)
        reuse_prefix = None
        if queue_leftover is not None and inference_delay is not None and inference_delay > 0:
            reuse_prefix = queue_leftover[:inference_delay]
        return self.rtr_refiner.refine(
            action_chunk,
            reuse_actions=reuse_prefix,
            inference_delay=inference_delay,
        )

    @property
    def action_horizon(self) -> int:
        return self.vae.input_dim_h

    @property
    def latent_horizon(self) -> int:
        return self.vae.downsampled_input_h
