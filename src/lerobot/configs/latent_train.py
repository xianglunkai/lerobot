#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.train import TrainPipelineConfig
from lerobot.vae.configuration import ActionVAEConfig


@dataclass
class LatentTrainPipelineConfig(TrainPipelineConfig):
    """Train a VLA policy in VAE latent action space (RTR-style)."""

    vae: ActionVAEConfig = field(default_factory=ActionVAEConfig)
    vae_checkpoint_path: str | None = None
    latent_dataset_statistics: str | None = None
    temporal_downsample_ratio: int | None = None
    latent_normalization_type: str = "NORMAL"
    match_rtr_n_action_steps: bool = True

    def validate(self) -> None:
        super().validate()
        self.vae.validate()

        if self.vae_checkpoint_path is None:
            raise ValueError("`vae_checkpoint_path` is required for latent policy training.")
        if not Path(self.vae_checkpoint_path).is_file():
            raise ValueError(f"VAE checkpoint not found: {self.vae_checkpoint_path}")

        if self.latent_dataset_statistics is None:
            raise ValueError(
                "`latent_dataset_statistics` is required (JSON with `latent_action` stats). "
                "Generate it via `lerobot-eval-vae --get_latent_statistics=true`."
            )
        if not Path(self.latent_dataset_statistics).is_file():
            raise ValueError(f"Latent statistics file not found: {self.latent_dataset_statistics}")

        if self.policy is None:
            raise ValueError("Policy config is required for latent training.")

        supported = {"pi05", "pi0", "smolvla"}
        if self.policy.type not in supported:
            raise ValueError(
                f"Latent training currently supports {sorted(supported)}, got '{self.policy.type}'."
            )
