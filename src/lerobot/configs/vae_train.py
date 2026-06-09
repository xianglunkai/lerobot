#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import draccus

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.vae.configuration import ActionVAEConfig

VAE_TRAIN_CONFIG_NAME = "vae_train_config.json"


@dataclass
class VAETrainPipelineConfig:
    dataset: DatasetConfig
    vae: ActionVAEConfig = field(default_factory=ActionVAEConfig)

    output_dir: Path | None = None
    job_name: str | None = None
    resume: bool = False
    seed: int | None = 1000

    num_workers: int = 4
    batch_size: int = 64
    steps: int = 50_000
    log_freq: int = 200
    save_freq: int = 10_000
    tolerance_s: float = 1e-4

    grad_clip_norm: float = 1.0
    fit_normalizer_batches: int = 256

    optimizer: AdamWConfig = field(default_factory=lambda: AdamWConfig(lr=1e-3, weight_decay=1e-4))
    scheduler: DiffuserSchedulerConfig | None = field(
        default_factory=lambda: DiffuserSchedulerConfig(name="cosine", num_warmup_steps=100)
    )
    wandb: WandBConfig = field(default_factory=WandBConfig)

    checkpoint_path: Path | None = field(init=False, default=None)

    def validate(self) -> None:
        self.vae.validate()

        if not self.job_name:
            self.job_name = "action_vae"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/vae_train") / train_dir

    def to_dict(self) -> dict:
        return draccus.encode(self)  # type: ignore[no-any-return]
