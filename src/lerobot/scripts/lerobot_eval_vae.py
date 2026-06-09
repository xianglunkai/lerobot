#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""Evaluate a trained action VAE and optionally compute latent-action statistics."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.vae_train import VAETrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging
from lerobot.vae import compute_latent_action_stats, load_action_vae
from lerobot.vae.pytorch_util import dict_apply


@dataclass
class VAEEvalPipelineConfig(VAETrainPipelineConfig):
    vae_checkpoint_path: str | None = None
    get_latent_statistics: bool = False
    eval_steps: int = 500
    latent_stats_path: str | None = None


def make_vae_dataset(cfg: VAETrainPipelineConfig):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    delta_timestamps = {ACTION: [i / ds_meta.fps for i in range(cfg.vae.horizon)]}
    return LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        tolerance_s=cfg.tolerance_s,
    )


@parser.wrap()
def eval_vae(cfg: VAEEvalPipelineConfig) -> None:
    register_third_party_plugins()
    cfg.validate()
    init_logging()

    if cfg.vae_checkpoint_path is None:
        raise ValueError("`vae_checkpoint_path` is required for VAE evaluation.")

    device = torch.device(cfg.vae.device if cfg.vae.device != "cuda" else "cuda:0")
    cfg.vae.device = str(device)

    dataset = make_vae_dataset(cfg)
    vae = load_action_vae(cfg.vae, cfg.vae_checkpoint_path)

    if cfg.latent_stats_path is not None:
        vae._load_latent_dataset_statistics(cfg.latent_stats_path)

    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=cfg.vae.horizon - 1,
        shuffle=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    total_losses: list[float] = []
    latencies: list[float] = []
    steps = len(dataloader) if cfg.get_latent_statistics else min(cfg.eval_steps, len(dataloader))

    for step, batch in enumerate(tqdm(dataloader, total=steps)):
        if step >= steps:
            break

        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True) if torch.is_tensor(x) else x)
        action = batch[ACTION]

        start = time.perf_counter()
        with torch.no_grad():
            predicted = vae.encode_then_decode(batch)
        latencies.append(time.perf_counter() - start)

        l1 = torch.mean(torch.abs(predicted - action)).item()
        total_losses.append(l1)

    mean_loss = sum(total_losses) / max(len(total_losses), 1)
    mean_latency = sum(latencies) / max(len(latencies), 1)
    logging.info("VAE eval | mean L1 recon loss=%.6f steps=%d latency=%.4fs", mean_loss, len(total_losses), mean_latency)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "dataset_stats.json"

    stats = dict(dataset.meta.stats)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if cfg.get_latent_statistics:
        latent_stats = compute_latent_action_stats(vae, dataloader, device=device, max_batches=steps)
        stats["latent_action"] = latent_stats
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logging.info("Saved latent action statistics to %s", stats_path)


def main() -> None:
    register_third_party_plugins()
    eval_vae()


if __name__ == "__main__":
    main()
