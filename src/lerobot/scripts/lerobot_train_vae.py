#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""Train the RTR action-chunk VAE on a LeRobot dataset."""

import json
import logging
import time
from pathlib import Path
from pprint import pformat

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.vae_train import VAE_TRAIN_CONFIG_NAME, VAETrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import format_big_number, init_logging
from lerobot.vae.factory import fit_action_normalizer, make_action_vae, save_vae_checkpoint
from lerobot.vae.modeling_action_vae import ActionVAE
from lerobot.vae.pytorch_util import dict_apply


def make_vae_dataset(cfg: VAETrainPipelineConfig) -> LeRobotDataset:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

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


def update_vae(
    train_metrics: MetricsTracker,
    vae: ActionVAE,
    batch: dict,
    optimizer: Optimizer,
    grad_clip_norm: float,
    lr_scheduler=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    vae.train()

    loss_dict = vae.compute_loss_and_metric(batch)
    loss = loss_dict["loss"]
    if not torch.is_tensor(loss):
        loss = torch.as_tensor(loss, device=vae.device)

    optimizer.zero_grad()
    loss.backward()

    if grad_clip_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(vae.optim_params, grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(vae.optim_params, float("inf"), error_if_nonfinite=False)

    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time

    output_dict = {
        k: float(v) if isinstance(v, (int, float)) else float(v.item()) if hasattr(v, "item") else float(v)
        for k, v in loss_dict.items()
        if k not in {"loss", "vq_code"}
    }
    return train_metrics, output_dict


@parser.wrap()
def train_vae(cfg: VAETrainPipelineConfig) -> None:
    register_third_party_plugins()
    cfg.validate()
    init_logging()

    logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = torch.device(cfg.vae.device if cfg.vae.device != "cuda" else "cuda:0")
    cfg.vae.device = str(device)

    dataset = make_vae_dataset(cfg)
    vae = make_action_vae(cfg.vae)

    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=cfg.vae.horizon - 1,
        shuffle=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    logging.info("Fitting VAE action normalizer...")
    fit_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=min(cfg.num_workers, 4),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    fit_action_normalizer(vae, fit_loader, max_batches=cfg.fit_normalizer_batches, device=device)

    optimizer = cfg.optimizer.build(vae.optim_params)
    lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler is not None else None

    step = 0
    if cfg.resume and cfg.checkpoint_path is not None:
        payload = torch.load(cfg.checkpoint_path, weights_only=False, map_location="cpu")
        vae.load_state_dict(payload)
        meta_path = Path(cfg.checkpoint_path).parent / VAE_TRAIN_CONFIG_NAME
        if meta_path.is_file():
            with meta_path.open(encoding="utf-8") as f:
                step = int(json.load(f).get("step", 0))

    start_step = step
    dl_iter = cycle(dataloader)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")

    train_metrics = {
        "loss": AverageMeter("loss", ":.4f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
    )

    for global_step in tqdm(range(start_step, cfg.steps)):
        batch = next(dl_iter)
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True) if torch.is_tensor(x) else x)
        train_tracker, _ = update_vae(
            train_tracker,
            vae,
            batch,
            optimizer,
            cfg.grad_clip_norm,
            lr_scheduler,
        )

        if global_step % cfg.log_freq == 0:
            logging.info(train_tracker)

        if cfg.save_freq > 0 and global_step > 0 and global_step % cfg.save_freq == 0:
            ckpt_dir = Path(cfg.output_dir) / "checkpoints"
            stats = dict(dataset.meta.stats)
            save_vae_checkpoint(vae, ckpt_dir, step=global_step, dataset_stats=stats)
            with (ckpt_dir / VAE_TRAIN_CONFIG_NAME).open("w", encoding="utf-8") as f:
                json.dump(
                    {"step": global_step, "vae": cfg.vae.__dict__, "dataset": cfg.dataset.repo_id},
                    f,
                    indent=2,
                )

    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    stats = dict(dataset.meta.stats)
    save_vae_checkpoint(vae, ckpt_dir, dataset_stats=stats)
    with (ckpt_dir / VAE_TRAIN_CONFIG_NAME).open("w", encoding="utf-8") as f:
        json.dump({"step": cfg.steps, "vae": cfg.vae.__dict__, "dataset": cfg.dataset.repo_id}, f, indent=2)
    logging.info(colored(f"Training complete. Checkpoint saved to {ckpt_dir / 'latest.ckpt'}", "green"))


def main() -> None:
    register_third_party_plugins()
    train_vae()


if __name__ == "__main__":
    main()
