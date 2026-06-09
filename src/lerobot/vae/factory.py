# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.utils.constants import ACTION

from lerobot.vae.configuration import ActionVAEConfig
from lerobot.vae.modeling_action_vae import ActionVAE
from lerobot.vae.normalizer import LinearNormalizer


def make_action_vae(cfg: ActionVAEConfig) -> ActionVAE:
    cfg.validate()
    return ActionVAE(
        horizon=cfg.horizon,
        shape_meta=cfg.shape_meta,
        n_latent_dims=cfg.n_latent_dims,
        mlp_layer_num=cfg.mlp_layer_num,
        use_conv_encoder=cfg.use_conv_encoder,
        conv_latent_dims=cfg.conv_latent_dims,
        conv_layer_num=cfg.conv_layer_num,
        use_rnn_decoder=cfg.use_rnn_decoder,
        rnn_latent_dims=cfg.rnn_latent_dims,
        rnn_layer_num=cfg.rnn_layer_num,
        use_vq=cfg.use_vq,
        n_embed=cfg.n_embed,
        vqvae_groups=cfg.vqvae_groups,
        kl_multiplier=cfg.kl_multiplier,
        eval=cfg.eval_mode,
        device=cfg.device,
        encoder_loss_multiplier=cfg.encoder_loss_multiplier,
        act_scale=cfg.act_scale,
        use_conv_decoder=cfg.use_conv_decoder,
        second_stage=cfg.second_stage,
    )


def load_action_vae(cfg: ActionVAEConfig, checkpoint_path: str | Path) -> ActionVAE:
    vae = make_action_vae(cfg)
    payload = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    vae.load_state_dict(payload)
    vae.to(cfg.device)
    vae.eval()
    return vae


def fit_action_normalizer(
    vae: ActionVAE,
    dataloader: DataLoader,
    *,
    max_batches: int = 256,
    device: torch.device | str | None = None,
) -> LinearNormalizer:
    """Fit the VAE's internal normalizer on action chunks from a dataloader."""
    device = device or vae.device
    actions: list[torch.Tensor] = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        action = batch[ACTION]
        if not torch.is_tensor(action):
            action = torch.as_tensor(action)
        actions.append(action.to(device))

    if not actions:
        raise ValueError("No action batches found to fit the VAE normalizer.")

    stacked = torch.cat(actions, dim=0)
    normalizer = LinearNormalizer()
    normalizer.fit({ACTION: stacked}, last_n_dims=1, mode="limits")
    vae.set_normalizer(normalizer)
    return normalizer


def save_vae_checkpoint(
    vae: ActionVAE,
    output_dir: Path,
    *,
    step: int | None = None,
    dataset_stats: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = "latest.ckpt" if step is None else f"{step:06d}.ckpt"
    ckpt_path = output_dir / ckpt_name
    payload = {"state_dicts": {"model": vae.state_dict()}}
    torch.save(payload, ckpt_path)

    if dataset_stats is not None:
        stats_path = output_dir / "dataset_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(dataset_stats, f, indent=2)

    return ckpt_path


def compute_latent_action_stats(
    vae: ActionVAE,
    dataloader: DataLoader,
    *,
    device: torch.device | str,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Encode dataset actions and compute latent-action statistics for latent policy training."""
    latent_chunks: list[np.ndarray] = []
    vae.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            action = batch[ACTION].to(device)
            latent = vae.encode_to_latent(action)
            arr = latent.detach().cpu().numpy()
            if arr.ndim == 3:
                arr = arr.reshape(-1, arr.shape[-1])
            elif arr.ndim == 2:
                pass
            else:
                raise ValueError(f"Unexpected latent shape: {arr.shape}")
            latent_chunks.append(arr)

    if not latent_chunks:
        raise ValueError("No batches available to compute latent action statistics.")

    latent_all = np.concatenate(latent_chunks, axis=0)
    q01, q10, q50, q90, q99 = np.percentile(latent_all, [1, 10, 50, 90, 99], axis=0)
    return {
        "count": [int(latent_all.shape[0])],
        "min": latent_all.min(axis=0).astype(np.float64).tolist(),
        "max": latent_all.max(axis=0).astype(np.float64).tolist(),
        "mean": latent_all.mean(axis=0).astype(np.float64).tolist(),
        "std": latent_all.std(axis=0, ddof=0).astype(np.float64).tolist(),
        "q01": q01.astype(np.float64).tolist(),
        "q10": q10.astype(np.float64).tolist(),
        "q50": q50.astype(np.float64).tolist(),
        "q90": q90.astype(np.float64).tolist(),
        "q99": q99.astype(np.float64).tolist(),
    }
