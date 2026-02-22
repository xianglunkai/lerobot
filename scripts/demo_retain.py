#!/usr/bin/env python3
"""Demo for RETAIN-style parameter merging on a toy policy.

This script trains a small "generalist" policy on diverse synthetic tasks,
then finetunes it on a narrow target-task dataset (small-n). It then merges
weights via linear interpolation and compares performance (MSE) on three sets:
  - ID (in-distribution) : held-out examples from finetune distribution
  - OOD (out-of-distribution) : shifted parameters for the same task
  - Generalist : held-out from pretraining tasks

Purpose: illustrate how simple weight-space merging can improve OOD generalization
and retain generalist capability (as shown in the RETAIN paper).
"""
from __future__ import annotations

import os
import math
import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lerobot.finetune.retain import merge_state_dicts


class ToyPolicy(nn.Module):
    def __init__(self, obs_dim: int = 8, hidden: int = 64, out_dim: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.head(h)


def make_task_dataset(freq: float, phase: float, amp: float, n: int, obs_dim: int, noise: float = 0.05):
    # simple synthetic mapping: obs -> 2-d action where obs encodes time-like variable
    t = np.linspace(0.0, 1.0, n)
    obs = np.stack([np.sin(2 * math.pi * (freq * t + phase + i * 0.1)) for i in range(obs_dim)], axis=1)
    # actions are two sinusoids derived from combination
    a1 = amp * np.sin(2 * math.pi * freq * t + phase)
    a2 = amp * np.cos(2 * math.pi * freq * t + phase)
    actions = np.stack([a1, a2], axis=1)
    obs = obs + noise * np.random.randn(*obs.shape)
    actions = actions + noise * np.random.randn(*actions.shape)
    return torch.tensor(obs, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)


def train_model(model: nn.Module, data: TensorDataset, epochs: int = 50, lr: float = 1e-3) -> None:
    loader = DataLoader(data, batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()


def eval_mse(model: nn.Module, data: TensorDataset) -> float:
    loader = DataLoader(data, batch_size=128, shuffle=False)
    loss_fn = nn.MSELoss(reduction="mean")
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            l = loss_fn(pred, yb).item()
            tot += l * xb.shape[0]
            n += xb.shape[0]
    return float(tot / n)


def main():
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    obs_dim = 8
    out_dim = 2

    # Pretraining: build many tasks (diverse freqs/amps)
    pre_datasets = []
    for freq in [1.0, 1.5, 2.0, 2.5]:
        for amp in [0.6, 0.8]:
            phase = random.uniform(0.0, 1.0)
            obs, acts = make_task_dataset(freq=freq, phase=phase, amp=amp, n=200, obs_dim=obs_dim)
            pre_datasets.append(TensorDataset(obs, acts))

    # Consolidate pretraining data
    pre_obs = torch.cat([d.tensors[0] for d in pre_datasets], dim=0)
    pre_acts = torch.cat([d.tensors[1] for d in pre_datasets], dim=0)
    pre_data = TensorDataset(pre_obs, pre_acts)

    # Train generalist (simulated pretrained model)
    generalist = ToyPolicy(obs_dim=obs_dim, out_dim=out_dim)
    train_model(generalist, pre_data, epochs=150, lr=3e-3)

    # Finetune dataset: single target freq/amp with limited samples
    target_freq = 1.2
    target_amp = 0.9
    # small narrow finetune set (50 samples) - narrow phase variation
    obs_ft, acts_ft = make_task_dataset(freq=target_freq, phase=0.1, amp=target_amp, n=50, obs_dim=obs_dim)
    ft_data = TensorDataset(obs_ft, acts_ft)

    # ID and OOD evaluation sets
    # ID: held-out samples from same target distribution (different phase)
    obs_id, acts_id = make_task_dataset(freq=target_freq, phase=0.15, amp=target_amp, n=200, obs_dim=obs_dim)
    id_data = TensorDataset(obs_id, acts_id)
    # OOD: same task but different freq/amp
    obs_ood, acts_ood = make_task_dataset(freq=1.8, phase=0.3, amp=0.7, n=200, obs_dim=obs_dim)
    ood_data = TensorDataset(obs_ood, acts_ood)

    # Generalist holdout (from pretraining tasks)
    gen_obs, gen_acts = make_task_dataset(freq=2.3, phase=0.4, amp=0.75, n=200, obs_dim=obs_dim)
    gen_data = TensorDataset(gen_obs, gen_acts)

    # Finetune a copy of the generalist on the small dataset
    finetuned = ToyPolicy(obs_dim=obs_dim, out_dim=out_dim)
    finetuned.load_state_dict(generalist.state_dict())
    train_model(finetuned, ft_data, epochs=200, lr=1e-3)

    # Evaluate
    pre_mse_id = eval_mse(generalist, id_data)
    pre_mse_ood = eval_mse(generalist, ood_data)
    pre_mse_gen = eval_mse(generalist, gen_data)

    ft_mse_id = eval_mse(finetuned, id_data)
    ft_mse_ood = eval_mse(finetuned, ood_data)
    ft_mse_gen = eval_mse(finetuned, gen_data)

    # Merge weights (global alpha sweep) and evaluate
    alphas = [0.25, 0.5, 0.75]
    merged_results = []
    pre_sd = {k: v.cpu() for k, v in generalist.state_dict().items()}
    ft_sd = {k: v.cpu() for k, v in finetuned.state_dict().items()}
    for a in alphas:
        merged_sd = merge_state_dicts(pre_sd, ft_sd, alpha=a)
        merged = ToyPolicy(obs_dim=obs_dim, out_dim=out_dim)
        merged.load_state_dict({k: merged_sd[k] for k in merged_sd})
        merged_results.append((a, eval_mse(merged, id_data), eval_mse(merged, ood_data), eval_mse(merged, gen_data)))

    # Print summary
    print("Pretrained MSE (ID, OOD, GEN):", pre_mse_id, pre_mse_ood, pre_mse_gen)
    print("Finetuned MSE (ID, OOD, GEN):", ft_mse_id, ft_mse_ood, ft_mse_gen)
    for a, mid, mood, mgen in merged_results:
        print(f"Merged alpha={a}: MSE ID={mid:.4f}, OOD={mood:.4f}, GEN={mgen:.4f}")

    # Plot ID/OOD comparison for the best alpha by OOD
    best = min(merged_results, key=lambda r: r[2])
    best_alpha = best[0]
    merged_sd = merge_state_dicts(pre_sd, ft_sd, alpha=best_alpha)
    merged = ToyPolicy(obs_dim=obs_dim, out_dim=out_dim)
    merged.load_state_dict({k: merged_sd[k] for k in merged_sd})

    # sample time series predictions for ID and OOD
    with torch.no_grad():
        id_pred = merged(id_data.tensors[0]).cpu().numpy()
        id_true = id_data.tensors[1].cpu().numpy()
        ood_pred = merged(ood_data.tensors[0]).cpu().numpy()
        ood_true = ood_data.tensors[1].cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    t = np.arange(id_pred.shape[0])
    axes[0].plot(t, id_true[:, 0], label="true id", color="k")
    axes[0].plot(t, id_pred[:, 0], label="merged pred id", color="C0")
    axes[0].set_title("ID predictions (dim0)")
    axes[0].legend()

    t2 = np.arange(ood_pred.shape[0])
    axes[1].plot(t2, ood_true[:, 0], label="true ood", color="k")
    axes[1].plot(t2, ood_pred[:, 0], label="merged pred ood", color="C1")
    axes[1].set_title("OOD predictions (dim0)")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "retain_demo.png")
    plt.savefig(out_path, dpi=160)
    print("Saved plot to", out_path)


if __name__ == "__main__":
    main()
