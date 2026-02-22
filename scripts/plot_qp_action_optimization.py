#!/usr/bin/env python3
"""Simulate VLA actions, apply EMA and QP-style optimizer, and plot results.

Usage: run from repository root:
    python3 scripts/plot_qp_action_optimization.py
"""
import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt


from lerobot.policies.pi05.modeling_pi05 import (
    
    intra_chunk_smoothing_vla_rail,
    optimize_actions_qp_with_constraints,
)


def ema_smooth(actions: torch.Tensor, s: float):
    if s <= 0.0:
        return actions
    alpha = 1.0 / (1.0 + float(s))
    B, T, A = actions.shape
    out = torch.empty_like(actions, dtype=torch.float32)
    out[:, 0, :] = actions[:, 0, :].to(torch.float32)
    one_minus = 1.0 - alpha
    for t in range(1, T):
        out[:, t, :] = alpha * actions[:, t, :].to(torch.float32) + one_minus * out[:, t - 1, :]
    return out.to(actions.dtype)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    import time

    B, T, A = 1, 60, 6
    device = torch.device("cpu")

    # Create ground-truth smooth trajectories (different freq/scale per-dim)
    t = torch.linspace(0.0, 1.0, T, device=device)
    base = torch.zeros(B, T, A, device=device)
    for a in range(A):
        freq = 1.0 + 0.5 * a
        amp = 0.8 - 0.08 * a
        base[0, :, a] = amp * torch.sin(2 * math.pi * freq * t) + 0.2 * a * t

    # Add Gaussian noise + a few impulse outliers
    noise = 0.08 * torch.randn(B, T, A, device=device)
    actions_noisy = base + noise
    # add spikes at random times
    spikes_idx = [8, 22, 41]
    for si in spikes_idx:
        actions_noisy[0, si, :] += torch.tensor([0.8, -0.6, 0.4, -0.5, 0.7, -0.3], device=device)

    # EMA baseline
    ema_actions = ema_smooth(actions_noisy, s=2.0)

    # intra-chunk polynomial smoother baseline
    poly_actions = intra_chunk_smoothing_vla_rail(actions_noisy, polynomial_order=3, preserve_boundaries=True)

    # QP optimizer settings
    dt = 0.02
    vel_limits = (-1e6, 1e6)
    acc_limits = (-1e6, 1e6)
    w_acc = 5
    w_jerk = 1

    t0 = time.perf_counter()
    qp_actions = optimize_actions_qp_with_constraints(
        actions_noisy,
        dt=dt,
        w_data=1.0,
        w_acc=w_acc,
        w_jerk=w_jerk,
        vel_limits=None,
        acc_limits=None,
        fix_ends=False,
        verbose=False,
    )
    tf = time.perf_counter()
    print(f"QP optimization took {1000 * (tf - t0):.2f} ms")

    # Diagnostic prints to help investigate scale/oscillation issues
    print("QP actions min/max:", float(qp_actions.min()), float(qp_actions.max()))
    try:
        print("QP actions sample [first dim, first 8]:", qp_actions[0, :8, 0].cpu().numpy())
        print("Noisy actions sample [first dim, first 8]:", actions_noisy[0, :8, 0].cpu().numpy())
    except Exception:
        pass

    # Metrics
    def rmse(x, y):
        return float(torch.sqrt(torch.mean((x - y) ** 2)).cpu().numpy())

    print("RMSE noisy vs base:", rmse(actions_noisy, base))
    print("RMSE ema vs base:", rmse(ema_actions, base))
    print("RMSE poly vs base:", rmse(poly_actions, base))
    print("RMSE qp vs base:", rmse(qp_actions, base))

    # check constraints
    v_qp = (qp_actions[:, 1:, :] - qp_actions[:, :-1, :]) / dt
    acc_qp = (qp_actions[:, 2:, :] - 2 * qp_actions[:, 1:-1, :] + qp_actions[:, :-2, :]) / (dt ** 2)
    print("QP velocity min/max:", float(v_qp.min()), float(v_qp.max()))
    print("QP acceleration min/max:", float(acc_qp.min()), float(acc_qp.max()))

    # plot first 3 dims
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = {
        "base": "k",
        "noisy": "b",
        "ema": "C1",
        "poly": "C2",
        "qp": "r",
    }

    dims = min(3, A)
    time = np.arange(T) * dt
    for i in range(dims):
        ax = axes[i]
        ax.plot(time, base[0, :, i].cpu().numpy(), label="base", color=colors["base"], linestyle="--", alpha=0.7, linewidth=1.5)
        ax.plot(time, actions_noisy[0, :, i].cpu().numpy(), label="noisy", color=colors["noisy"], linewidth=1.5)
        ax.plot(time, ema_actions[0, :, i].cpu().numpy(), label="ema", color=colors["ema"], linestyle="--", alpha=0.7)
        ax.plot(time, poly_actions[0, :, i].cpu().numpy(), label="poly", color=colors["poly"], linestyle="--", alpha=0.7)
        ax.plot(time, qp_actions[0, :, i].cpu().numpy(), label="qp", color=colors["qp"], linewidth=2.0)
        ax.set_ylabel(f"dim {i}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "qp_action_optimization.png")
    plt.savefig(out_path, dpi=180)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
