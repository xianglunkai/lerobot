#!/usr/bin/env python3
"""Simulate multi-joint signals and plot Kalman filter results.

Save plot to `outputs/kalman_simulation.png`.
"""
import os
import numpy as np
import time
import matplotlib.pyplot as plt

from lerobot.processor.filter_processor import KalmanFilterProcessor
from lerobot.processor.core import TransitionKey


def run_simulation_and_plot(n_joints=6, duration=10.0, dt=0.02, noise_std=0.1, lp_cutoff=3.0):
    steps = int(duration / dt)
    t = np.arange(steps) * dt

    joint_names = [f"joint_{i}" for i in range(n_joints)]

    true_signals = {}
    measurements = {}

    rng = np.random.RandomState(0)
    for i, name in enumerate(joint_names):
        amp = 0.5 + 0.1 * i
        freq = 0.5
        phase = i * 0.2
        true = amp * np.sin(2 * np.pi * freq * t + phase)
        meas = true + rng.normal(scale=noise_std, size=steps)
        true_signals[name] = true
        measurements[name] = meas

    proc = KalmanFilterProcessor(process_noise=1, measurement_noise=1, dt=dt, adaptive_R=True)

    estimates = {name: np.zeros(steps) for name in joint_names}
    # Simple per-joint first-order low-pass filter state
    lp_estimates = {name: np.zeros(steps) for name in joint_names}
    lp_last: dict[str, float] = {}
    # compute low-pass alpha from cutoff frequency (Hz)
    if lp_cutoff <= 0.0:
        lp_alpha = 1.0
    else:
        rc = 1.0 / (2 * np.pi * lp_cutoff)
        lp_alpha = dt / (dt + rc)

    for step in range(steps):
        meas_dict = {name: float(measurements[name][step]) for name in joint_names}
        transition = {TransitionKey.OBSERVATION: meas_dict}
        t0 = time.perf_counter()
        out = proc(transition)
        comp = out.get(TransitionKey.COMPLEMENTARY_DATA.value, {})
        states = comp.get("kalman_states", {})
        for name in joint_names:
            estimates[name][step] = states[name]["angle"]
            # low-pass filter the noisy measurement for comparison
            meas_val = float(measurements[name][step])
            if name not in lp_last:
                lp_last[name] = meas_val
            lp_last[name] = lp_alpha * meas_val + (1.0 - lp_alpha) * lp_last[name]
            lp_estimates[name][step] = lp_last[name]
        t1 = time.perf_counter()
        # print(f"Step {step+1}/{steps}, KF update time: {(t1 - t0)*1000:.2f} ms")

    # Compute RMSE
    rmse_raw_per_joint = [np.sqrt(np.mean((measurements[n] - true_signals[n]) ** 2)) for n in joint_names]
    rmse_kf_per_joint = [np.sqrt(np.mean((estimates[n] - true_signals[n]) ** 2)) for n in joint_names]
    rmse_lp_per_joint = [np.sqrt(np.mean((lp_estimates[n] - true_signals[n]) ** 2)) for n in joint_names]
    rmse_raw = float(np.mean(rmse_raw_per_joint))
    rmse_kf = float(np.mean(rmse_kf_per_joint))
    rmse_lp = float(np.mean(rmse_lp_per_joint))
    print(f"RMSE raw: {rmse_raw:.6f}, RMSE KF: {rmse_kf:.6f}, RMSE LP: {rmse_lp:.6f}")

    # Plot results: grid of subplots
    ncols = 2
    nrows = (n_joints + 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for idx, name in enumerate(joint_names):
        ax = axes[idx]
        ax.plot(t, true_signals[name], color="#0d11e7", linewidth=1.0, label="true")
        ax.plot(t, measurements[name], color="#d6c80a", linestyle="dotted", linewidth=1.0, label="noisy")
        ax.plot(t, lp_estimates[name], color="#17becf", linestyle="-.", linewidth=1.0, label=f"lp (cutoff={lp_cutoff}Hz)")
        ax.plot(t, estimates[name], color="#e2112c", linestyle="--", linewidth=1.0, label="kf")
        ax.set_title(name)
        if idx == 0:
            ax.legend()
        ax.grid(True, linestyle=":", linewidth=0.5)

    # Hide any unused axes
    for j in range(n_joints, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Kalman Filter Simulation: {n_joints} joints, dt={dt}, duration={duration}s\nRMSE raw={rmse_raw:.4f}, RMSE KF={rmse_kf:.4f}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "kalman_simulation.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    run_simulation_and_plot()
