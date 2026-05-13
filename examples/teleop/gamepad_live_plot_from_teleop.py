#!/usr/bin/env python3
"""Live plot example using GamepadTeleop outputs.

This mirrors the spacemouse example but uses the gamepad teleop implementation.
Usage:
    python examples/teleop/gamepad_live_plot_from_teleop.py [--history 600] [--save out.png]
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from lerobot.teleoperators.gamepad.teleop_gamepad import GamepadTeleop

logger = logging.getLogger("gamepad_teleop_plot")


try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except Exception:
    logger.error("matplotlib is required to display live plots")
    raise


def run_from_teleop(duration: float, history: int = 600, save: Optional[str] = None):
    config_cls = GamepadTeleop.config_class
    config = config_cls()

    teleop = GamepadTeleop(config)
    teleop.connect()

    if not teleop.is_connected:
        raise RuntimeError("GamepadTeleop failed to connect to device")

    buf_lock = threading.Lock()
    state_buffer: Deque[tuple[float, dict]] = deque(maxlen=history)
    stop_event = threading.Event()

    def reader_loop():
        logger.info("Gamepad reader thread started")
        while not stop_event.is_set():
            try:
                action = teleop.get_action()
            except Exception:
                logger.exception("Error calling teleop.get_action(); stopping reader")
                stop_event.set()
                break

            with buf_lock:
                state_buffer.append((time.time(), action))

            time.sleep(0.01)
        logger.info("Gamepad reader thread exiting")

    reader = threading.Thread(target=reader_loop, daemon=True, name="GamepadReader")
    reader.start()

    # Build plotting arrays from teleop.action_features
    names_map = teleop.action_features["names"]
    ordered_names = [None] * len(names_map)
    for k, idx in names_map.items():
        ordered_names[idx] = k
    action_names = ordered_names
    num_plots = len(action_names)

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]
    lines = []
    for i, ax in enumerate(axes):
        line, = ax.plot([], [], linewidth=2, color="tab:blue")
        ax.set_ylabel(action_names[i])
        ax.grid(alpha=0.3)
        lines.append(line)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Gamepad Teleop live axes")

    start_time = time.time()

    def update(frame):
        with buf_lock:
            data = list(state_buffer)
        if not data:
            return lines

        timestamps = np.array([t - start_time for t, _ in data], dtype=float)
        values = np.stack([np.array([d[name] for _, d in data], dtype=float) for name in action_names], axis=0)

        for i, line in enumerate(lines):
            line.set_data(timestamps, values[i])
            axes[i].relim()
            axes[i].autoscale_view(True, True, True)

        if len(timestamps) > 0:
            x_min = max(0.0, timestamps[-1] - max(5.0, duration / 10.0))
            for ax in axes:
                ax.set_xlim(x_min, timestamps[-1] + 0.1)

        return lines

    anim = FuncAnimation(fig, update, interval=50, blit=False)

    logger.info("Opening gamepad live plot window (close to stop)")
    try:
        plt.show()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    stop_event.set()
    reader.join(timeout=1.0)

    if save:
        try:
            fig.savefig(save)
            logger.info("Saved plot to %s", save)
        except Exception:
            logger.exception("Failed to save figure")

    try:
        if hasattr(teleop, "disconnect"):
            teleop.disconnect()
    except Exception:
        logger.exception("Error closing teleop")


def main():
    parser = argparse.ArgumentParser(description="GamepadTeleop live plot")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--history", type=int, default=6000)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        run_from_teleop(duration=args.duration, history=args.history, save=args.save)
    except Exception:
        logger.exception("Live plot from gamepad teleop failed")


if __name__ == "__main__":
    main()
