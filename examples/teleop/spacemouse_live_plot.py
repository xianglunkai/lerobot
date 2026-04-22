#!/usr/bin/env python3
"""Live SpaceMouse plot example.

Connects to a real SpaceMouse via `pyspacemouse`, polls its state in a background
thread and draws translation/rotation axes in real time using Matplotlib.

Usage:
    python examples/teleop/spacemouse_live_plot.py [--duration 60] [--history 600] [--save out.png]

Notes:
 - Run this script from a terminal (main thread) so Matplotlib can open a GUI window.
 - If `pyspacemouse` is not available or the device is not connected, the script will exit with an error.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np

logger = logging.getLogger("spacemouse_live")


def run_live_plot(duration: float = 60.0, history: int = 600, save: Optional[str] = None):
    try:
        import pyspacemouse
    except Exception as e:
        logger.error("pyspacemouse is required for this example. Install it and ensure the device is connected.")
        raise

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception as e:
        logger.error("matplotlib is required to display live plots")
        raise

    # Shared buffer for the reader thread -> animation
    buf_lock = threading.Lock()
    state_buffer: Deque[tuple[float, dict]] = deque(maxlen=history)

    stop_event = threading.Event()

    # Open device
    device = pyspacemouse.open()
    if not device:
        raise RuntimeError("Failed to open SpaceMouse device via pyspacemouse.open()")

    def reader_loop():
        logger.info("SpaceMouse reader thread started")
        while not stop_event.is_set():
            try:
                s = device.read()
            except Exception:
                logger.exception("Error reading from SpaceMouse (device disconnected?)")
                stop_event.set()
                break

            # Build a simple dict of signal values we want to visualize
            sample = {
                "x": float(getattr(s, "x", 0.0)),
                "y": float(getattr(s, "y", 0.0)),
                "z": float(getattr(s, "z", 0.0)),
                "roll": float(getattr(s, "roll", 0.0)),
                "pitch": float(getattr(s, "pitch", 0.0)),
                "yaw": float(getattr(s, "yaw", 0.0)),
            }
            with buf_lock:
                state_buffer.append((time.time(), sample))

            # Sleep a little: driver typically updates ~100-200Hz; reduce CPU by a short sleep
            time.sleep(0.002)

        logger.info("SpaceMouse reader thread exiting")

    reader = threading.Thread(target=reader_loop, daemon=True, name="SpaceMouseReader")
    reader.start()

    # Matplotlib figure setup
    action_names = ["x", "y", "z", "roll", "pitch", "yaw"]
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
    fig.suptitle("SpaceMouse live axes")

    start_time = time.time()

    def update(frame):
        # Read buffer copy under lock
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

        # keep x window to recent history seconds
        if len(timestamps) > 0:
            x_min = max(0.0, timestamps[-1] - max(5.0, duration / 10.0))
            for ax in axes:
                ax.set_xlim(x_min, timestamps[-1] + 0.1)

        return lines

    anim = FuncAnimation(fig, update, interval=50, blit=False)

    logger.info("Opening live plot window (close the window or press Ctrl-C to stop)")
    try:
        plt.show()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    # Shutdown
    stop_event.set()
    reader.join(timeout=1.0)

    # Optional save
    if save:
        try:
            fig.savefig("spacemouse_plot.png")
            logger.info("Saved plot to spacemouse_plot.png")
        except Exception:
            logger.exception("Failed to save figure")

    # Close device if pyspacemouse has any cleanup (not strictly required)
    try:
        if hasattr(pyspacemouse, "close"):
            pyspacemouse.close()
    except Exception:
        logger.exception("Error closing pyspacemouse")


def main():
    parser = argparse.ArgumentParser(description="SpaceMouse live plot example")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds (unused; window close stops)")
    parser.add_argument("--history", type=int, default=60000, help="Number of samples to keep in history")
    parser.add_argument("--save", type=bool, default=True, help="Whether to save final figure")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        run_live_plot(duration=args.duration, history=args.history, save=args.save)
    except Exception as e:
        logger.exception(f"Live plot failed: {e}")


if __name__ == "__main__":
    main()
