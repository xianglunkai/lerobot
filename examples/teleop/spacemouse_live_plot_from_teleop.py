#!/usr/bin/env python3
"""Live plot example using SpacemouseTeleop outputs.

This script constructs `SpacemouseTeleop` from the library, connects the
device, starts a background reader that calls `get_action()` (so the teleop's
own background reader stays useful) and appends the returned action dicts to a
shared deque. The main thread runs Matplotlib's FuncAnimation to render the
latest history in real time. Close the window or Ctrl-C to exit.

Usage:
    python examples/teleop/spacemouse_live_plot_from_teleop.py [--history 600] [--save out.png]

Run from a terminal (main thread) to allow Matplotlib to open a GUI window.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpacemouseTeleop

logger = logging.getLogger("spacemouse_teleop_plot")


try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except Exception as e:
    logger.error("matplotlib is required to display live plots")
    raise


def run_from_teleop(duration: float, history: int = 600, save: Optional[str] = None):
    # Instantiate config via the class attribute on SpacemouseTeleop
    config_cls = SpacemouseTeleop.config_class
    config = config_cls()

    teleop = SpacemouseTeleop(config)
    teleop.connect()

    if not teleop.is_connected():
        raise RuntimeError("SpacemouseTeleop failed to connect to device")

    # Circular numpy buffer populated by reader thread for efficient reads in update()
    buf_lock = threading.Lock()
    state_buffer: Deque[tuple[float, dict]] = deque(maxlen=history)
    
    stop_event = threading.Event()

    def reader_loop():
        logger.info("Teleop reader thread started")
        while not stop_event.is_set():
            # Simple and robust: call teleop.get_action() and append returned
            # action dict converted to ordered numeric vector. This mirrors what
            # users expect and keeps the example minimal.
            try:
               
                action = teleop.get_action()
               
            except Exception:
                logger.exception("Error calling teleop.get_action(); stopping reader")
                stop_event.set()
                break

            with buf_lock:
                state_buffer.append((time.time(), action))


            # Sleep a little: driver typically updates ~100-200Hz; reduce CPU by a short sleep
            time.sleep(0.02)

        logger.info("Teleop reader thread exiting")

    reader = threading.Thread(target=reader_loop, daemon=True, name="TeleopReader")
    reader.start()

    # Matplotlib figure setup
    # action_features is a property returning a dict; get the names map and
    # build an ordered list of names by their index
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
        line, = ax.plot([], [], linewidth=2, color="tab:red")
        ax.set_ylabel(action_names[i])
        ax.grid(alpha=0.3)
        lines.append(line)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Teleop live axes")

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
        if hasattr(teleop, "disconnect"):
            teleop.disconnect()
    except Exception:
        logger.exception("Error closing teleop")

def main():
    parser = argparse.ArgumentParser(description="SpacemouseTeleop live plot")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds (unused; window close stops)")
    parser.add_argument("--history", type=int, default=60000, help="Number of samples to keep in history")
    parser.add_argument("--save", type=bool, default=True, help="Whether to save final figure")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        run_from_teleop(duration=args.duration, history=args.history, save=args.save)
    except Exception:
        logger.exception("Live plot from teleop failed")


if __name__ == "__main__":
    main()
