#!/usr/bin/env python

"""Simple Ruckig simulation examples.

Run:
    python examples/teleop/ruckig_simple_sim.py

Optional:
    python examples/teleop/ruckig_simple_sim.py --plot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrajectoryLog:
    times: list[float]
    positions: list[list[float]]
    velocities: list[list[float]]
    accelerations: list[list[float]]


def run_ruckig(
    dof: int,
    dt: float,
    current_position: list[float],
    target_position: list[float],
    max_velocity: list[float],
    max_acceleration: list[float],
    max_jerk: list[float],
) -> TrajectoryLog:
    """Run one online trajectory generation from current to target."""
    try:
        from ruckig import InputParameter, OutputParameter, Result, Ruckig
    except Exception as exc:
        raise RuntimeError(
            "ruckig is not installed. Install with: pip install ruckig"
        ) from exc

    otg = Ruckig(dof, dt)
    inp = InputParameter(dof)
    out = OutputParameter(dof)

    inp.current_position = current_position
    inp.current_velocity = [0.0] * dof
    inp.current_acceleration = [0.0] * dof

    inp.target_position = target_position
    inp.target_velocity = [0.0] * dof
    inp.target_acceleration = [0.0] * dof

    inp.max_velocity = max_velocity
    inp.max_acceleration = max_acceleration
    inp.max_jerk = max_jerk

    t = 0.0
    times: list[float] = [t]
    positions: list[list[float]] = [list(inp.current_position)]
    velocities: list[list[float]] = [list(inp.current_velocity)]
    accelerations: list[list[float]] = [list(inp.current_acceleration)]

    while True:
        result = otg.update(inp, out)
        if result not in (Result.Working, Result.Finished):
            raise RuntimeError(f"Ruckig update failed with status: {result}")

        t += dt
        times.append(t)
        positions.append(list(out.new_position))
        velocities.append(list(out.new_velocity))
        accelerations.append(list(out.new_acceleration))

        out.pass_to_input(inp)
        if result == Result.Finished:
            break

    return TrajectoryLog(
        times=times,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
    )


def print_summary(name: str, log: TrajectoryLog) -> None:
    """Print compact trajectory summary."""
    duration = log.times[-1]
    final_pos = log.positions[-1]
    steps = len(log.times)
    print(f"\n=== {name} ===")
    print(f"steps: {steps}")
    print(f"duration: {duration:.3f}s")
    print(f"final position: {[round(v, 6) for v in final_pos]}")


def maybe_plot(log_1d: TrajectoryLog, log_6d: TrajectoryLog) -> None:
    """Optional plotting if matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed, skip plotting.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    titles = ["position", "velocity", "acceleration"]
    logs = [log_1d.positions, log_1d.velocities, log_1d.accelerations]
    for ax, title, ys in zip(axes, titles, logs):
        y = [row[0] for row in ys]
        ax.plot(log_1d.times, y, label=f"1D {title}")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].set_title("Ruckig 1D profile")

    plt.tight_layout()
    plt.show()

    # Print 6D final snapshot so user can confirm all axes converge.
    print("\n6D final states:")
    print(f"  q:   {[round(v, 6) for v in log_6d.positions[-1]]}")
    print(f"  qd:  {[round(v, 6) for v in log_6d.velocities[-1]]}")
    print(f"  qdd: {[round(v, 6) for v in log_6d.accelerations[-1]]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Ruckig simulation examples")
    parser.add_argument("--plot", action="store_true", help="Plot 1D position/velocity/acceleration")
    args = parser.parse_args()

    # Example 1: one joint from 0 -> 1 rad
    log_1d = run_ruckig(
        dof=1,
        dt=1.0 / 200.0,
        current_position=[0.0],
        target_position=[1.0],
        max_velocity=[1.2],
        max_acceleration=[2.5],
        max_jerk=[8.0],
    )
    print_summary("1D Joint Move", log_1d)

    # Example 2: six-joint move (similar to robot arm joints)
    log_6d = run_ruckig(
        dof=6,
        dt=1.0 / 200.0,
        current_position=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        target_position=[0.4, -0.2, 0.6, -0.5, 0.3, -0.1],
        max_velocity=[1.4, 1.4, 1.2, 1.8, 1.8, 2.0],
        max_acceleration=[3.0, 3.0, 2.5, 4.0, 4.0, 4.5],
        max_jerk=[12.0, 12.0, 10.0, 16.0, 16.0, 18.0],
    )
    print_summary("6D Joint Move", log_6d)

    if args.plot:
        maybe_plot(log_1d, log_6d)


if __name__ == "__main__":
    main()
