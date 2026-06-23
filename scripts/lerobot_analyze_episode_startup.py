#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.

"""
Analyze idle / startup motion at the beginning of each episode in a LeRobot dataset.

Reads actions (and optionally states) from parquet only — no video decoding.

Examples:

```bash
lerobot-analyze-episode-startup \\
    --dataset screw_sorting_single_v30 \\
    --root /workspace/huggingface/lerobot/screw_sorting_single_v30

lerobot-analyze-episode-startup \\
    --dataset local/my_dataset \\
    --output-csv outputs/startup_stats.csv \\
    --thresholds 1e-4,1e-3,1e-2,0.05 \\
    --json
```
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_report import resolve_dataset_root
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE


@dataclass
class EpisodeStartupStats:
    episode_index: int
    episode_length: int
    idle_seconds_at_fps: float | None
    # Norm of action relative to the first frame (||a_t - a_0||).
    cum_norm_max: float
    cum_norm_p95_idle: float | None
    cum_norm_at_frame_1: float | None
    cum_norm_at_frame_5: float | None
    cum_norm_at_frame_10: float | None
    # Norm of action step delta (||a_t - a_{t-1}||), t >= 1.
    delta_norm_max: float
    delta_norm_p95_idle: float | None
    delta_norm_at_frame_1: float | None
    # First frame index where metric exceeds threshold (None = never in episode).
    first_motion_frame_cum: int | None
    first_motion_frame_delta: int | None
    first_motion_frame_state_cum: int | None
    # First-frame action vector summary.
    action0_l2_norm: float
    action0_abs_max: float
    action0_abs_mean: float


def _as_float_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32).flatten()


def _stack_episode(values: list[Any]) -> torch.Tensor:
    if not values:
        return torch.empty(0, 0)
    return torch.stack([_as_float_tensor(v) for v in values])


def _series_at(series: torch.Tensor, idx: int) -> float | None:
    if series.numel() == 0 or idx < 0 or idx >= series.numel():
        return None
    return float(series[idx].item())


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    weight = pos - lo
    return sorted_vals[lo] * (1.0 - weight) + sorted_vals[hi] * weight


def _find_first_above(series: torch.Tensor, threshold: float) -> int | None:
    if series.numel() == 0:
        return None
    above = (series > threshold).nonzero(as_tuple=False)
    if above.numel() == 0:
        return None
    return int(above[0, 0].item())


def _resolve_state_key(dataset: LeRobotDataset, state_key: str | None) -> str | None:
    if state_key is not None:
        if state_key not in dataset.meta.features:
            raise ValueError(
                f"State key '{state_key}' not found in dataset features: {list(dataset.meta.features)}"
            )
        return state_key
    if OBS_STATE in dataset.meta.features:
        return OBS_STATE
    return None


def analyze_episode(
    episode_index: int,
    actions: torch.Tensor,
    states: torch.Tensor | None,
    fps: int,
    cum_threshold: float,
    delta_threshold: float,
    state_threshold: float,
) -> EpisodeStartupStats:
    """Compute startup statistics for one episode."""
    ep_len = int(actions.shape[0])
    action0 = actions[0] if ep_len > 0 else torch.empty(0)

    if ep_len >= 2:
        cum_norms = torch.linalg.vector_norm(actions - action0.unsqueeze(0), dim=1)
        delta_norms = torch.linalg.vector_norm(actions[1:] - actions[:-1], dim=1)
    elif ep_len == 1:
        cum_norms = torch.zeros(1)
        delta_norms = torch.empty(0)
    else:
        cum_norms = torch.empty(0)
        delta_norms = torch.empty(0)

    first_motion_cum = _find_first_above(cum_norms, cum_threshold)
    first_motion_delta = _find_first_above(delta_norms, delta_threshold) if delta_norms.numel() else None
    if first_motion_delta is not None:
        # delta_norms[i] corresponds to frame i+1
        first_motion_delta += 1

    first_motion_state = None
    if states is not None and states.shape[0] >= 2:
        state0 = states[0]
        state_cum_norms = torch.linalg.vector_norm(states - state0.unsqueeze(0), dim=1)
        first_motion_state = _find_first_above(state_cum_norms, state_threshold)

    idle_end = first_motion_cum if first_motion_cum is not None else ep_len
    idle_cum = cum_norms[:idle_end].tolist() if cum_norms.numel() else []
    idle_delta = delta_norms[: max(idle_end - 1, 0)].tolist() if delta_norms.numel() else []

    return EpisodeStartupStats(
        episode_index=episode_index,
        episode_length=ep_len,
        idle_seconds_at_fps=(idle_end / fps) if fps > 0 else None,
        cum_norm_max=float(cum_norms.max().item()) if cum_norms.numel() else 0.0,
        cum_norm_p95_idle=_percentile(idle_cum, 0.95),
        cum_norm_at_frame_1=_series_at(cum_norms, 1),
        cum_norm_at_frame_5=_series_at(cum_norms, 5),
        cum_norm_at_frame_10=_series_at(cum_norms, 10),
        delta_norm_max=float(delta_norms.max().item()) if delta_norms.numel() else 0.0,
        delta_norm_p95_idle=_percentile(idle_delta, 0.95),
        delta_norm_at_frame_1=_series_at(delta_norms, 0),
        first_motion_frame_cum=first_motion_cum,
        first_motion_frame_delta=first_motion_delta,
        first_motion_frame_state_cum=first_motion_state,
        action0_l2_norm=float(torch.linalg.vector_norm(action0).item()) if action0.numel() else 0.0,
        action0_abs_max=float(action0.abs().max().item()) if action0.numel() else 0.0,
        action0_abs_mean=float(action0.abs().mean().item()) if action0.numel() else 0.0,
    )


def _summarize(values: list[float | None]) -> dict[str, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p90": _percentile(clean, 0.90),
        "p95": _percentile(clean, 0.95),
    }


def _summarize_int(values: list[int | None]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None]
    never = sum(1 for v in values if v is None)
    summary = _summarize(clean)
    summary["never_triggered"] = never
    return summary


def _resolve_repo_id(dataset_root: Path, root: Path | None) -> str:
    base_root = root.expanduser().resolve() if root is not None else HF_LEROBOT_HOME.resolve()
    if dataset_root.is_relative_to(base_root):
        return dataset_root.relative_to(base_root).as_posix()
    return f"local/{dataset_root.name}"


def analyze_dataset(
    dataset: LeRobotDataset,
    episodes: list[int] | None,
    cum_threshold: float,
    delta_threshold: float,
    state_threshold: float,
    state_key: str | None,
    sweep_thresholds: list[float] | None,
) -> dict[str, Any]:
    if ACTION not in dataset.meta.features:
        raise ValueError(f"Dataset has no '{ACTION}' feature. Available: {list(dataset.meta.features)}")

    resolved_state_key = _resolve_state_key(dataset, state_key)
    fps = int(dataset.meta.fps)
    hf_ds = dataset.hf_dataset

    total_episodes = dataset.meta.total_episodes
    episode_indices = list(range(total_episodes)) if episodes is None else episodes

    per_episode: list[EpisodeStartupStats] = []
    sweep_rows: list[dict[str, Any]] = []
    action_dim: int | None = None

    for ep_idx in episode_indices:
        ep = dataset.meta.episodes[ep_idx]
        from_idx = int(ep["dataset_from_index"])
        to_idx = int(ep["dataset_to_index"])

        actions = _stack_episode(hf_ds[ACTION][from_idx:to_idx])
        if action_dim is None and actions.ndim == 2:
            action_dim = int(actions.shape[1])
        states = None
        if resolved_state_key is not None:
            states = _stack_episode(hf_ds[resolved_state_key][from_idx:to_idx])

        stats = analyze_episode(
            episode_index=ep_idx,
            actions=actions,
            states=states,
            fps=fps,
            cum_threshold=cum_threshold,
            delta_threshold=delta_threshold,
            state_threshold=state_threshold,
        )
        per_episode.append(stats)

        if sweep_thresholds and actions.shape[0] >= 1:
            action0 = actions[0]
            cum_norms = (
                torch.linalg.vector_norm(actions - action0.unsqueeze(0), dim=1)
                if actions.shape[0] >= 1
                else torch.empty(0)
            )
            delta_norms = (
                torch.linalg.vector_norm(actions[1:] - actions[:-1], dim=1)
                if actions.shape[0] >= 2
                else torch.empty(0)
            )
            state_cum_norms = None
            if states is not None and states.shape[0] >= 1:
                state0 = states[0]
                state_cum_norms = torch.linalg.vector_norm(states - state0.unsqueeze(0), dim=1)

            for threshold in sweep_thresholds:
                delta_hit = _find_first_above(delta_norms, threshold) if delta_norms.numel() else None
                sweep_rows.append(
                    {
                        "episode_index": ep_idx,
                        "threshold": threshold,
                        "first_motion_frame_cum": _find_first_above(cum_norms, threshold),
                        "first_motion_frame_delta": (delta_hit + 1) if delta_hit is not None else None,
                        "first_motion_frame_state_cum": (
                            _find_first_above(state_cum_norms, threshold)
                            if state_cum_norms is not None
                            else None
                        ),
                    }
                )

    idle_frames_cum = [ep.first_motion_frame_cum for ep in per_episode]
    idle_frames_delta = [ep.first_motion_frame_delta for ep in per_episode]
    idle_frames_state = [ep.first_motion_frame_state_cum for ep in per_episode]
    idle_seconds = [ep.idle_seconds_at_fps for ep in per_episode]

    # Recommend threshold: slightly above p95 of idle-period cum norms across all episodes.
    idle_cum_p95_samples = [ep.cum_norm_p95_idle for ep in per_episode if ep.cum_norm_p95_idle is not None]
    idle_delta_p95_samples = [ep.delta_norm_p95_idle for ep in per_episode if ep.delta_norm_p95_idle is not None]

    recommended_cum = (
        _percentile(idle_cum_p95_samples, 0.95) * 1.5 if idle_cum_p95_samples else None
    )
    recommended_delta = (
        _percentile(idle_delta_p95_samples, 0.95) * 1.5 if idle_delta_p95_samples else None
    )

    return {
        "repo_id": dataset.repo_id,
        "fps": fps,
        "action_dim": action_dim,
        "state_key": resolved_state_key,
        "thresholds_used": {
            "cum": cum_threshold,
            "delta": delta_threshold,
            "state": state_threshold,
        },
        "recommended_thresholds": {
            "cum_from_idle_p95_x1.5": recommended_cum,
            "delta_from_idle_p95_x1.5": recommended_delta,
        },
        "summary": {
            "episodes_analyzed": len(per_episode),
            "idle_frames_cum": _summarize_int(idle_frames_cum),
            "idle_frames_delta": _summarize_int(idle_frames_delta),
            "idle_frames_state_cum": _summarize_int(idle_frames_state),
            "idle_seconds_cum": _summarize(idle_seconds),
            "cum_norm_p95_idle": _summarize(idle_cum_p95_samples),
            "delta_norm_p95_idle": _summarize(idle_delta_p95_samples),
            "action0_l2_norm": _summarize([ep.action0_l2_norm for ep in per_episode]),
            "action0_abs_max": _summarize([ep.action0_abs_max for ep in per_episode]),
        },
        "episodes": [asdict(ep) for ep in per_episode],
        "threshold_sweep": sweep_rows,
    }


def _print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    rec = report["recommended_thresholds"]
    thr = report["thresholds_used"]

    print(f"Dataset: {report['repo_id']}  fps={report['fps']}")
    if report["state_key"]:
        print(f"State key: {report['state_key']}")
    print(f"Episodes analyzed: {summary['episodes_analyzed']}")
    print()
    print("Thresholds used for first-motion detection:")
    print(f"  cum   (||a_t - a_0||)         > {thr['cum']}")
    print(f"  delta (||a_t - a_(t-1)||)     > {thr['delta']}")
    print(f"  state (||s_t - s_0||)         > {thr['state']}")
    print()
    print("Recommended thresholds (1.5x p95 of idle-period noise):")
    print(f"  cum:   {rec['cum_from_idle_p95_x1.5']}")
    print(f"  delta: {rec['delta_from_idle_p95_x1.5']}")
    print()

    def _print_block(title: str, block: dict[str, float | None]) -> None:
        print(title)
        if block.get("count", 0) == 0:
            print("  (no data)")
            return
        print(
            f"  min={block['min']:.4g}  median={block['median']:.4g}  "
            f"mean={block['mean']:.4g}  p90={block['p90']:.4g}  p95={block['p95']:.4g}  max={block['max']:.4g}"
        )
        if "never_triggered" in block:
            print(f"  never triggered: {int(block['never_triggered'])}")

    _print_block("Idle frames until cum motion (first ||a_t-a_0|| > threshold):", summary["idle_frames_cum"])
    _print_block("Idle frames until delta motion (first ||a_t-a_{t-1}|| > threshold):", summary["idle_frames_delta"])
    if report["state_key"]:
        _print_block("Idle frames until state motion (first ||s_t-s_0|| > threshold):", summary["idle_frames_state_cum"])
    _print_block("Idle duration [s] at dataset fps (based on cum metric):", summary["idle_seconds_cum"])
    _print_block("Idle-period cum norm p95 per episode:", summary["cum_norm_p95_idle"])
    _print_block("Idle-period delta norm p95 per episode:", summary["delta_norm_p95_idle"])
    _print_block("First-frame action L2 norm:", summary["action0_l2_norm"])
    _print_block("First-frame action |max|:", summary["action0_abs_max"])

    print()
    print("Per-episode (first 10):")
    header = (
        f"{'ep':>4} {'len':>5} {'idle_cum':>9} {'idle_delta':>11} "
        f"{'idle_s':>7} {'cum_p95':>8} {'dlt_p95':>8} {'|a0|':>8}"
    )
    print(header)
    print("-" * len(header))
    for ep in report["episodes"][:10]:
        print(
            f"{ep['episode_index']:4d} {ep['episode_length']:5d} "
            f"{str(ep['first_motion_frame_cum']):>9} {str(ep['first_motion_frame_delta']):>11} "
            f"{ep['idle_seconds_at_fps'] or 0:7.2f} "
            f"{ep['cum_norm_p95_idle'] or 0:8.4g} "
            f"{ep['delta_norm_p95_idle'] or 0:8.4g} "
            f"{ep['action0_l2_norm']:8.4g}"
        )
    if summary["episodes_analyzed"] > 10:
        print(f"... and {summary['episodes_analyzed'] - 10} more episodes")


def _write_episode_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not episodes:
        return
    fieldnames = list(episodes[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episodes)


def _write_sweep_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_episodes(value: str, total: int) -> list[int]:
    value = value.strip().lower()
    if value in {"all", "*"}:
        return list(range(total))
    indices: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return sorted(set(i for i in indices if 0 <= i < total))


def _parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze startup / idle motion at the beginning of LeRobot dataset episodes."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset repo id or local path (same as lerobot-dataset-report).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=f"Base directory for dataset lookup. Default: {HF_LEROBOT_HOME}",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episodes to analyze, e.g. 'all', '0-9', '0,2,5'.",
    )
    parser.add_argument(
        "--cum-threshold",
        type=float,
        default=1e-4,
        help="Threshold on ||a_t - a_0|| to detect first motion.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=1e-4,
        help="Threshold on ||a_t - a_{t-1}|| to detect first motion.",
    )
    parser.add_argument(
        "--state-threshold",
        type=float,
        default=1e-4,
        help="Threshold on ||s_t - s_0|| when observation.state is available.",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default=None,
        help=f"State feature key. Default: auto-detect '{OBS_STATE}' if present.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Optional comma-separated thresholds for sweep CSV, e.g. '1e-4,1e-3,0.01'.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Write per-episode statistics to this CSV file.",
    )
    parser.add_argument(
        "--output-sweep-csv",
        type=Path,
        default=None,
        help="Write threshold sweep results to this CSV (requires --thresholds).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report to stdout.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 before loading the dataset.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    dataset_root = resolve_dataset_root(args.dataset, args.root)
    repo_id = _resolve_repo_id(dataset_root, args.root)

    dataset = LeRobotDataset(repo_id, root=dataset_root)
    episode_indices = _parse_episodes(args.episodes, dataset.meta.total_episodes)
    sweep_thresholds = _parse_float_list(args.thresholds) if args.thresholds else None

    report = analyze_dataset(
        dataset=dataset,
        episodes=episode_indices,
        cum_threshold=args.cum_threshold,
        delta_threshold=args.delta_threshold,
        state_threshold=args.state_threshold,
        state_key=args.state_key,
        sweep_thresholds=sweep_thresholds,
    )

    _print_report(report)

    if args.output_csv is not None:
        _write_episode_csv(args.output_csv, report["episodes"])
        print(f"\nWrote per-episode CSV: {args.output_csv}")

    if args.output_sweep_csv is not None:
        if not sweep_thresholds:
            raise ValueError("--output-sweep-csv requires --thresholds")
        _write_sweep_csv(args.output_sweep_csv, report["threshold_sweep"])
        print(f"Wrote threshold sweep CSV: {args.output_sweep_csv}")

    if args.json:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
