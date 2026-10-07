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
# See the License for the specific language governing permissions and
# limitations under the License.
"""Read and edit the official language columns on one local dataset."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.annotations.steerable_pipeline.writer import speech_atom
from lerobot.datasets.io_utils import (
    load_episodes,
    load_info,
    write_info,
    write_table_one_row_group_per_episode,
)
from lerobot.datasets.language import (
    LANGUAGE_EVENTS,
    LANGUAGE_PERSISTENT,
    column_for_style,
    language_feature_info,
    validate_camera_field,
)

PERSISTENT_STYLES = ("subtask", "plan", "memory", "task_aug", "motion")
EVENT_STYLES = ("interjection", "vqa", "trace", None)
EDITOR_EVENT_STYLES = ("interjection", "vqa", "trace", "say")
_EPISODE_LINE = re.compile(r"(\w+) episode (\d+)/(\d+) \(idx=(\d+)\) done in ([\d.]+)s")
_PHASE_LINE = re.compile(r"phase=(\w+) starting on (\d+)")
_WROTE_LINE = re.compile(r"wrote (\d+) shard")


def list_episodes(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    info = load_info(root)
    episodes = load_episodes(root)
    annotated = _annotated_flags(root, episodes)
    rows = []
    for index in range(len(episodes)):
        episode = episodes[index]
        episode_index = int(episode["episode_index"])
        tasks = episode.get("tasks") or []
        task = tasks if isinstance(tasks, str) else (str(tasks[0]) if tasks else "")
        rows.append(
            {
                "index": episode_index,
                "length": int(episode["length"]),
                "task": task,
                "annotated": annotated.get(episode_index, False),
            }
        )
    video_keys = [key for key, spec in info.features.items() if spec.get("dtype") == "video"]
    return {
        "root": str(root),
        "fps": int(info.fps),
        "total_episodes": len(rows),
        "video_keys": video_keys,
        "has_language_columns": LANGUAGE_PERSISTENT in info.features,
        "episodes": rows,
    }


def dataset_overview(root: Path) -> dict[str, Any]:
    """Episode list plus the on-disk layout policies read from ``meta/info.json``."""
    summary = list_episodes(root)
    info = load_info(root.expanduser().resolve())
    summary.update(
        {
            "codebase_version": info.codebase_version,
            "robot_type": info.robot_type,
            "total_frames": int(info.total_frames),
            "total_tasks": int(info.total_tasks),
            "splits": dict(info.splits),
            "data_path": info.data_path,
            "video_path": info.video_path,
            "features": {
                key: {
                    "dtype": spec.get("dtype"),
                    "shape": list(spec.get("shape") or []),
                    "names": spec.get("names"),
                }
                for key, spec in info.features.items()
            },
            "tasks": sorted({row["task"] for row in summary["episodes"] if row["task"]}),
            "files": list_files(root),
        }
    )
    return summary


def list_files(root: Path) -> list[dict[str, Any]]:
    root = root.expanduser().resolve()
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".annotate_staging" in path.parts:
            continue
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size})
    return rows


def parquet_page(
    root: Path,
    relative: str,
    *,
    offset: int = 0,
    limit: int = 25,
    episode_index: int | None = None,
) -> dict[str, Any]:
    """One page of a dataset parquet file. Cells keep every value; the page scrolls sideways."""
    root = root.expanduser().resolve()
    path = _safe_parquet(root, relative)
    schema = [{"name": field.name, "type": str(field.type)} for field in pq.read_schema(path)]
    if episode_index is None:
        indices = list(range(pq.read_metadata(path).num_rows))
    else:
        ids = pq.read_table(path, columns=["episode_index"]).column(0).to_pylist()
        indices = [i for i, episode_id in enumerate(ids) if int(episode_id) == episode_index]
    offset = max(0, offset)
    limit = min(max(1, limit), 200)
    chosen = indices[offset : offset + limit]
    table = pq.read_table(path).take(chosen) if chosen else pq.read_table(path).slice(0, 0)
    rows = []
    for position, raw in enumerate(table.to_pylist()):
        full = _json_value(raw)
        rows.append(
            {
                "index": chosen[position],
                "cells": {key: _cell(value) for key, value in full.items()},
                "raw": full,
            }
        )
    return {
        "path": relative,
        "rows_total": len(indices),
        "offset": offset,
        "limit": limit,
        "schema": schema,
        "rows": rows,
        "bytes": path.stat().st_size,
    }


def staging_progress(root: Path) -> dict[str, Any]:
    """Structured staging rows: what was labeled, and whether parquet already has it."""
    root = root.expanduser().resolve()
    staging = root / ".annotate_staging"
    modules: dict[str, list[int]] = {"plan": [], "interjections": [], "vqa": []}
    episodes: list[dict[str, Any]] = []
    style_counts: Counter[str] = Counter()
    style_episodes: dict[str, set[int]] = defaultdict(set)
    latest_episode = -1
    latest_rows: list[dict[str, Any]] = []
    if staging.is_dir():
        for episode_dir in sorted(path for path in staging.glob("episode_*") if path.is_dir()):
            episode_index = int(episode_dir.name.removeprefix("episode_"))
            present = dict.fromkeys(modules, False)
            rows: list[dict[str, Any]] = []
            for name in modules:
                path = episode_dir / f"{name}.jsonl"
                if not path.is_file() or path.stat().st_size == 0:
                    continue
                present[name] = True
                modules[name].append(episode_index)
                staged_rows = _staged_rows(path, name, style_counts)
                rows.extend(staged_rows)
                for row in staged_rows:
                    style_episodes[row["style"]].add(episode_index)
            if episode_index >= latest_episode and rows:
                latest_episode = episode_index
                latest_rows = [
                    {
                        "module": row["module"],
                        "style": row["style"],
                        "timestamp": row["timestamp"],
                        "content": row["content"][:180],
                    }
                    for row in rows
                    if row["style"] in {"subtask", "plan", "memory", "interjection", "vqa"}
                ][-8:]
            if any(present.values()):
                episodes.append({"episode": episode_index, "modules": present, "rows": rows})
    info_episodes = None
    if (root / "meta" / "info.json").is_file():
        info_episodes = int(load_info(root).total_episodes)
    on_disk = _language_counts(root)
    return {
        "total_episodes": info_episodes,
        "modules": {name: {"done": len(ids), "episodes": ids} for name, ids in modules.items()},
        "parquet_written": bool(on_disk),
        "style_counts": dict(style_counts),
        "styles": {style: sorted(ids) for style, ids in sorted(style_episodes.items())},
        "episodes": episodes,
        "on_disk": {str(episode): counts for episode, counts in on_disk.items()},
        "latest": {"episode": latest_episode if latest_episode >= 0 else None, "rows": latest_rows},
    }


def parse_annotate_log(text: str) -> dict[str, Any]:
    phase = None
    phase_total = None
    completed: list[dict[str, Any]] = []
    wrote = None
    for line in text.splitlines():
        started = _PHASE_LINE.search(line)
        if started:
            phase = started.group(1)
            phase_total = int(started.group(2))
        done = _EPISODE_LINE.search(line)
        if done:
            completed.append(
                {
                    "phase": done.group(1),
                    "number": int(done.group(2)),
                    "total": int(done.group(3)),
                    "episode": int(done.group(4)),
                    "seconds": float(done.group(5)),
                }
            )
        written = _WROTE_LINE.search(line)
        if written:
            wrote = int(written.group(1))
    return {
        "phase": phase,
        "phase_total": phase_total,
        "completed": completed[-12:],
        "last": completed[-1] if completed else None,
        "wrote_shards": wrote,
        "finished": wrote is not None,
    }


def materialize_dataset(source: Path, mode: str, output: str | None) -> Path:
    """Return the dataset directory the annotator should write.

    ``overwrite`` keeps ``source`` and snapshots its parquet once.
    ``copy`` duplicates metadata and parquet into a new directory and links
    the video files, so the original dataset stays unchanged.
    """
    source = source.expanduser().resolve()
    if mode == "overwrite":
        backup_shard(source)
        return source
    if mode != "copy":
        raise ValueError("mode must be overwrite or copy")
    destination = (
        Path(output).expanduser().resolve() if output else source.parent / f"{source.name}_annotated"
    )
    if destination == source:
        raise ValueError("the generated dataset must be a different directory")
    if destination.exists():
        raise FileExistsError(f"{destination} already exists")
    shutil.copytree(
        source, destination, ignore=lambda directory, names: _copy_ignore(source, directory, names)
    )
    videos = source / "videos"
    if videos.is_dir():
        for path in videos.rglob("*"):
            if not path.is_file():
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path.resolve())
    return destination


def backup_shard(root: Path) -> None:
    """Copy the data parquet and info.json once, before the first language write."""
    destination = root / "meta" / "backup_before_language"
    if destination.exists():
        return
    destination.mkdir(parents=True)
    shutil.copy2(root / "meta" / "info.json", destination / "info.json")
    for path in (root / "data").rglob("*.parquet"):
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def load_episode(root: Path, episode_index: int) -> dict[str, Any]:
    root = root.expanduser().resolve()
    info = load_info(root)
    episodes = load_episodes(root)
    episode = _episode(episodes, episode_index)
    path = _data_path(root, info, episode)
    table = pq.read_table(path, columns=_read_columns(path))
    mask = [int(value) == episode_index for value in table.column("episode_index").to_pylist()]
    timestamps = [
        float(value) for value, keep in zip(table.column("timestamp").to_pylist(), mask, strict=True) if keep
    ]
    persistent: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    if LANGUAGE_PERSISTENT in table.column_names:
        persistent_col = table.column(LANGUAGE_PERSISTENT).to_pylist()
        events_col = table.column(LANGUAGE_EVENTS).to_pylist()
        first = True
        stored_timestamps = table.column("timestamp").to_pylist()
        for keep, ts, persistent_rows, event_rows in zip(
            mask, stored_timestamps, persistent_col, events_col, strict=True
        ):
            if not keep:
                continue
            if first:
                persistent = [_public_persistent(row) for row in persistent_rows or []]
                first = False
            for row in event_rows or []:
                events.append(_public_event(row, float(ts)))
    video = {}
    for key in [name for name, spec in info.features.items() if spec.get("dtype") == "video"]:
        video[key] = {
            "path": _video_relative(info, episode, key),
            "start": float(episode[f"videos/{key}/from_timestamp"]),
            "end": float(episode[f"videos/{key}/to_timestamp"]),
        }
    tasks = episode.get("tasks") or []
    task = tasks if isinstance(tasks, str) else (str(tasks[0]) if tasks else "")
    return {
        "episode_index": episode_index,
        "task": task,
        "timestamps": timestamps,
        "persistent": persistent,
        "events": events,
        "video": video,
    }


def save_episode(
    root: Path, episode_index: int, persistent: list[dict[str, Any]], events: list[dict[str, Any]]
) -> None:
    root = root.expanduser().resolve()
    info = load_info(root)
    episodes = load_episodes(root)
    episode = _episode(episodes, episode_index)
    path = _data_path(root, info, episode)
    table = pq.read_table(path)
    timestamps = [
        float(ts)
        for ep, ts in zip(
            table.column("episode_index").to_pylist(),
            table.column("timestamp").to_pylist(),
            strict=True,
        )
        if int(ep) == episode_index
    ]
    if not timestamps:
        raise ValueError(f"episode {episode_index} has no frames in {path}")
    persistent_rows = [_normalize_persistent(row) for row in persistent]
    event_buckets = _bucket_events(events, timestamps)
    _rewrite_episode(table, path, episode_index, persistent_rows, event_buckets)
    _ensure_features(root)


def _rewrite_episode(
    table: pa.Table,
    path: Path,
    episode_index: int,
    persistent_rows: list[dict[str, Any]],
    event_buckets: dict[float, list[dict[str, Any]]],
) -> None:
    episode_ids = [int(value) for value in table.column("episode_index").to_pylist()]
    frame_ts = [float(value) for value in table.column("timestamp").to_pylist()]
    if LANGUAGE_PERSISTENT in table.column_names:
        old_persistent = table.column(LANGUAGE_PERSISTENT).to_pylist()
        old_events = table.column(LANGUAGE_EVENTS).to_pylist()
    else:
        old_persistent = [[] for _ in episode_ids]
        old_events = [[] for _ in episode_ids]
    new_persistent = []
    new_events = []
    for index, episode_id in enumerate(episode_ids):
        if episode_id == episode_index:
            new_persistent.append(persistent_rows)
            new_events.append(event_buckets.get(frame_ts[index], []))
        else:
            new_persistent.append(old_persistent[index] or [])
            new_events.append(old_events[index] or [])
    columns = []
    names = []
    for name in table.column_names:
        if name in (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS):
            continue
        columns.append(table.column(name))
        names.append(name)
    columns.extend([pa.array(new_persistent), pa.array(new_events)])
    names.extend([LANGUAGE_PERSISTENT, LANGUAGE_EVENTS])
    rewritten = pa.Table.from_arrays(columns, names=names)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_table_one_row_group_per_episode(rewritten, temporary)
    temporary.replace(path)


def _normalize_persistent(row: dict[str, Any]) -> dict[str, Any]:
    style = row.get("style")
    if style not in PERSISTENT_STYLES:
        raise ValueError(f"persistent style must be one of {PERSISTENT_STYLES}, got {style!r}")
    if column_for_style(style) != LANGUAGE_PERSISTENT:
        raise ValueError(f"{style} does not belong in language_persistent")
    camera = row.get("camera") or None
    validate_camera_field(style, camera)
    content = row.get("content")
    if content is None or not str(content).strip():
        raise ValueError("persistent rows need non-empty content")
    return {
        "role": str(row.get("role") or "assistant"),
        "content": str(content),
        "style": style,
        "timestamp": float(row["timestamp"]),
        "camera": camera,
        "tool_calls": None,
    }


def _bucket_events(
    events: list[dict[str, Any]], timestamps: list[float]
) -> dict[float, list[dict[str, Any]]]:
    buckets: dict[float, list[dict[str, Any]]] = {}
    for row in events:
        style = row.get("style")
        if style == "":
            style = None
        snapped = _snap(float(row["timestamp"]), timestamps)
        if style == "say":
            text = str(row.get("content") or "").strip()
            if not text:
                raise ValueError("a say event needs the spoken text in content")
            atom = speech_atom(snapped, text)
            stored = {key: atom[key] for key in ("role", "content", "style", "camera", "tool_calls")}
            buckets.setdefault(snapped, []).append(stored)
            continue
        if style not in EVENT_STYLES:
            raise ValueError(f"event style must be one of {EDITOR_EVENT_STYLES}, got {style!r}")
        if column_for_style(style) != LANGUAGE_EVENTS:
            raise ValueError(f"{style} does not belong in language_events")
        camera = row.get("camera") or None
        validate_camera_field(style, camera)
        stored = {
            "role": str(row.get("role") or "assistant"),
            "content": None if row.get("content") in (None, "") else str(row.get("content")),
            "style": style,
            "camera": camera,
            "tool_calls": row.get("tool_calls"),
        }
        if stored["content"] is None and not stored["tool_calls"]:
            raise ValueError("an event needs content or tool_calls")
        buckets.setdefault(snapped, []).append(stored)
    return buckets


def _snap(timestamp: float, timestamps: list[float]) -> float:
    nearest = min(timestamps, key=lambda item: abs(item - timestamp))
    if abs(nearest - timestamp) > 0.05:
        raise ValueError(f"event timestamp {timestamp} is not within 0.05s of a frame")
    return nearest


def _annotated_flags(root: Path, episodes: Any) -> dict[int, bool]:
    flags: dict[int, bool] = {}
    seen_paths: set[Path] = set()
    info = load_info(root)
    for index in range(len(episodes)):
        episode = episodes[index]
        path = _data_path(root, info, episode)
        if path in seen_paths:
            continue
        seen_paths.add(path)
        if not path.is_file() or LANGUAGE_PERSISTENT not in pq.read_schema(path).names:
            continue
        table = pq.read_table(path, columns=["episode_index", LANGUAGE_PERSISTENT])
        current: int | None = None
        for episode_id, rows in zip(
            table.column("episode_index").to_pylist(),
            table.column(LANGUAGE_PERSISTENT).to_pylist(),
            strict=True,
        ):
            episode_id = int(episode_id)
            if episode_id == current:
                continue
            current = episode_id
            flags[episode_id] = bool(rows)
    return flags


def _episode(episodes: Any, episode_index: int) -> dict[str, Any]:
    for index in range(len(episodes)):
        episode = episodes[index]
        if int(episode["episode_index"]) == episode_index:
            return episode
    raise ValueError(f"episode {episode_index} was not found")


def _data_path(root: Path, info: Any, episode: dict[str, Any]) -> Path:
    relative = info.data_path.format(
        chunk_index=int(episode["data/chunk_index"]),
        file_index=int(episode["data/file_index"]),
    )
    return root / relative


def _video_relative(info: Any, episode: dict[str, Any], video_key: str) -> str:
    template = info.video_path
    if not template:
        raise ValueError("dataset has no video_path")
    return template.format(
        video_key=video_key,
        chunk_index=int(episode[f"videos/{video_key}/chunk_index"]),
        file_index=int(episode[f"videos/{video_key}/file_index"]),
    )


def _read_columns(path: Path) -> list[str]:
    names = pq.read_schema(path).names
    columns = ["episode_index", "timestamp"]
    if LANGUAGE_PERSISTENT in names:
        columns.extend([LANGUAGE_PERSISTENT, LANGUAGE_EVENTS])
    return columns


def _public_persistent(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": row.get("role"),
        "content": row.get("content"),
        "style": row.get("style"),
        "timestamp": float(row["timestamp"]),
        "camera": row.get("camera"),
    }


def _public_event(row: dict[str, Any], timestamp: float) -> dict[str, Any]:
    style = row.get("style")
    content = row.get("content")
    tool_calls = row.get("tool_calls")
    if style is None and tool_calls:
        style = "say"
        content = _say_text(tool_calls)
    return {
        "role": row.get("role"),
        "content": content,
        "style": style,
        "timestamp": timestamp,
        "camera": row.get("camera"),
        "tool_calls": tool_calls,
    }


def _say_text(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list) or not tool_calls:
        return ""
    function = tool_calls[0].get("function") if isinstance(tool_calls[0], dict) else None
    arguments = function.get("arguments") if isinstance(function, dict) else None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    if isinstance(arguments, dict):
        return str(arguments.get("text") or "")
    return ""


def _copy_ignore(source: Path, directory: str, names: list[str]) -> list[str]:
    if Path(directory).resolve() != source:
        return []
    return [name for name in names if name in {".annotate_staging", "videos"}]


def _safe_parquet(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path.suffix != ".parquet" or root not in path.parents:
        raise ValueError(f"{relative} is not a parquet file inside the dataset")
    if not path.is_file():
        raise ValueError(f"{relative} was not found")
    return path


def _json_value(value: Any) -> Any:
    """Keep every element. Only rewrite values ``json.dumps`` cannot emit."""
    if isinstance(value, float):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _staged_rows(path: Path, module: str, style_counts: Counter[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        style = row.get("style") or "say"
        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            content = _say_text(row.get("tool_calls"))
        style_name = str(style)
        style_counts[style_name] += 1
        column = "language_persistent" if style_name in PERSISTENT_STYLES else "language_events"
        rows.append(
            {
                "module": module,
                "style": style_name,
                "role": row.get("role"),
                "timestamp": row.get("timestamp"),
                "column": column,
                "content": content if isinstance(content, str) else "",
            }
        )
    return rows


def _language_counts(root: Path) -> dict[int, dict[str, int]]:
    """How many language rows are already stored on the first frame of each episode."""
    data = root / "data"
    if not data.is_dir():
        return {}
    counts: dict[int, dict[str, int]] = {}
    for path in sorted(data.rglob("*.parquet")):
        names = pq.read_schema(path).names
        if LANGUAGE_PERSISTENT not in names:
            continue
        table = pq.read_table(path, columns=["episode_index", LANGUAGE_PERSISTENT, LANGUAGE_EVENTS])
        seen: set[int] = set()
        for episode_id, persistent, events in zip(
            table.column("episode_index").to_pylist(),
            table.column(LANGUAGE_PERSISTENT).to_pylist(),
            table.column(LANGUAGE_EVENTS).to_pylist(),
            strict=True,
        ):
            episode_id = int(episode_id)
            if episode_id in seen:
                continue
            seen.add(episode_id)
            counts[episode_id] = {
                "persistent": len(persistent or []),
                "events": len(events or []),
            }
    return counts


def _ensure_features(root: Path) -> None:
    info = load_info(root)
    changed = False
    for key, spec in language_feature_info().items():
        if key not in info.features:
            info.features[key] = {**spec, "shape": list(spec["shape"])}
            changed = True
    if changed:
        write_info(info, root)
