"""Utilities to convert a LeRobot dataset from codebase version v3.0 back to v2.0.
The script mirrors :mod:`lerobot.datasets.v21.convert_dataset_v21_to_v30` but applies the reverse
transformations so an existing dataset created with the new consolidated file
layout can be ported back to the legacy per-episode structure.
Usage examples
--------------
Convert a dataset that already exists locally::
    python src/lerobot/datasets/v30/convert_dataset_v30_to_v21.py \
        --repo-id=lerobot/pusht \
        --root=/path/to/datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2  # type: ignore
import jsonlines
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import tqdm
from huggingface_hub import snapshot_download

try:
    import decord  # type: ignore
    HAVE_DECORD = True
except Exception:
    HAVE_DECORD = False

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    EPISODES_DIR,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    load_info,
    load_tasks,
    serialize_dict,
    unflatten_dict,
    write_info,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging

V20 = "v2.0"
V30 = "v3.0"

# Mapping from v3.0 video keys to (v2.0 column name, short video dir)
V30_TO_V20_VIDEO_MAPPING = {
    "observation.images.high": ("observation.images.cam_high", "cam_high"),
    "observation.images.left": ("observation.images.cam_left_wrist", "cam_left_wrist"),
    "observation.images.right": ("observation.images.cam_right_wrist", "cam_right_wrist"),
}

LEGACY_DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
MIN_VIDEO_DURATION = 1e-6

def _to_serializable(value: Any) -> Any:
    """Convert numpy/pyarrow values into standard Python types for JSON dumps."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    return value


def _normalize_task_value(task_value: Any) -> str | None:
    serializable = _to_serializable(task_value)
    if serializable is None:
        return None
    if isinstance(serializable, str):
        return serializable
    return json.dumps(serializable, ensure_ascii=False)


def validate_local_dataset_version(local_path: Path) -> None:
    info = load_info(local_path)
    dataset_version = info.get("codebase_version", "unknown")
    if dataset_version != V30:
        raise ValueError(
            f"Local dataset has codebase version '{dataset_version}', expected '{V30}'. "
            f"This script converts datasets from v3.0 back to v2.0."
        )


def load_episode_records(root: Path) -> list[dict[str, Any]]:
    """Load the consolidated metadata rows stored in ``meta/episodes``."""

    episodes_dir = root / EPISODES_DIR
    pq_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not pq_paths:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}.")

    records: list[dict[str, Any]] = []
    for pq_path in pq_paths:
        table = pq.read_table(pq_path)
        records.extend(table.to_pylist())

    records.sort(key=lambda rec: int(rec["episode_index"]))
    return records


def convert_tasks(root: Path, new_root: Path) -> None:
    logging.info("Converting tasks parquet to legacy JSONL")
    tasks = load_tasks(root)
    tasks = tasks.sort_values("task_index")

    out_path = new_root / LEGACY_TASKS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(out_path, mode="w") as writer:
        for task, row in tasks.iterrows():
            task_value = _normalize_task_value(task)
            if task_value is None:
                continue
            writer.write({
                "task_index": int(row["task_index"]),
                "task": task_value,
            })


def convert_info(
    root: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],  # v2 column names
) -> None:
    info = load_info(root)
    logging.info("Converting info.json metadata to v2.0 schema")
    total_episodes = info.get("total_episodes") or len(episode_records)
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    info["codebase_version"] = V20
    info["data_path"] = LEGACY_DATA_PATH_TEMPLATE
    if info.get("video_path") is not None and len(video_keys) > 0:
        info["video_path"] = LEGACY_VIDEO_PATH_TEMPLATE
    else:
        info["video_path"] = None
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)

    # 重命名 features 键：v3 -> v2
    new_features: dict[str, Any] = {}
    for key, ft in info["features"].items():
        if key in V30_TO_V20_VIDEO_MAPPING:
            v2_key = V30_TO_V20_VIDEO_MAPPING[key][0]
        else:
            v2_key = key
        if ft.get("dtype") != "video":
            ft.pop("fps", None)
        if isinstance(ft.get("dtype"), str) and ft["dtype"].lower() == "list":
            item_dtype = ft.pop("item_dtype", None) or ft.pop("value_dtype", None) or "float32"
            ft["dtype"] = "video" if v2_key in video_keys else item_dtype
        new_features[v2_key] = ft
    info["features"] = new_features

    info["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0
    info["total_videos"] = total_episodes * len(video_keys)
    write_info(info, new_root)


def _group_episodes_by_data_file(
    episode_records: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in episode_records:
        key = (
            int(record["data/chunk_index"]),
            int(record["data/file_index"]),
        )
        grouped[key].append(record)
    return grouped


def _decode_video_frames(video_path: Path, num_rows: int) -> tuple[list[bytes | None], list[str]]:
    """Decode video into PNG bytes (RGB) for embedding into struct<bytes,path>.
    Fallback: return all None if解码失败。"""
    if not video_path.exists():
        return [None] * num_rows, [f"frame_{i:06d}.png" for i in range(num_rows)]
    frames: list[bytes | None] = []
    # 优先 decord
    if HAVE_DECORD:
        try:
            vr = decord.VideoReader(str(video_path))
            length = len(vr)
            for i in range(min(length, num_rows)):
                img = vr[i].asnumpy()  # RGB
                bgr = img[..., ::-1]
                ok, buf = cv2.imencode(".png", bgr)
                frames.append(buf.tobytes() if ok else None)
        except Exception:
            frames = []
    # 其次 OpenCV
    if not frames:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            idx = 0
            while idx < num_rows:
                ret, frame = cap.read()
                if not ret:
                    break
                ok, buf = cv2.imencode(".png", frame)
                frames.append(buf.tobytes() if ok else None)
                idx += 1
            cap.release()
    if not frames:
        return [None] * num_rows, [f"frame_{i:06d}.png" for i in range(num_rows)]
    if len(frames) < num_rows:
        last = frames[-1]
        frames.extend([last] * (num_rows - len(frames)))
    elif len(frames) > num_rows:
        frames = frames[:num_rows]
    paths = [f"frame_{i:06d}.png" for i in range(num_rows)]
    return frames, paths


def _extract_camera_name(video_feature_name: str) -> str:
    # Extract short name from v2.0 style names like observation.images.cam_high -> cam_high
    if video_feature_name.startswith("observation.images.cam_"):
        return video_feature_name.split(".", 2)[-1]  # e.g., cam_high
    return video_feature_name


def convert_data(
    root: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],  # v2 column names
    task_lookup: dict[int, str],
    embed_images: bool,
) -> None:
    logging.info("Converting consolidated parquet files back to per-episode files")
    grouped = _group_episodes_by_data_file(episode_records)

    for (chunk_idx, file_idx), records in tqdm.tqdm(grouped.items(), desc="convert data files"):
        source_path = root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        if not source_path.exists():
            raise FileNotFoundError(f"Expected source parquet file not found: {source_path}")

        table = pq.read_table(source_path)
        records = sorted(records, key=lambda rec: int(rec["dataset_from_index"]))
        file_offset = int(records[0]["dataset_from_index"])

        for record in records:
            episode_index = int(record["episode_index"])
            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            length = stop - start
            if length <= 0:
                raise ValueError(f"Invalid episode length: episode_index={episode_index}, length={length}")

            episode_table = table.slice(start, length)

            # 如果缺失 velocity / effort 列，补零
            state_col_name = "observation.state"
            if state_col_name in episode_table.column_names:
                try:
                    first_state = episode_table.column(state_col_name)[0].as_py()
                    dim = len(first_state)
                    list_type = pa.list_(pa.float32(), list_size=dim)
                except Exception:
                    list_type = pa.list_(pa.float32())
                    dim = None
                if "observation.velocity" not in episode_table.column_names:
                    zeros = [[0.0] * dim if dim is not None else [] for _ in range(episode_table.num_rows)]
                    episode_table = episode_table.append_column("observation.velocity", pa.array(zeros, type=list_type))
                if "observation.effort" not in episode_table.column_names:
                    zeros = [[0.0] * dim if dim is not None else [] for _ in range(episode_table.num_rows)]
                    episode_table = episode_table.append_column("observation.effort", pa.array(zeros, type=list_type))

            dest_chunk = episode_index // DEFAULT_CHUNK_SIZE
            dest_path = new_root / LEGACY_DATA_PATH_TEMPLATE.format(
                episode_chunk=dest_chunk,
                episode_index=episode_index,
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # 添加图像列：使用 v2.0 列名
            if video_keys:
                for v2_col_name in video_keys:
                    if v2_col_name in episode_table.column_names:
                        continue
                    cam_dir = _extract_camera_name(v2_col_name)  # cam_high 等
                    video_rel_path = LEGACY_VIDEO_PATH_TEMPLATE.format(
                        episode_chunk=dest_chunk,
                        video_key=cam_dir,
                        episode_index=episode_index,
                    )
                    num_rows = episode_table.num_rows
                    full_video_path = new_root / video_rel_path
                    if embed_images:
                        bytes_list, paths_list = _decode_video_frames(full_video_path, num_rows)
                        bytes_array = pa.array(bytes_list, type=pa.binary())
                        path_array = pa.array(paths_list, type=pa.string())
                    else:
                        bytes_array = pa.array([None] * num_rows, type=pa.binary())
                        path_array = pa.array([video_rel_path] * num_rows, type=pa.string())
                    struct_array = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"])
                    episode_table = episode_table.append_column(v2_col_name, struct_array)

            # 任务列
            if "task" not in episode_table.column_names:
                task_value = record.get("task")
                if task_value is None:
                    task_idx = record.get("task_index")
                    try:
                        task_value = task_lookup[int(task_idx)] if task_idx is not None else None
                    except (KeyError, TypeError, ValueError):
                        task_value = None
                task_value = _normalize_task_value(task_value)
                if task_value is not None:
                    task_array = pa.array([task_value] * episode_table.num_rows, type=pa.string())
                    episode_table = episode_table.append_column("task", task_array)

            expected_prefix_order = [
                "observation.state",
                "action",
                "observation.velocity",
                "observation.effort",
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task",
                "task_index",
            ]
            existing_cols = episode_table.column_names
            ordered = [c for c in expected_prefix_order if c in existing_cols]
            remaining = [c for c in existing_cols if c not in ordered]
            final_order = ordered + remaining
            if final_order != existing_cols:
                try:
                    episode_table = episode_table.select(final_order)
                except Exception as exc:
                    logging.warning("Reorder failed episode=%s path=%s error=%s", episode_index, dest_path, exc)

            episode_table = _sanitize_table_metadata(episode_table)
            pq.write_table(episode_table, dest_path)


def _group_episodes_by_video_file(
    episode_records: Iterable[dict[str, Any]],
    video_key: str,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    chunk_column = f"videos/{video_key}/chunk_index"
    file_column = f"videos/{video_key}/file_index"

    for record in episode_records:
        if chunk_column not in record or file_column not in record:
            continue
        chunk_idx = record.get(chunk_column)
        file_idx = record.get(file_column)
        if chunk_idx is None or file_idx is None:
            continue
        grouped[(int(chunk_idx), int(file_idx))].append(record)
    return grouped


def _validate_video_paths(src: Path, dst: Path) -> None:
    """Validate source and destination paths to prevent security issues."""

    # Convert to Path objects if they aren't already
    src = Path(src)
    dst = Path(dst)

    # Resolve paths to handle symlinks and normalize them
    try:
        src_resolved = src.resolve()
        dst_resolved = dst.resolve()
    except OSError as exc:
        raise ValueError(f"Invalid path provided: {exc}") from exc

    # Check that source file exists and is a regular file
    if not src_resolved.exists():
        raise FileNotFoundError(f"Source video file does not exist: {src_resolved}")

    if not src_resolved.is_file():
        raise ValueError(f"Source path is not a regular file: {src_resolved}")

    # Validate file extensions for video files
    valid_video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    if src_resolved.suffix.lower() not in valid_video_extensions:
        raise ValueError(f"Source file does not have a valid video extension: {src_resolved}")

    if dst_resolved.suffix.lower() not in valid_video_extensions:
        raise ValueError(f"Destination file does not have a valid video extension: {dst_resolved}")

    # Check for path traversal attempts in the original paths
    src_str = str(src)
    dst_str = str(dst)

    # Ensure paths don't contain null bytes or other control characters
    for path_str, name in [(src_str, "source"), (dst_str, "destination")]:
        if "\0" in path_str:
            raise ValueError(f"Path contains null bytes: {name} path")
        if any(ord(c) < 32 and c not in ["\t", "\n", "\r"] for c in path_str):
            raise ValueError(f"Path contains invalid control characters: {name} path")

    # Additional check: ensure resolved paths don't point to system directories
    system_dirs = {"/etc", "/sys", "/proc", "/dev", "/boot", "/root"}
    for resolved_path, name in [(src_resolved, "source"), (dst_resolved, "destination")]:
        path_str = str(resolved_path)
        for sys_dir in system_dirs:
            if path_str.startswith(sys_dir + "/") or path_str == sys_dir:
                raise ValueError(f"Path points to system directory: {name} path {resolved_path}")

    # Ensure the destination directory can be created safely
    try:
        dst_parent = dst_resolved.parent
        if not dst_parent.exists():
            # Check if we can create the parent directory structure
            dst_parent.resolve()
    except OSError as exc:
        raise ValueError(f"Cannot create destination directory: {exc}") from exc


def _extract_video_segment(
    src: Path,
    dst: Path,
    start: float,
    end: float,
) -> None:
    # Validate paths to prevent security issues
    _validate_video_paths(src, dst)

    # Validate numeric parameters to prevent injection
    if not (0 <= start <= 86400):  # 24 hours max
        raise ValueError(f"Invalid start time: {start}")
    if not (0 <= end <= 86400):  # 24 hours max
        raise ValueError(f"Invalid end time: {end}")
    if start >= end:
        raise ValueError(f"Start time {start} must be less than end time {end}")

    duration = max(end - start, MIN_VIDEO_DURATION)

    # Validate duration is reasonable
    if duration > 3600:  # 1 hour max
        raise ValueError(f"Video segment duration too long: {duration} seconds")

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Build command with validated parameters
    # Change from copy to re-encode to H.264 to handle AV1 and other codecs reliably
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-an",  # Remove audio if present
        "-avoid_negative_ts",
        "1",
        "-y",
        str(dst),
    ]

    try:
        # Use more secure subprocess call with explicit timeout
        result = subprocess.run(
            cmd, 
            check=True, 
            timeout=600,  # Increased timeout for re-encoding
            capture_output=True, 
            text=True
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out while processing video '{src}' -> '{dst}'") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable not found; it is required for video conversion") from exc
    except subprocess.CalledProcessError as exc:
        error_msg = f"ffmpeg failed while splitting video '{src}' into '{dst}'"
        if exc.stderr:
            error_msg += f". Error: {exc.stderr.strip()}"
        raise RuntimeError(error_msg) from exc


def convert_videos(root: Path, new_root: Path, episode_records: list[dict[str, Any]], video_keys_v3: list[str]) -> None:
    if len(video_keys_v3) == 0:
        logging.info("No video features detected; skipping video conversion")
        return
    logging.info("Converting concatenated MP4 files back to per-episode videos")
    for v3_key in video_keys_v3:
        if v3_key in V30_TO_V20_VIDEO_MAPPING:
            v2_col, dest_dir = V30_TO_V20_VIDEO_MAPPING[v3_key]
        else:
            dest_dir = _extract_camera_name(v3_key)
        # 分组使用 v3 元数据键
        grouped = _group_episodes_by_video_file(episode_records, v3_key)
        if len(grouped) == 0:
            logging.info("No video metadata found for key '%s'; skipping", v3_key)
            continue
        for (chunk_idx, file_idx), records in tqdm.tqdm(grouped.items(), desc=f"convert videos ({v3_key})"):
            src_path = root / DEFAULT_VIDEO_PATH.format(video_key=v3_key, chunk_index=chunk_idx, file_index=file_idx)
            if not src_path.exists():
                raise FileNotFoundError(f"Expected MP4 file not found: {src_path}")
            records = sorted(records, key=lambda rec: float(rec[f"videos/{v3_key}/from_timestamp"]))
            for record in records:
                episode_index = int(record["episode_index"])
                start = float(record[f"videos/{v3_key}/from_timestamp"])
                end = float(record[f"videos/{v3_key}/to_timestamp"])
                dest_chunk = episode_index // DEFAULT_CHUNK_SIZE
                dest_path = new_root / LEGACY_VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=dest_chunk,
                    video_key=dest_dir,  # 目标目录用 v2 短名
                    episode_index=episode_index,
                )
                _extract_video_segment(src_path, dest_path, start=start, end=end)


def convert_episodes_metadata(new_root: Path, episode_records: list[dict[str, Any]]) -> None:
    logging.info("Reconstructing legacy episodes and episodes_stats JSONL files")

    episodes_path = new_root / LEGACY_EPISODES_PATH
    stats_path = new_root / LEGACY_EPISODES_STATS_PATH
    episodes_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(episodes_path, mode="w") as episodes_writer, jsonlines.open(
        stats_path, mode="w"
    ) as stats_writer:
        for record in sorted(episode_records, key=lambda rec: int(rec["episode_index"])):
            legacy_episode = {
                key: value
                for key, value in record.items()
                if not key.startswith("data/")
                and not key.startswith("videos/")
                and not key.startswith("stats/")
                and not key.startswith("meta/")
                and key not in {"dataset_from_index", "dataset_to_index"}
            }

            serializable_episode = {key: _to_serializable(value) for key, value in legacy_episode.items()}
            task_value = serializable_episode.get("task")
            normalized_task = _normalize_task_value(task_value) if task_value is not None else None
            if normalized_task is None:
                serializable_episode.pop("task", None)
            else:
                serializable_episode["task"] = normalized_task
            episodes_writer.write(serializable_episode)

            stats_flat = {key: record[key] for key in record if key.startswith("stats/")}
            stats_nested = unflatten_dict(stats_flat).get("stats", {})
            stats_serialized = serialize_dict(stats_nested)
            stats_writer.write(
                {
                    "episode_index": int(record["episode_index"]),
                    "stats": stats_serialized,
                }
            )


def copy_global_stats(root: Path, new_root: Path) -> None:
    source_stats = root / "meta" / "stats.json"
    if source_stats.exists():
        target_stats = new_root / "meta" / "stats.json"
        target_stats.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_stats, target_stats)


def copy_ancillary_directories(root: Path, new_root: Path) -> None:
    for subdir in ["images"]:
        source = root / subdir
        if source.exists():
            shutil.copytree(source, new_root / subdir, dirs_exist_ok=True)


def _build_task_lookup(root: Path) -> dict[int, str]:
    lookup: dict[int, str] = {}
    try:
        tasks_df = load_tasks(root)
    except FileNotFoundError:
        return lookup
    if tasks_df is None or len(tasks_df) == 0:
        return lookup
    for task_value, row in tasks_df.iterrows():
        task_idx = row.get("task_index")
        try:
            task_idx = int(task_idx)
        except (TypeError, ValueError):
            continue
        task_name = _normalize_task_value(task_value)
        if task_name is None:
            continue
        lookup[task_idx] = task_name
    return lookup


def _sanitize_table_metadata(table: pa.Table) -> pa.Table:
    metadata = table.schema.metadata
    if not metadata or b"huggingface" not in metadata:
        return table
    try:
        hf_metadata = json.loads(metadata[b"huggingface"].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return table
    changed = False
    def _replace(node: Any) -> None:
        nonlocal changed
        if isinstance(node, dict):
            node_type = node.get("_type")
            if node_type == "List":
                node["_type"] = "Sequence"
                changed = True
            dtype = node.get("dtype")
            if isinstance(dtype, str) and dtype.lower() == "list":
                node["dtype"] = "Sequence"
                changed = True
            for value in node.values():
                if isinstance(value, (dict, list)):
                    _replace(value)
        elif isinstance(node, list):
            for item in node:
                _replace(item)
    if "features" in hf_metadata:
        _replace(hf_metadata["features"])
    info_features = hf_metadata.get("info", {}).get("features")
    if info_features:
        _replace(info_features)
    if not changed:
        return table
    new_metadata = dict(metadata)
    new_metadata[b"huggingface"] = json.dumps(hf_metadata, separators=(",", ":")).encode("utf-8")
    return table.replace_schema_metadata(new_metadata)


def convert_dataset(
    repo_id: str,
    root: str | Path | None = None,
    force_conversion: bool = False,
    offline: bool = False,
    embed_images: bool = True,
) -> None:
    """Convert a v3.0 dataset (consolidated layout) back to v2.0 legacy layout.

    Parameters
    ----------
    repo_id: str
        Hugging Face dataset repo id (e.g. ``lerobot/pusht``). Used only for logging when offline.
    root: path | None
        If None, defaults to ``$HF_LEROBOT_HOME/repo_id``. If provided it can either point
        directly to the dataset directory (containing ``info.json``) OR to a parent directory
        under which a subfolder named after ``repo_id`` exists / will be created. We detect this
        automatically to avoid doubling the repo name.
    force_conversion: bool
        If true and dataset directory exists, remove it and (re)download unless ``offline``.
    offline: bool
        If true, never attempt any network calls. The dataset must already exist locally.
    """

    # Resolve base root directory
    if root is None:
        candidate = HF_LEROBOT_HOME / repo_id
    else:
        root = Path(root)
        # If the provided root already contains info.json, treat it as dataset root.
        if (root / "info.json").is_file():
            candidate = root
        # If a subdirectory with the repo name exists and has info.json treat that as root.
        elif (root / repo_id / "info.json").is_file():
            candidate = root / repo_id
        else:
            # If neither exists we assume we should place dataset under root/repo_id
            candidate = root / repo_id
    root = candidate

    if root.exists() and force_conversion:
        logging.info("--force-conversion enabled: removing existing snapshot at %s", root)
        shutil.rmtree(root)

    if root.exists():
        validate_local_dataset_version(root)
        logging.info("Using existing local dataset at %s", root)
    else:
        if offline:
            raise FileNotFoundError(
                f"Dataset directory {root} does not exist in offline mode. Please place an existing "
                "v3.0 dataset there before running the converter with --offline."
            )
        logging.info("Downloading dataset snapshot from the Hub (offline=False)")
        snapshot_download(repo_id, repo_type="dataset", local_dir=root)

    task_lookup = _build_task_lookup(root)
    episode_records = load_episode_records(root)
    for record in episode_records:
        task_idx = record.get("task_index")
        try:
            record["task"] = task_lookup[int(task_idx)]
        except (KeyError, TypeError, ValueError):
            continue
    video_keys_v3 = [
        key
        for key, ft in load_info(root)["features"].items()
        if ft.get("dtype") == "video"
    ]
    video_keys_v2 = [
        V30_TO_V20_VIDEO_MAPPING[k][0] if k in V30_TO_V20_VIDEO_MAPPING else k
        for k in video_keys_v3
    ]
    backup_root = root.parent / f"{root.name}_{V30}"
    new_root = root.parent / f"{root.name}_{V20}"

    if backup_root.is_dir():
        shutil.rmtree(backup_root)
    if new_root.is_dir():
        shutil.rmtree(new_root)

    new_root.mkdir(parents=True, exist_ok=True)

    # 顺序调整：先拆分视频，再写数据文件（需要已存在的 per-episode mp4）
    convert_info(root, new_root, episode_records, video_keys_v2)
    copy_global_stats(root, new_root)
    convert_tasks(root, new_root)
    convert_videos(root, new_root, episode_records, video_keys_v3)
    convert_data(root, new_root, episode_records, video_keys_v2, task_lookup, embed_images)
    convert_episodes_metadata(new_root, episode_records)
    copy_ancillary_directories(root, new_root)

    shutil.move(str(root), str(backup_root))
    shutil.move(str(new_root), str(root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local directory under which the dataset should be stored.",
    )
    parser.add_argument(
        "--force-conversion",
        action="store_true",
        help="Ignore any existing local snapshot and re-download it from the Hub.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run strictly offline: do not attempt to download; dataset must already exist locally.",
    )
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="在每个 observation.images.* 列中嵌入 PNG bytes（可能较慢）。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    init_logging()
    args = parse_args()
    convert_dataset(**vars(args))