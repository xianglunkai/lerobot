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
"""Dataset overview, parquet preview, copy export, and manual say edits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "lerobot_annotate_web"
sys.path.insert(0, str(_EXAMPLE))
from review import (  # noqa: E402
    _bucket_events,
    _public_event,
    dataset_overview,
    load_episode,
    materialize_dataset,
    parquet_page,
    parse_annotate_log,
    save_episode,
    staging_progress,
)


def _dataset(root: Path) -> Path:
    root.mkdir()
    info = {
        "codebase_version": "v3.0",
        "fps": 10,
        "robot_type": "so100",
        "total_episodes": 2,
        "total_frames": 4,
        "total_tasks": 1,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [10], "names": None},
            "observation.images.cam": {"dtype": "video", "shape": [8, 8, 3], "names": ["h", "w", "c"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    meta = root / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    episodes = meta / "episodes" / "chunk-000"
    episodes.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "length": [2, 2],
                "tasks": [["take a tissue"], ["take a tissue"]],
                "data/chunk_index": [0, 0],
                "data/file_index": [0, 0],
                "videos/observation.images.cam/from_timestamp": [0.0, 1.0],
                "videos/observation.images.cam/to_timestamp": [0.2, 1.2],
                "videos/observation.images.cam/chunk_index": [0, 0],
                "videos/observation.images.cam/file_index": [0, 0],
            }
        ),
        episodes / "file-000.parquet",
    )
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    action = [[float(i)] * 10 for i in range(4)]
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0, 1, 1],
                "timestamp": [0.0, 0.1, 0.0, 0.1],
                "frame_index": [0, 1, 2, 3],
                "action": action,
            }
        ),
        data / "file-000.parquet",
    )
    video = root / "videos" / "observation.images.cam" / "chunk-000"
    video.mkdir(parents=True)
    (video / "file-000.mp4").write_bytes(b"not-a-real-video")
    staging = root / ".annotate_staging" / "episode_000000"
    staging.mkdir(parents=True)
    (staging / "plan.jsonl").write_text(
        json.dumps(
            {
                "role": "assistant",
                "content": "open the drawer",
                "style": "subtask",
                "timestamp": 0.0,
                "tool_calls": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_overview_lists_features_files_and_skips_staging(tmp_path: Path):
    root = _dataset(tmp_path / "ds")
    overview = dataset_overview(root)
    assert overview["total_episodes"] == 2
    assert overview["features"]["action"]["shape"] == [10]
    assert overview["video_keys"] == ["observation.images.cam"]
    paths = [item["path"] for item in overview["files"]]
    assert "data/chunk-000/file-000.parquet" in paths
    assert "videos/observation.images.cam/chunk-000/file-000.mp4" in paths
    assert all(".annotate_staging" not in path for path in paths)
    progress = staging_progress(root)
    assert progress["modules"]["plan"]["done"] == 1
    assert progress["latest"]["rows"][0]["content"] == "open the drawer"
    episode = progress["episodes"][0]
    assert episode["rows"][0]["style"] == "subtask"
    assert episode["rows"][0]["column"] == "language_persistent"
    assert progress["parquet_written"] is False
    assert progress["style_counts"]["subtask"] == 1
    assert progress["styles"]["subtask"] == [0]


def test_parquet_page_keeps_every_vector_value(tmp_path: Path):
    root = _dataset(tmp_path / "ds")
    page = parquet_page(root, "data/chunk-000/file-000.parquet", episode_index=1, limit=10)
    assert page["rows_total"] == 2
    action = page["rows"][0]["raw"]["action"]
    assert action == [2.0] * 10
    assert page["rows"][0]["cells"]["action"] == json.dumps(action)


def test_save_episode_round_trips_subtask_and_say(tmp_path: Path):
    root = _dataset(tmp_path / "ds")
    save_episode(
        root,
        0,
        [{"style": "subtask", "role": "assistant", "timestamp": 0.0, "content": "open the drawer"}],
        [{"style": "say", "role": "assistant", "timestamp": 0.0, "content": "here you go", "camera": ""}],
    )
    loaded = load_episode(root, 0)
    assert loaded["persistent"][0]["content"] == "open the drawer"
    assert loaded["events"][0]["style"] == "say"
    assert loaded["events"][0]["content"] == "here you go"
    other = load_episode(root, 1)
    assert other["persistent"] == []
    assert "language_persistent" in json.loads((root / "meta" / "info.json").read_text())["features"]


def test_say_editor_style_becomes_a_speech_atom():
    buckets = _bucket_events(
        [{"style": "say", "timestamp": 0.02, "content": "here you go", "camera": ""}],
        [0.0, 0.1],
    )
    stored = buckets[0.0][0]
    assert stored["style"] is None
    assert stored["content"] is None
    public = _public_event(stored, 0.0)
    assert public["style"] == "say"
    assert public["content"] == "here you go"


def test_copy_links_videos_and_leaves_the_source(tmp_path: Path):
    source = _dataset(tmp_path / "src")
    destination = materialize_dataset(source, "copy", str(tmp_path / "out"))
    copied = destination / "data" / "chunk-000" / "file-000.parquet"
    video = destination / "videos" / "observation.images.cam" / "chunk-000" / "file-000.mp4"
    assert copied.is_file() and not copied.is_symlink()
    assert video.is_symlink()
    assert not (destination / ".annotate_staging").exists()
    assert (source / "data" / "chunk-000" / "file-000.parquet").read_bytes() == copied.read_bytes()


def test_parse_annotate_log_reads_the_latest_episode():
    progress = parse_annotate_log(
        "[annotate] phase=plan starting on 28 episode(s)\n"
        "[annotate]   plan episode 2/28 (idx=1) done in 57.9s\n"
        "[annotate] wrote 1 shard(s)\n"
    )
    assert progress["phase"] == "plan"
    assert progress["last"]["episode"] == 1
    assert progress["finished"] is True


def test_rejected_image_batch_is_skipped():
    from run_pipeline import _SkipRejectedImages

    class _Inner:
        def generate_json(self, batch, **kwargs):
            raise RuntimeError("Error code: 400 data_inspection_failed")

    assert _SkipRejectedImages(_Inner()).generate_json([{"role": "user"}]) == [None]


def test_existing_staging_skips_the_module(tmp_path: Path):
    from run_pipeline import _SkipFinished

    from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging

    class _Inner:
        def __init__(self) -> None:
            self.calls = 0

        def run_episode(self, record, staging) -> None:
            del record, staging
            self.calls += 1

    staging = EpisodeStaging(tmp_path, 0)
    staging.write(
        "plan",
        [{"role": "assistant", "content": "open the drawer", "style": "subtask", "timestamp": 0.0}],
    )
    inner = _Inner()
    record = type("Record", (), {"episode_index": 0})()
    _SkipFinished(inner, "plan").run_episode(record, staging)
    assert inner.calls == 0


def test_video_stream_stops_when_the_browser_disconnects():
    import io

    from app import stream_file

    class _Writer:
        def __init__(self) -> None:
            self.chunks = 0

        def __call__(self, data: bytes) -> None:
            del data
            self.chunks += 1
            raise ConnectionResetError(104, "Connection reset by peer")

    writer = _Writer()
    stream_file(io.BytesIO(b"abcdef"), writer, 0, 6)
    assert writer.chunks == 1


def test_video_stream_sends_the_requested_range():
    import io

    from app import stream_file

    sent = bytearray()
    stream_file(io.BytesIO(b"abcdef"), sent.extend, 2, 3)
    assert bytes(sent) == b"cde"


def test_memory_and_subtask_selection_skips_the_other_modules():
    from run_pipeline import feature_plan

    from lerobot.annotations.steerable_pipeline.config import InterjectionsConfig, PlanConfig, VqaConfig

    selection = feature_plan("subtask,memory")
    plan = PlanConfig(
        enabled=selection.plan_enabled,
        n_task_rephrasings=selection.n_task_rephrasings,
        emit_plan=selection.emit_plan,
        emit_memory=selection.emit_memory,
    )
    assert plan.enabled is True
    assert plan.emit_plan is False
    assert plan.emit_memory is True
    assert plan.n_task_rephrasings == 0
    assert InterjectionsConfig(enabled=selection.interjections_enabled).enabled is False
    assert VqaConfig(enabled=selection.vqa_enabled).enabled is False
    assert selection.plan_enabled is True
    assert selection.emit_plan is False
    assert selection.emit_memory is True
    assert selection.n_task_rephrasings == 0
    assert selection.interjections_enabled is False
    assert selection.vqa_enabled is False
    assert selection.keep_styles == frozenset({"subtask", "memory"})
    assert selection.narrow is True
    assert selection.spec() == "subtask,memory"


def test_interjections_still_segment_subtasks_but_only_write_the_pair():
    from run_pipeline import feature_plan

    selection = feature_plan("interjections")
    assert selection.plan_enabled is True
    assert selection.emit_plan is False
    assert selection.emit_memory is False
    assert selection.interjections_enabled is True
    assert selection.keep_styles == frozenset({"interjection", "say"})


def test_vqa_only_disables_plan_and_interjections():
    from run_pipeline import feature_plan

    selection = feature_plan("vqa")
    assert selection.plan_enabled is False
    assert selection.interjections_enabled is False
    assert selection.vqa_enabled is True
    assert selection.keep_styles == frozenset({"vqa"})


def test_all_features_keep_the_default_modules():
    from run_pipeline import feature_plan

    selection = feature_plan("all")
    assert selection.narrow is False
    assert selection.emit_plan is True
    assert selection.emit_memory is True
    assert selection.n_task_rephrasings == 4
    assert selection.interjections_enabled is True
    assert selection.vqa_enabled is True


def test_unknown_feature_is_rejected():
    from run_pipeline import feature_plan

    with pytest.raises(ValueError, match="unknown features"):
        feature_plan("motion")


def test_narrow_selection_filters_staging_and_leaves_the_original(tmp_path: Path):
    from run_pipeline import feature_plan, filtered_staging_dir

    from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging

    selection = feature_plan("subtask,memory")
    staging = EpisodeStaging(tmp_path, 3)
    staging.write(
        "plan",
        [
            {"role": "assistant", "content": "open", "style": "subtask", "timestamp": 0.0},
            {"role": "assistant", "content": "steps", "style": "plan", "timestamp": 0.0},
            {"role": "assistant", "content": "drawer is open", "style": "memory", "timestamp": 1.0},
            {"role": "assistant", "content": "rephrase", "style": "task_aug", "timestamp": 0.0},
        ],
    )
    staging.write(
        "interjections",
        [
            {"role": "user", "content": "hey", "style": "interjection", "timestamp": 1.0},
            {"role": "assistant", "content": None, "style": None, "timestamp": 1.0},
        ],
    )
    filtered = filtered_staging_dir(tmp_path, [3], selection.keep_styles)
    kept = EpisodeStaging(filtered, 3)
    assert [row["style"] for row in kept.read("plan")] == ["subtask", "memory"]
    assert kept.read("interjections") == []
    assert len(EpisodeStaging(tmp_path, 3).read("plan")) == 4
    assert len(EpisodeStaging(tmp_path, 3).read("interjections")) == 2
