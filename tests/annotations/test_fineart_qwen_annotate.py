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
"""Span normalization and parquet writes for the FineART Qwen annotator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "fineart_annotate"
sys.path.insert(0, str(_EXAMPLE))

from fineart_qwen import (  # noqa: E402
    normalize_spans,
    parse_episode_spec,
    persistent_rows,
    register_language_features,
    rewrite_shard,
)

pytest.importorskip("pyarrow")


def test_normalize_spans_covers_the_episode_without_gaps():
    spans = normalize_spans(
        [
            {"text": "place the cup", "start": 1.2, "end": 1.4},
            {"text": "pick up the cup", "start": 0.2, "end": 1.0},
        ],
        duration_s=3.0,
        min_s=0.5,
    )

    assert spans[0]["start"] == 0.0
    assert spans[-1]["end"] == 3.0
    assert spans[0]["text"] == "pick up the cup"
    for previous, current in zip(spans, spans[1:], strict=False):
        assert previous["end"] == current["start"]


def test_parse_episode_spec_rejects_out_of_range():
    assert parse_episode_spec("0-1,1", 4) == [0, 1]
    with pytest.raises(ValueError, match="out of range"):
        parse_episode_spec("4", 4)


def test_rewrite_shard_broadcasts_subtasks_and_keeps_other_episodes(tmp_path: Path):
    shard = tmp_path / "file.parquet"
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0, 1, 1],
                "timestamp": [0.0, 0.5, 0.0, 0.5],
            }
        ),
        shard,
    )
    rows = persistent_rows([{"text": "pick up the cup", "start": 0.0, "end": 1.0}])
    rewrite_shard(shard, {0: rows})

    table = pq.read_table(shard)
    persistent = table.column("language_persistent").to_pylist()
    events = table.column("language_events").to_pylist()
    assert persistent[0] == persistent[1] == rows
    assert persistent[2] == persistent[3] == []
    assert events == [[], [], [], []]
    assert table.column("timestamp").to_pylist() == [0.0, 0.5, 0.0, 0.5]


def test_register_language_features_updates_info_json(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 10,
                "features": {"action": {"dtype": "float32", "shape": [2], "names": None}},
                "total_episodes": 0,
                "total_frames": 0,
                "total_tasks": 0,
                "chunks_size": 1000,
                "data_files_size_in_mb": 100,
                "video_files_size_in_mb": 200,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": None,
            }
        ),
        encoding="utf-8",
    )

    register_language_features(tmp_path)
    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    assert info["features"]["language_persistent"]["dtype"] == "language"
    assert info["features"]["language_events"]["dtype"] == "language"
