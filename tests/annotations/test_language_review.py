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
"""Partial annotation writes keep episodes that were not part of the run."""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.annotations.steerable_pipeline.reader import EpisodeRecord
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging
from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "lerobot_annotate_web"
sys.path.insert(0, str(_EXAMPLE))
from review import _rewrite_episode  # noqa: E402

pytest.importorskip("pyarrow")


def _shard(path: Path) -> None:
    persistent = [
        [{"role": "assistant", "content": "keep me", "style": "subtask", "timestamp": 0.0, "camera": None, "tool_calls": None}],
        [{"role": "assistant", "content": "keep me", "style": "subtask", "timestamp": 0.0, "camera": None, "tool_calls": None}],
        [],
        [],
    ]
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0, 1, 1],
                "timestamp": [0.0, 0.5, 0.0, 0.5],
                "language_persistent": persistent,
                "language_events": [[], [], [], []],
            }
        ),
        path,
    )


def test_writer_preserves_episodes_outside_the_run(tmp_path: Path):
    shard = tmp_path / "file.parquet"
    _shard(shard)
    EpisodeStaging(tmp_path / "staging", 1).write(
        "plan",
        [
            {
                "role": "assistant",
                "content": "pick up the tissue",
                "style": "subtask",
                "timestamp": 0.0,
                "camera": None,
                "tool_calls": None,
            }
        ],
    )
    record = EpisodeRecord(
        episode_index=1,
        episode_task="take a tissue",
        frame_timestamps=(0.0, 0.5),
        frame_indices=(2, 3),
        data_path=shard,
        row_offset=2,
        row_count=2,
    )
    LanguageColumnsWriter().write_all([record], tmp_path / "staging", tmp_path)

    table = pq.read_table(shard)
    persistent = table.column("language_persistent").to_pylist()
    assert persistent[0][0]["content"] == "keep me"
    assert persistent[2][0]["content"] == "pick up the tissue"
    assert persistent[3][0]["content"] == "pick up the tissue"


def test_manual_edit_replaces_only_the_selected_episode(tmp_path: Path):
    shard = tmp_path / "file.parquet"
    _shard(shard)
    table = pq.read_table(shard)
    _rewrite_episode(
        table,
        shard,
        0,
        [
            {
                "role": "assistant",
                "content": "edited",
                "style": "subtask",
                "timestamp": 0.0,
                "camera": None,
                "tool_calls": None,
            }
        ],
        {},
    )
    persistent = pq.read_table(shard).column("language_persistent").to_pylist()
    assert persistent[0][0]["content"] == "edited"
    assert persistent[2] == []
