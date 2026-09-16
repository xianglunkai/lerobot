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

"""Scheme A: compose high-level ``task`` with optional ``Subtask:`` (string prompt)."""

from __future__ import annotations

import random
from typing import Any

SUBTASK_TAG_KEY = "Subtask"


def has_subtask_marker(task: str | None) -> bool:
    return bool(task) and f"{SUBTASK_TAG_KEY}:" in task


def compose_task_with_subtask(task: str | None, subtask: str | None) -> str:
    """Insert ``Subtask: …`` before an optional trailing ``Advantage:`` line."""
    base = task or ""
    if not subtask or not str(subtask).strip() or has_subtask_marker(base):
        return base
    line = f"{SUBTASK_TAG_KEY}: {str(subtask).strip()}"
    if not base:
        return line
    lines = base.split("\n")
    adv_idx = next((i for i, s in enumerate(lines) if s.startswith("Advantage:")), None)
    if adv_idx is None:
        return f"{base}\n{line}"
    lines.insert(adv_idx, line)
    return "\n".join(lines)


def apply_subtask_conditioning(
    batch: dict[str, Any],
    *,
    dropout_prob: float,
    rng: random.Random,
) -> dict[str, Any]:
    """In-place: maybe append Subtask to each ``batch['task']``. No-op if no ``subtask``."""
    if "subtask" not in batch or "task" not in batch:
        return batch
    tasks = batch["task"]
    subtasks = batch["subtask"]
    if isinstance(subtasks, str):
        subtasks = [subtasks]
    batch["task"] = [
        compose_task_with_subtask(
            task,
            subtask if (subtask and str(subtask).strip() and rng.random() >= dropout_prob) else None,
        )
        for task, subtask in zip(tasks, subtasks, strict=True)
    ]
    return batch
