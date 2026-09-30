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

"""ACP prompt routing aligned with RLinf ``compute_cfg_routing_masks``."""

from __future__ import annotations

import random

from lerobot.rl.acp_tags import build_acp_tagged_task


def resolve_acp_conditioned_task(
    task: str,
    is_positive: bool,
    *,
    positive_only_conditional: bool,
    unconditional_prob: float,
    rng: random.Random,
) -> str:
    """Map one sample to the language prompt used for CFG/ACP training.

    Mirrors RLinf ``compute_cfg_routing_masks``:

    - ``positive_only_conditional=True``: negatives always plain task; only positives
      may receive ``Advantage: positive`` (subject to ``unconditional_prob`` dropout).
    - ``positive_only_conditional=False``: both branches may receive explicit tags
      (positive or negative), each subject to dropout.
    """
    if positive_only_conditional:
        if not is_positive:
            return task
        if unconditional_prob > 0.0 and rng.random() < unconditional_prob:
            return task
        return build_acp_tagged_task(task, is_positive=True)

    if unconditional_prob > 0.0 and rng.random() < unconditional_prob:
        return task
    return build_acp_tagged_task(task, is_positive=is_positive)
