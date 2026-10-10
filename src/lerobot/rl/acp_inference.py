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

"""ACP / CFG-RL inference helpers (RLinf-aligned prompt + mode resolution)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.rl.acp_tags import build_acp_tagged_task
from lerobot.rl.subtask_prompt import compose_task_with_subtask

# RLinf ``guidance_type``: no_guide | positive  (+ optional negative, unused here).
ACP_INFER_MODES = ("no_guide", "positive")
# Accepted aliases mapped to ``positive`` for older scripts / docs.
_POSITIVE_ALIASES = frozenset({"positive", "cond_positive", "cfg_positive"})


@dataclass
class ACPInferenceConfig:
    """ACP / CFG-RL inference controls (RLinf-aligned).

    - ``mode="no_guide"``: plain task only (RLinf ``guidance_type=no_guide``).
    - ``mode="positive"``: CFG with ``Advantage: positive``; mix weight is ``cfg_beta``
      (RLinf ``guidance_type=positive`` + ``cfgrl_guidance_scale``). When ``cfg_beta≈1``,
      sampling uses the conditional branch only (single forward).
    """

    mode: str = "no_guide"
    cfg_beta: float = 1.0


def resolve_acp_inference_mode(acp: ACPInferenceConfig) -> str:
    """Resolve final mode to ``no_guide`` or ``positive``."""
    mode = acp.mode
    if mode in _POSITIVE_ALIASES:
        return "positive"
    if mode not in ACP_INFER_MODES:
        raise ValueError(
            f"`acp_inference.mode` must be one of {ACP_INFER_MODES} "
            f"(aliases {_POSITIVE_ALIASES - {'positive'}} map to positive), got {mode!r}."
        )
    return mode


def build_policy_tasks(
    *,
    single_task: str,
    subtask: str = "",
    acp: ACPInferenceConfig,
) -> tuple[str, str | None]:
    """Return (uncond_or_primary_task, cond_task_or_None).

    - no_guide: (base, None)
    - positive: (base, base⊕Advantage:positive) — sampler may run only cond when β≈1
    """
    base = single_task
    if subtask and str(subtask).strip():
        base = compose_task_with_subtask(base, str(subtask).strip())

    mode = resolve_acp_inference_mode(acp)
    if mode == "no_guide":
        return base, None
    return base, build_acp_tagged_task(base, is_positive=True)


def attach_cfg_language_tokens(
    preprocessed_primary: dict[str, Any],
    preprocessed_cond: dict[str, Any],
) -> dict[str, Any]:
    """Copy conditional language tensors onto the primary preprocessed batch for Pi05 CFG."""
    from lerobot.policies.pi05.modeling_pi05 import (
        OBS_LANGUAGE_COND_ATTENTION_MASK,
        OBS_LANGUAGE_COND_TOKENS,
    )
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    preprocessed_primary[OBS_LANGUAGE_COND_TOKENS] = preprocessed_cond[OBS_LANGUAGE_TOKENS]
    preprocessed_primary[OBS_LANGUAGE_COND_ATTENTION_MASK] = preprocessed_cond[
        OBS_LANGUAGE_ATTENTION_MASK
    ]
    return preprocessed_primary
