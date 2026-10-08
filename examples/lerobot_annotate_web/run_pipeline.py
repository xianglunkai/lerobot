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
"""Run the official ``lerobot-annotate`` modules against a local dataset.

The VLM is DashScope's OpenAI-compatible API (``qwen3.8-max``), the same
credentials annotate-tools uses. Outputs are the official language columns:
subtask, plan, memory, task_aug, interjection, say, and vqa.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.annotations.steerable_pipeline.config import (
    AnnotationPipelineConfig,
    ExecutorConfig,
    InterjectionsConfig,
    PlanConfig,
    VlmConfig,
    VqaConfig,
)
from lerobot.annotations.steerable_pipeline.executor import Executor
from lerobot.annotations.steerable_pipeline.frames import make_frame_provider
from lerobot.annotations.steerable_pipeline.modules import (
    GeneralVqaModule,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging
from lerobot.annotations.steerable_pipeline.validator import StagingValidator
from lerobot.annotations.steerable_pipeline.vlm_client import make_vlm_client
from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review import materialize_dataset  # noqa: E402


class _SkipFinished:
    """Do not call the model again when that module's JSONL is already on disk."""

    def __init__(self, inner: Any, module: str) -> None:
        self._inner = inner
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def run_episode(self, record: Any, staging: Any) -> None:
        path = staging.path_for(self._module)
        if path.is_file() and path.stat().st_size > 0:
            print(
                f"[annotate] skip {self._module} episode {record.episode_index}: staging already exists",
                flush=True,
            )
            return
        self._inner.run_episode(record, staging)


class _SkipRejectedImages:
    """DashScope sometimes rejects a benign robot frame. Skip that call and continue."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def generate_json(self, messages_batch: Any, **kwargs: Any) -> list[Any]:
        try:
            return self._inner.generate_json(messages_batch, **kwargs)
        except Exception as exc:
            if "data_inspection_failed" not in str(exc) and "DataInspectionFailed" not in str(exc):
                raise
            print("[annotate] WARNING: DashScope rejected one image batch; this call is skipped", flush=True)
            return [None] * len(list(messages_batch))


DEFAULT_MODEL = "qwen3.8-max"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALL_FEATURES: tuple[str, ...] = ("subtask", "plan", "memory", "task_aug", "interjections", "vqa")
_STAGING_MODULES: tuple[str, ...] = ("plan", "interjections", "vqa")


@dataclass(frozen=True)
class FeatureSelection:
    """Which official language styles this run generates and writes."""

    features: frozenset[str]
    plan_enabled: bool
    emit_plan: bool
    emit_memory: bool
    n_task_rephrasings: int
    interjections_enabled: bool
    vqa_enabled: bool
    keep_styles: frozenset[str]

    @property
    def narrow(self) -> bool:
        return self.features != frozenset(ALL_FEATURES)

    def spec(self) -> str:
        return ",".join(name for name in ALL_FEATURES if name in self.features)


def feature_plan(spec: str | None) -> FeatureSelection:
    """Map a feature list onto the official plan / interjection / vqa flags.

    ``subtask`` rows are how the plan module segments an episode, so plan,
    memory, task_aug, and interjections still run that module. Styles that
    were not selected are left in staging and dropped when parquet is written.
    Interjections always include the paired ``say`` speech atom.
    """
    if spec is None or spec.strip() == "all":
        chosen = frozenset(ALL_FEATURES)
    else:
        chosen = frozenset(part.strip() for part in spec.split(",") if part.strip())
        unknown = chosen - set(ALL_FEATURES)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown features: {names}. Choose from {', '.join(ALL_FEATURES)}.")
        if not chosen:
            raise ValueError("select at least one feature")
    plan_family = {"subtask", "plan", "memory", "task_aug"}
    keep: set[str] = set()
    for name in chosen:
        if name == "interjections":
            keep.update({"interjection", "say"})
        else:
            keep.add(name)
    return FeatureSelection(
        features=chosen,
        plan_enabled=bool(chosen & plan_family) or "interjections" in chosen,
        emit_plan="plan" in chosen,
        emit_memory="memory" in chosen,
        n_task_rephrasings=4 if "task_aug" in chosen else 0,
        interjections_enabled="interjections" in chosen,
        vqa_enabled="vqa" in chosen,
        keep_styles=frozenset(keep),
    )


def _row_style(row: dict[str, Any]) -> str:
    style = row.get("style")
    if style is None:
        return "say"
    return str(style)


def filtered_staging_dir(source: Path, episode_indices: Iterable[int], keep: frozenset[str]) -> Path:
    """Copy staging for this run, keeping only the selected styles.

    The original JSONL stays intact so a later run can still reuse it.
    """
    dest = source / "_selected"
    if dest.exists():
        shutil.rmtree(dest)
    for episode_index in episode_indices:
        src = EpisodeStaging(source, episode_index)
        dst = EpisodeStaging(dest, episode_index)
        for module in _STAGING_MODULES:
            if not src.path_for(module).is_file():
                continue
            dst.write(module, [row for row in src.read(module) if _row_style(row) in keep])
    return dest


class _SelectiveWriter(LanguageColumnsWriter):
    """Write parquet from a style-filtered copy of staging."""

    def __init__(self, keep: frozenset[str]) -> None:
        super().__init__()
        self._keep = keep

    def write_all(self, records: Any, staging_dir: Path, root: Path) -> list[Path]:
        filtered = filtered_staging_dir(
            staging_dir,
            [int(record.episode_index) for record in records],
            self._keep,
        )
        return super().write_all(records, filtered, root)


def load_dashscope() -> None:
    if os.environ.get("DASHSCOPE_API_KEY", "").strip():
        return
    override = os.environ.get("FINEART_ANNOTATE_ENV", "").strip()
    candidates = [
        Path(override) if override else None,
        Path(__file__).resolve().parents[2].parent / "annotate-tools" / "annotator" / ".env",
    ]
    for path in candidates:
        if path is None or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            name, value = text.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))
        if os.environ.get("DASHSCOPE_API_KEY", "").strip():
            return
    raise RuntimeError("DASHSCOPE_API_KEY is missing. Export it or keep annotate-tools/annotator/.env.")


def run(root: Path, episodes: list[int] | None, features: str = "all") -> None:
    selection = feature_plan(features)
    load_dashscope()
    print(f"[annotate] features {selection.spec()}", flush=True)
    config = AnnotationPipelineConfig(
        root=root,
        only_episodes=tuple(episodes) if episodes else None,
        plan=PlanConfig(
            enabled=selection.plan_enabled,
            n_task_rephrasings=selection.n_task_rephrasings,
            emit_plan=selection.emit_plan,
            emit_memory=selection.emit_memory,
        ),
        interjections=InterjectionsConfig(
            enabled=selection.interjections_enabled,
            max_interjections_per_episode=2,
        ),
        vqa=VqaConfig(
            enabled=selection.vqa_enabled,
            vqa_emission_hz=0.2,
            restrict_to_default_camera=True,
            K=1,
        ),
        vlm=VlmConfig(
            backend="openai",
            model_id=os.environ.get("FINEART_VLM_MODEL", DEFAULT_MODEL),
            api_base=os.environ.get("DASHSCOPE_BASE_URL", "").strip() or DEFAULT_BASE_URL,
            api_key=os.environ["DASHSCOPE_API_KEY"],
            auto_serve=False,
            camera_key="observation.images.high",
            chat_template_kwargs={"enable_thinking": False},
            max_new_tokens=2048,
            client_concurrency=2,
        ),
        executor=ExecutorConfig(episode_parallelism=1),
    )
    frame_provider = make_frame_provider(root, camera_key=config.vlm.camera_key, video_backend=None)
    vlm = _SkipRejectedImages(make_vlm_client(config.vlm))
    executor = Executor(
        config=config,
        plan=_SkipFinished(
            PlanSubtasksMemoryModule(vlm=vlm, config=config.plan, frame_provider=frame_provider),
            "plan",
        ),
        interjections=_SkipFinished(
            InterjectionsAndSpeechModule(
                vlm=vlm, config=config.interjections, seed=config.seed, frame_provider=frame_provider
            ),
            "interjections",
        ),
        vqa=_SkipFinished(
            GeneralVqaModule(vlm=vlm, config=config.vqa, seed=config.seed, frame_provider=frame_provider),
            "vqa",
        ),
        writer=_SelectiveWriter(selection.keep_styles) if selection.narrow else LanguageColumnsWriter(),
        validator=StagingValidator(
            dataset_camera_keys=tuple(getattr(frame_provider, "camera_keys", []) or []) or None
        ),
    )
    summary = executor.run(root)
    print(f"[annotate] wrote {len(summary.written_paths)} shard(s)", flush=True)


def _parse_episodes(spec: str) -> list[int] | None:
    spec = spec.strip()
    if not spec or spec == "all":
        return None
    chosen: list[int] = []
    for part in spec.split(","):
        piece = part.strip()
        if "-" in piece:
            start_s, end_s = piece.split("-", 1)
            chosen.extend(range(int(start_s), int(end_s) + 1))
        else:
            chosen.append(int(piece))
    return sorted(set(chosen))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official LeRobot language annotation locally.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", default="all")
    parser.add_argument(
        "--features",
        default="all",
        help="Comma-separated styles, or all. Example: subtask,memory",
    )
    parser.add_argument("--mode", choices=("overwrite", "copy"), default="overwrite")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    feature_plan(args.features)
    source = args.root.expanduser().resolve()
    target = materialize_dataset(source, args.mode, args.output or None)
    print(f"[annotate] writing {target}", flush=True)
    run(target, _parse_episodes(args.episodes), args.features)


if __name__ == "__main__":
    main()
