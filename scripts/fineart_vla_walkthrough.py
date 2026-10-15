#!/usr/bin/env python
"""Walk one real zip-tie episode through FineART-VLA's own functions.

Reads episode 0 of zip_tie_clean_v30_ep127_lerobot_annotate. Recipe selection,
message rendering, state discretization and the flow interpolant all call the
library code. PaliGemma is not loaded, so there is no token id and no predicted
velocity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.language_render import _select_recipe, render_sample
from lerobot.datasets.recipe import TrainingRecipe, render_message_turns
from lerobot.policies.common.flow_matching import sample_noise, sample_time_beta
from lerobot.policies.common.vla_utils import pad_vector
from lerobot.policies.fineart_vla.configuration_fineart_vla import _fineart_vla_default_recipe
from lerobot.policies.fineart_vla.processor_fineart_vla import _format_messages, discretize_state_str

ROOT = Path("/home/xlk/work/data/huggingface/lerobot/zip_tie_clean_v30_ep127_lerobot_annotate")
DATA = ROOT / "data/chunk-000/file-001.parquet"
EPISODE = 0
CHUNK = 50
MAX_ACTION_DIM = 32


def quantile_norm(values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Pi0.5 QUANTILES: 2 * (x - q01) / (q99 - q01) - 1."""
    denom = np.where(q99 - q01 == 0, 1.0, q99 - q01)
    return (2.0 * (values - q01) / denom - 1.0).astype(np.float32)


def main() -> None:
    stats = json.loads((ROOT / "meta/stats.json").read_text())
    state_q01 = np.asarray(stats["observation.state"]["q01"], dtype=np.float32)
    state_q99 = np.asarray(stats["observation.state"]["q99"], dtype=np.float32)
    action_q01 = np.asarray(stats["action"]["q01"], dtype=np.float32)
    action_q99 = np.asarray(stats["action"]["q99"], dtype=np.float32)
    tasks = pq.read_table(ROOT / "meta/tasks.parquet").to_pydict()
    task_by_index = dict(zip(tasks["task_index"], tasks["task"], strict=True))

    table = pq.read_table(
        DATA,
        columns=[
            "episode_index",
            "index",
            "timestamp",
            "task_index",
            "observation.state",
            "action",
            "language_persistent",
            "language_events",
        ],
    )
    rows = table.to_pydict()
    positions = [i for i, episode in enumerate(rows["episode_index"]) if episode == EPISODE]
    recipe = TrainingRecipe.from_dict(_fineart_vla_default_recipe())
    names = list(recipe.blend)

    def branch(sample_idx: int) -> str:
        selected = _select_recipe(recipe, sample_idx)
        for name, component in recipe.blend.items():
            if component is selected:
                return name
        raise RuntimeError(sample_idx)

    chosen: dict[str, int] = {}
    for pos in positions:
        if pos + CHUNK > positions[-1] + 1:
            continue
        timestamp = float(rows["timestamp"][pos])
        if not (7.5 < timestamp < 17.5):
            continue
        name = branch(int(rows["index"][pos]))
        chosen.setdefault(name, pos)
        if len(chosen) == len(names):
            break
    if set(chosen) != set(names):
        raise RuntimeError(f"episode {EPISODE} did not yield both branches: {chosen}")

    print(f"dataset {ROOT.name} episode {EPISODE} file {DATA.name}")
    print("task", task_by_index[int(rows["task_index"][positions[0]])])

    low_pos = chosen["low_level_execution"]
    for name in names:
        pos = chosen[name]
        sample_idx = int(rows["index"][pos])
        timestamp = float(rows["timestamp"][pos])
        task = task_by_index[int(rows["task_index"][pos])]
        persistent = rows["language_persistent"][pos]
        events = rows["language_events"][pos] or []
        rendered = render_sample(
            recipe=recipe,
            persistent=persistent,
            events=events,
            t=timestamp,
            sample_idx=sample_idx,
            task=task,
        )
        assert rendered is not None
        messages = [dict(message) for message in rendered["messages_rendered"]]
        streams = rendered["message_streams"]
        targets = rendered["target_message_indices"]
        state = np.asarray(rows["observation.state"][pos], dtype=np.float32)
        state_n = quantile_norm(state, state_q01, state_q99)
        state_str = discretize_state_str(state_n)
        print(
            f"\n=== {name}  parquet_row={pos}  dataset_index={sample_idx}  t={timestamp:.4f}s"
        )
        print("streams", streams, "targets", targets)
        print("state raw[:4]", [round(float(v), 4) for v in state[:4]])
        print("state norm[:4]", [round(float(v), 4) for v in state_n[:4]])
        print("state bins", state_str)
        if any(stream == "low_level" for stream in streams):
            for message in reversed(messages):
                if message.get("role") == "user":
                    message["content"] = f"{message['content']}, State: {state_str};"
                    break
        prompt, spans = _format_messages(messages, targets, "<eos>" if targets else None)
        print(prompt.rstrip("\n"))
        if targets:
            start, end = spans[targets[0]]
            print(f"supervised chars [{start}, {end}): {prompt[start:end]!r}")
        else:
            print("text_labels stay -100 on every token; CE sum is 0")

    pos = low_pos
    actions = np.stack(
        [np.asarray(rows["action"][i], dtype=np.float32) for i in range(pos, pos + CHUNK)]
    )
    actions_n = quantile_norm(actions, action_q01, action_q99)
    action_t = pad_vector(torch.from_numpy(actions_n).unsqueeze(0), MAX_ACTION_DIM)
    torch.manual_seed(0)
    noise = sample_noise(action_t.shape, action_t.device)
    time = sample_time_beta(
        action_t.shape[0],
        action_t.device,
        alpha=1.5,
        beta=1.0,
        scale=0.999,
        offset=0.001,
    )
    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * action_t
    u_t = noise - action_t
    print(f"\n=== flow on the low-level frame, chunk {CHUNK} x 14, padded to {MAX_ACTION_DIM}")
    print("seed 0, sample_noise + sample_time_beta(alpha=1.5, beta=1, scale=0.999, offset=0.001)")
    print("t", round(float(time), 6))
    print("a_norm[0,:4]", [round(float(v), 4) for v in action_t[0, 0, :4]])
    print("x_t[0,:4]", [round(float(v), 4) for v in x_t[0, 0, :4]])
    print("u_t[0,:4]", [round(float(v), 4) for v in u_t[0, 0, :4]])
    print("MSE uses u_t[:,:,:14] against action_out_proj; that projection is not run here")

    task = task_by_index[int(rows["task_index"][pos])]
    runtime = render_message_turns(recipe.prompt_turns("subtask"), {"task": task})
    gen_prompt, _ = _format_messages(runtime["messages_rendered"], [], None)
    print("\n=== inference next_subtask from this task")
    print((gen_prompt + "Assistant:").rstrip("\n"))
    state = np.asarray(rows["observation.state"][pos], dtype=np.float32)
    state_str = discretize_state_str(quantile_norm(state, state_q01, state_q99))
    print("action step appends the current bins after the task-slot string:")
    print(f"State: {state_str};")


if __name__ == "__main__":
    main()
