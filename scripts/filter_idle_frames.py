#!/usr/bin/env python
"""
Trim idle prefix frames at the start of each LeRobot episode.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # 必须最先设置

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DONE, REWARD

# ====== 配置区 ======
ORIGINAL_REPO_ID = "screw_sorting_single_v30"
CLEANED_REPO_ID = "screw_sorting_single_cleaned_v30"
ORIGINAL_ROOT = "/workspace/huggingface/lerobot/screw_sorting_single_v30"
CLEANED_ROOT = "/workspace/huggingface/lerobot/screw_sorting_single_cleaned_v30"

ACTION_CHANGE_THRESHOLD = 1.2e-3
BUFFER_LEN = 5
MIN_KEEP_FRAMES = 20
# ===================

SKIP_KEYS = frozenset(
    {"task_index", "timestamp", "episode_index", "frame_index", "index", "task", "subtask"}
)


def find_first_non_idle_with_buffer(actions, threshold, buffer_len):
    if len(actions) == 0:
        return None
    first_action = actions[0]
    for i, act in enumerate(actions):
        if torch.linalg.vector_norm(act - first_action) > threshold:
            return max(0, i - buffer_len)
    return None


def _to_writer_image(value: torch.Tensor, expected_shape: tuple[int, ...]) -> torch.Tensor:
    """Convert __getitem__ output (usually CHW) to the layout declared in dataset features."""
    if value.ndim != 3 or len(expected_shape) != 3:
        return value

    # Metadata may store HWC, e.g. (480, 640, 3), while __getitem__ returns CHW (3, 480, 640).
    if value.shape[0] in (1, 3) and expected_shape[-1] in (1, 3) and tuple(value.shape[1:]) == expected_shape[:2]:
        return value.permute(1, 2, 0)

    # Metadata stores CHW, e.g. (3, 480, 640) — already correct.
    return value


def build_clean_frame(raw_frame: dict, features: dict) -> dict:
    clean_frame = {}
    for key, value in raw_frame.items():
        if key in SKIP_KEYS:
            continue

        ft = features.get(key)
        if ft is not None and ft["dtype"] in ("image", "video") and isinstance(value, torch.Tensor):
            value = _to_writer_image(value, tuple(ft["shape"]))

        if key in (DONE, REWARD) and isinstance(value, torch.Tensor) and value.dim() == 0:
            value = value.unsqueeze(0)
        if key.startswith("complementary_info") and isinstance(value, torch.Tensor) and value.dim() == 0:
            value = value.unsqueeze(0)
        clean_frame[key] = value
    clean_frame["task"] = raw_frame["task"]
    return clean_frame


def main():
    original_ds = LeRobotDataset(ORIGINAL_REPO_ID, root=ORIGINAL_ROOT)
    total_episodes = original_ds.meta.total_episodes

    cleaned_ds = LeRobotDataset.create(
        repo_id=CLEANED_REPO_ID,
        fps=int(original_ds.meta.fps),
        features=original_ds.meta.info["features"],
        root=CLEANED_ROOT,
        robot_type=original_ds.meta.robot_type,
        use_videos=len(original_ds.meta.video_keys) > 0,
    )

    kept, skipped = 0, 0
    hf_ds = original_ds.hf_dataset
    features = original_ds.meta.features

    for ep_idx in range(total_episodes):
        ep = original_ds.meta.episodes[ep_idx]
        from_idx, to_idx = ep["dataset_from_index"], ep["dataset_to_index"]

        actions = torch.stack(
            [torch.as_tensor(a, dtype=torch.float32) for a in hf_ds["action"][from_idx:to_idx]]
        )
        start_in_ep = find_first_non_idle_with_buffer(actions, ACTION_CHANGE_THRESHOLD, BUFFER_LEN)
        keep_len = to_idx - from_idx - (start_in_ep or 0)

        if start_in_ep is None or keep_len < MIN_KEEP_FRAMES:
            skipped += 1
            print(f"Episode {ep_idx}: skip (keep_len={keep_len})")
            continue

        global_start = from_idx + start_in_ep
        for idx in range(global_start, to_idx):
            cleaned_ds.add_frame(build_clean_frame(original_ds[idx], features))

        cleaned_ds.save_episode()
        kept += 1
        print(f"Episode {ep_idx}: trimmed {start_in_ep}, kept {keep_len}")

    if kept == 0:
        raise RuntimeError("No episodes kept after trimming.")

    cleaned_ds.finalize()
    print(f"Done: kept {kept}/{total_episodes}, skipped {skipped}")


if __name__ == "__main__":
    main()
