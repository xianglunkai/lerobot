#!/usr/bin/env python3
"""从 LeRobot 数据集统计 episode 长度，并建议 failure_reward。"""

from pathlib import Path
import numpy as np
import pandas as pd

DATASET_ROOT = Path("/data/huggingface/lerobot/lerobot-data-collection/hil_screw_sorting_20260710_172937_after_deletion")
SUCCESS_FIELD = "episode_success"  # 若没有此列，会当作全部成功

# 读取所有 episode 元数据
ep_files = sorted((DATASET_ROOT / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
if not ep_files:
    raise FileNotFoundError(f"未找到 episode parquet: {DATASET_ROOT / 'meta/episodes'}")

df = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
lengths = df["length"].astype(int).to_numpy()

# 查看 episode_index = 5 的 episode_success 值
row = df[df["episode_index"] == 60]
print(row[["episode_index", "episode_success", "length"]].to_string(index=False))

print("=== 全部 episode ===")
print(f"数量: {len(lengths)}")
print(f"min / median / mean / max: {lengths.min()} / {np.median(lengths):.0f} / {lengths.mean():.1f} / {lengths.max()}")

for p in [10, 25, 50, 75, 90]:
    print(f"  p{p}: {np.percentile(lengths, p):.0f}")

# 按成功/失败分组（若有标注）
if SUCCESS_FIELD in df.columns:
    for label in df[SUCCESS_FIELD].unique():
        sub = df[df[SUCCESS_FIELD] == label]["length"].astype(int).to_numpy()
        print(f"\n=== {SUCCESS_FIELD}={label!r} (n={len(sub)}) ===")
        print(f"min / median / mean / max: {sub.min()} / {np.median(sub):.0f} / {sub.mean():.1f} / {sub.max()}")
else:
    print(f"\n(无 {SUCCESS_FIELD} 列，无法按成功/失败分组)")

# 建议 failure_reward
L_med = int(np.median(lengths))
L_max = int(lengths.max())
for k in [2, 3, 4, 5]:
    f = -k * L_med
    print(f"\n建议 failure_reward ≈ -{k}×median = {f}  (k={k}, L_median={L_med})")

# 粗算 return 范围（gamma=1, failure_reward=-3*L_med）
F = -3 * L_med
ret_success_start = -(L_med - 1)          # 成功 episode 起点 return
ret_failure_start = -(L_max - 1) + F      # 最长失败 episode 起点 return（最坏情况）
print(f"\n=== 粗算 return 范围 (failure_reward={F}, L_max={L_max}) ===")
print(f"成功起点 return (median 长度): {ret_success_start}")
print(f"失败起点 return (最坏):         {ret_failure_start}")
print(f"归一化后成功起点约: {ret_success_start / abs(ret_failure_start):.3f}")
print(f"归一化后失败起点约: {ret_failure_start / abs(ret_failure_start):.3f}")