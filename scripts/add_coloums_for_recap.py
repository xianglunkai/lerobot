#!/usr/bin/env python3
"""
为 LeRobot 数据集添加缺失的列（is_intervention 和 episode_success）、更新 info.json 并验证
用法: python add_columns.py /path/to/dataset_root
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import random


def add_is_intervention(data_dir: Path):
    """在 data/chunk-*/file-*.parquet 中添加 complementary_info.is_intervention 列"""
    parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        print(f"⚠️ 未找到任何 parquet 文件在 {data_dir}")
        return []

    processed = []
    for pq_file in parquet_files:
        print(f"📄 处理 {pq_file}")
        df = pd.read_parquet(pq_file)

        if "complementary_info.is_intervention" in df.columns:
            print(f"   ✅ 列已存在，跳过")
            continue

        # 添加 float32 列，全为 0.0
        df["complementary_info.is_intervention"] = 1.0
        df["complementary_info.is_intervention"] = df["complementary_info.is_intervention"].astype("float32")

        df.to_parquet(pq_file, index=False)
        print(f"   ✅ 已添加列，行数={len(df)}")
        processed.append(pq_file)

    print("✅ 完成 complementary_info.is_intervention 添加\n")
    return processed


def add_episode_success(meta_episodes_dir: Path):
    """在 meta/episodes/chunk-*/file-*.parquet 中添加 episode_success 列"""
    parquet_files = sorted(meta_episodes_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        print(f"⚠️ 未找到任何 parquet 文件在 {meta_episodes_dir}")
        return []

    processed = []
    for pq_file in parquet_files:
        print(f"📄 处理 {pq_file}")
        df = pd.read_parquet(pq_file)

        if "episode_success" in df.columns:
            print(f"   ✅ 列已存在，跳过")
            continue

        # 添加字符串列，值为 'success'
        df["episode_success"] = "success"

        df.to_parquet(pq_file, index=False)
        print(f"   ✅ 已添加列，行数={len(df)}")
        processed.append(pq_file)

    print("✅ 完成 episode_success 添加\n")
    return processed


def update_info_features(root: Path):
    """更新 meta/info.json 的 features 字段，添加 complementary_info.is_intervention 定义（若缺失）"""
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"⚠️ info.json 不存在: {info_path}，跳过特征更新")
        return

    with open(info_path, "r") as f:
        info = json.load(f)

    features = info.get("features", {})
    if "complementary_info.is_intervention" in features:
        print("✅ info.json 已包含 complementary_info.is_intervention 特征，无需更新")
        return

    # 添加新特征定义（与另一个数据集保持一致）
    features["complementary_info.is_intervention"] = {
        "dtype": "float32",
        "shape": [1],
        "names": ["is_intervention"]
    }
    info["features"] = features

    # 写回，保持缩进为 4 空格（与原始文件风格一致）
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print("✅ info.json 已更新，添加了 complementary_info.is_intervention 特征")


def verify_columns(root: Path, sample_size=3):
    """
    验证添加的列是否正确：
    - data/ 下的文件应包含 complementary_info.is_intervention (float32, 值=0.0)
    - meta/episodes/ 下的文件应包含 episode_success (str, 值='success')
    """
    print("=" * 60)
    print("🔍 Step 3: 验证部分数据")
    print("=" * 60)

    all_ok = True

    # 1. 验证 data/ 下的文件
    data_dir = root / "data"
    if data_dir.exists():
        files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
        if not files:
            print("⚠️ data/ 下没有 parquet 文件，跳过验证")
        else:
            # 随机抽样（若文件数少于 sample_size 则全取）
            sample = random.sample(files, min(sample_size, len(files)))
            print(f"📂 验证 data/ 下的 {len(sample)} 个文件:")
            for f in sample:
                df = pd.read_parquet(f)
                col = "complementary_info.is_intervention"
                if col not in df.columns:
                    print(f"   ❌ {f.name} 缺少列 '{col}'")
                    all_ok = False
                else:
                    # 检查类型
                    dtype = df[col].dtype
                    if dtype != "float32":
                        print(f"   ⚠️ {f.name} 列类型为 {dtype}，期望 float32")
                        all_ok = False
                    # 检查值是否全为 0.0
                    if not (df[col] == 0.0).all():
                        print(f"   ⚠️ {f.name} 列值不全为 0.0")
                        all_ok = False
                    # 打印示例值
                    sample_vals = df[col].head(3).tolist()
                    print(f"   ✅ {f.name} 列存在，类型 {dtype}，示例值 {sample_vals}")
    else:
        print("⚠️ data/ 目录不存在，跳过验证")

    # 2. 验证 meta/episodes/ 下的文件
    meta_episodes = root / "meta" / "episodes"
    if meta_episodes.exists():
        files = sorted(meta_episodes.glob("chunk-*/file-*.parquet"))
        if not files:
            print("⚠️ meta/episodes/ 下没有 parquet 文件，跳过验证")
        else:
            sample = random.sample(files, min(sample_size, len(files)))
            print(f"\n📂 验证 meta/episodes/ 下的 {len(sample)} 个文件:")
            for f in sample:
                df = pd.read_parquet(f)
                col = "episode_success"
                if col not in df.columns:
                    print(f"   ❌ {f.name} 缺少列 '{col}'")
                    all_ok = False
                else:
                    dtype = df[col].dtype
                    if dtype not in ("object", "string"):
                        print(f"   ⚠️ {f.name} 列类型为 {dtype}，期望 object/string")
                        all_ok = False
                    if not (df[col] == "success").all():
                        print(f"   ⚠️ {f.name} 列值不全为 'success'")
                        all_ok = False
                    sample_vals = df[col].head(3).tolist()
                    print(f"   ✅ {f.name} 列存在，类型 {dtype}，示例值 {sample_vals}")
    else:
        print("⚠️ meta/episodes/ 目录不存在，跳过验证")

    if all_ok:
        print("\n🎉 验证通过：所有抽查文件均包含正确的列和值。")
    else:
        print("\n⚠️ 验证发现异常，请检查上述警告。")
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="为 LeRobot 数据集添加缺失的列（is_intervention 和 episode_success），更新 info.json 并验证"
    )
    parser.add_argument(
        "root",
        type=str,
        help="数据集根目录，例如 /workspace/huggingface/lerobot/screw_sorting_single_rl_tmp_v30"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过验证步骤"
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"❌ 错误: 目录 {root} 不存在")
        return

    # Step 1: 处理 data/
    data_dir = root / "data"
    if data_dir.exists():
        add_is_intervention(data_dir)
    else:
        print(f"⚠️ 警告: data 目录不存在: {data_dir}")

    # Step 2: 更新 meta/info.json 中的 features
    update_info_features(root)

    # Step 3: 处理 meta/episodes/
    meta_episodes = root / "meta" / "episodes"
    if meta_episodes.exists():
        add_episode_success(meta_episodes)
    else:
        print(f"⚠️ 警告: meta/episodes 目录不存在: {meta_episodes}")

    # Step 4: 验证（除非用户跳过）
    if not args.no_verify:
        verify_columns(root)
    else:
        print("⏭️ 验证步骤已跳过")

    print("\n🎉 所有操作完成！")


if __name__ == "__main__":
    main()