#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import h5py
import numpy as np

def list_datasets(hdf_file):
    """递归列出所有数据集路径、形状和类型"""
    print("字段信息列表:\n")
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"字段名: {name}")
            print(f"  形状 (shape): {obj.shape}")
            print(f"  类型 (dtype): {obj.dtype}\n")
    hdf_file.visititems(visit)

def show_dataset_content(hdf_file, dataset_path):
    """打印指定字段的内容"""
    if dataset_path not in hdf_file:
        print(f"❌ 未找到字段: {dataset_path}")
        return

    dataset = hdf_file[dataset_path]
    print(f"字段名: {dataset_path}")
    print(f"形状: {dataset.shape}")
    print(f"类型: {dataset.dtype}")
    print("内容预览:")

    # 仅打印部分数据，避免太大
    data = dataset[()]
    if data.size > 50:
        print(data.flatten()[:50])
        print(f"... (共 {data.size} 个元素，已截断显示)")
    else:
        print(data)

def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python check_hdf5.py /path/to/file.hdf5 [dataset_path]")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"文件不存在: {path}")
        sys.exit(1)

    try:
        with h5py.File(path, "r") as f:
            if len(sys.argv) == 2:
                list_datasets(f)
            else:
                dataset_path = sys.argv[2]
                show_dataset_content(f, dataset_path)
    except Exception as e:
        print(f"读取文件出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
