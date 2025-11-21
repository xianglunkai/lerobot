#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import pandas as pd
import numpy as np

def print_parquet_schema(parquet_schema, indent=0):
    """递归打印 Parquet Schema"""
    space = "  " * indent
    for col in parquet_schema:
        print(f"{space}- Parquet字段: {col.name} (物理类型: {col.physical_type})")
        # 如果是 group（list/struct）
        if hasattr(col, "num_fields") and col.num_fields > 0:
            for i in range(col.num_fields):
                child = col.field(i)
                print_parquet_schema(child, indent + 1)

def compute_shape(col):
    """推断列的 shape"""
    # 标量列
    if not isinstance(col.iloc[0], (list, np.ndarray)):
        return (len(col),)

    # list 或 array
    first = col.iloc[0]
    if isinstance(first, list):
        return (len(col), len(first))
    if isinstance(first, np.ndarray):
        return (len(col),) + first.shape

    return (len(col),)

def list_fields(df, parquet_schema):
    print("✅ 字段信息（可直接读取的 df.columns）:\n")

    for c in df.columns:
        col = df[c]
        dtype = col.dtype

        # 尝试推断 shape
        try:
            shape = compute_shape(col)
        except Exception:
            shape = "Unknown"

        print(f"- 字段名: {c}")
        print(f"    类型 (dtype): {dtype}")
        print(f"    形状 (shape): {shape}\n")

    print("\n📦 Parquet 原始 Schema（结构信息，不用于访问字段）:\n")
    print_parquet_schema(parquet_schema)

def show_field_content(df, field_name):
    if field_name not in df.columns:
        print(f"❌ 未找到字段 {field_name}")
        print("可用字段:")
        for c in df.columns:
            print(" ", c)
        return

    col = df[field_name]

    # 类型
    print(f"字段名: {field_name}")
    print(f"类型 (dtype): {col.dtype}")

    # 形状
    try:
        shape = compute_shape(col)
    except:
        shape = "Unknown"

    print(f"形状 (shape): {shape}")

    # 值预览
    print("前10条:")
    for i, v in enumerate(col.head(1000)):
        print(f"{i}: {v}")

def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python check_parquet.py file.parquet [字段名]")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"文件不存在: {path}")
        sys.exit(1)

    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(str(path))
    parquet_schema = parquet_file.schema

    df = parquet_file.read().to_pandas()

    if len(sys.argv) == 2:
        list_fields(df, parquet_schema)
    else:
        show_field_content(df, sys.argv[2])

if __name__ == "__main__":
    main()
