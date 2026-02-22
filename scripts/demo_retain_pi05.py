#!/usr/bin/env python3
"""RETAIN demo for PI05 checkpoints (pi05_base + fold_towel).

This script loads two checkpoint files (supports `model.safetensors` and
PyTorch `state_dict`), performs weight-space merging (global and modality-aware),
and writes merged checkpoints to `outputs/` for inspection or further evaluation.

It does not instantiate the large PI05 model by default to avoid heavy runtime
unless the user enables `--instantiate`.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict

import torch

try:
    from safetensors.torch import load_file as st_load, save_file as st_save
    HAS_SAFETENSORS = True
except Exception:
    HAS_SAFETENSORS = False

from lerobot.finetune.retain import merge_state_dicts


def load_state(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors") and HAS_SAFETENSORS:
        sd = st_load(path)
        return {k: v.clone().detach().cpu() for k, v in sd.items()}
    else:
        # Attempt torch.load
        obj = torch.load(path, map_location="cpu")
        # If file contains a nested 'state_dict' key (huggingface style), try to extract
        if isinstance(obj, dict) and "state_dict" in obj:
            obj = obj["state_dict"]
        return {k: v.clone().detach().cpu() for k, v in obj.items()}


def save_state(sd: Dict[str, torch.Tensor], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".safetensors") and HAS_SAFETENSORS:
        st_save(sd, path)
    else:
        torch.save(sd, path)


def detect_language_prefixes(keys: list[str]) -> list[str]:
    # heuristic: return prefixes that look like language-backbone names
    candidates = set()
    for k in keys:
        if "language_model" in k or "paligemma" in k or "gemma" in k or "lm." in k:
            # use the first component as prefix
            parts = k.split(".")
            if len(parts) > 1:
                candidates.add(parts[0])
    return sorted(list(candidates))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default="pretrain_model/pi05_base/model.safetensors")
    p.add_argument("--finetuned", default="checkpoints/fold_towel/030000/pretrained_model/model.safetensors")
    p.add_argument("--outdir", default="outputs/retain_pi05")
    p.add_argument("--alphas", default="0.25,0.5,0.75")
    p.add_argument("--instantiate", action="store_true", help="Attempt to instantiate PI05Policy and run a quick forward (requires heavy deps)")
    args = p.parse_args()

    pre_path = args.pretrained
    ft_path = args.finetuned
    outdir = args.outdir
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    print("Loading pretrained:", pre_path)
    sd_pre = load_state(pre_path)
    print("Loading finetuned:", ft_path)
    sd_ft = load_state(ft_path)

    keys_pre = set(sd_pre.keys())
    keys_ft = set(sd_ft.keys())
    print(f"Pre keys: {len(keys_pre)}, FT keys: {len(keys_ft)}, intersection: {len(keys_pre & keys_ft)}")

    # report some example keys
    sample_keys = list(sorted(keys_pre | keys_ft))[:50]
    print("Sample keys:")
    for k in sample_keys[:20]:
        print("  ", k)

    # detect language/backbone prefixes heuristically
    lang_prefixes = detect_language_prefixes(list(keys_pre | keys_ft))
    print("Detected language-like prefixes:", lang_prefixes)

    os.makedirs(outdir, exist_ok=True)

    # Global merges
    for a in alphas:
        merged = merge_state_dicts(sd_pre, sd_ft, alpha=a)
        out_path = os.path.join(outdir, f"merged_global_a{a:.2f}.safetensors")
        if not out_path.endswith(".safetensors"):
            out_path = out_path + ".safetensors"
        print(f"Saving merged global alpha={a} -> {out_path}")
        save_state(merged, out_path)

    # Modality specific: merge only language/backbone params heavier
    if len(lang_prefixes) > 0:
        for prefix in lang_prefixes:
            # set alpha override for keys starting with prefix
            per_prefix = {prefix: 0.75}  # example: give finetuned more weight for LM
            merged = merge_state_dicts(sd_pre, sd_ft, alpha=0.5, per_prefix_alpha=per_prefix)
            out_path = os.path.join(outdir, f"merged_{prefix}_weighted.safetensors")
            print(f"Saving modality-weighted merged (prefix={prefix}) -> {out_path}")
            save_state(merged, out_path)

    print("Done. Merged checkpoints saved to:", outdir)
    print("Next: you can instantiate `PI05Policy.from_pretrained(pretrained_name_or_path=outdir/merged_*.safetensors)` to evaluate, or run evaluation scripts with the merged checkpoint.")


if __name__ == "__main__":
    main()
