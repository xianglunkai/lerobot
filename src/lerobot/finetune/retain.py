"""Utilities for parameter merging (RETAIN) inspired by
"Robust Finetuning of Vision-Language-Action".

Provides small helpers to merge two PyTorch state_dicts, with optional
per-parameter-group alphas (by name prefix or regex).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping

import torch


def merge_state_dicts(
    state_pre: Mapping[str, torch.Tensor],
    state_ft: Mapping[str, torch.Tensor],
    alpha: float = 0.5,
    per_prefix_alpha: Dict[str, float] | None = None,
) -> Dict[str, torch.Tensor]:
    """Linearly merge two state_dict-like mappings.

    Args:
        state_pre: pretrained state_dict mapping (name -> tensor)
        state_ft: finetuned state_dict mapping (name -> tensor)
        alpha: global interpolation weight in [0,1], produces
            merged = (1-alpha)*pre + alpha*ft
        per_prefix_alpha: optional dict mapping name-prefix (literal)
            or regex (if wrapped with r"^...$") to an alpha override.

    Returns:
        merged state dict (new tensors on CPU)
    """
    merged = {}
    per_prefix_alpha = per_prefix_alpha or {}

    # Precompile regexes: if a prefix string starts and ends with / treat as regex
    compiled: list[tuple[re.Pattern | None, float, str]] = []
    for k, a in per_prefix_alpha.items():
        try:
            pat = re.compile(k)
            compiled.append((pat, float(a), k))
        except re.error:
            # fallback: prefix match (literal)
            compiled.append((None, float(a), k))

    # Merge keys present in either dict
    keys = set(state_pre.keys()) | set(state_ft.keys())
    for name in keys:
        p = state_pre.get(name, None)
        f = state_ft.get(name, None)

        # determine alpha override
        a_use = alpha
        for pat, a_val, raw in compiled:
            if pat is None:
                if name.startswith(raw):
                    a_use = a_val
                    break
            else:
                if pat.search(name):
                    a_use = a_val
                    break

        if p is None:
            merged[name] = f.clone().detach().cpu()
        elif f is None:
            merged[name] = p.clone().detach().cpu()
        else:
            # ensure same shape
            if p.shape != f.shape:
                # fallback: prefer finetuned if shapes mismatch
                merged[name] = f.clone().detach().cpu()
            else:
                merged[name] = ((1.0 - a_use) * p + a_use * f).clone().detach().cpu()

    return merged


def save_state_dict(state: Mapping[str, torch.Tensor], path: str) -> None:
    torch.save(state, path)


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")
