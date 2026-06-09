from collections.abc import Callable
from typing import TypeVar

import torch

T = TypeVar("T")


def dict_apply(x: dict[str, T], func: Callable[[T], T]) -> dict[str, T]:
    result: dict[str, T] = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result
