from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .pipeline import PolicyActionProcessorStep, ProcessorStepRegistry


def apply_forward_tracker(
    actions: Tensor,
    *,
    dt: float,
    lead_s: float,
    exclude_dims: tuple[int, ...],
    prev_targets: list[Tensor],
) -> tuple[Tensor, list[Tensor]]:
    """Lead-compensate a joint trajectory: y = q_d + T * dq_d/dt.

    Velocity matches realtime-vla-v2 ``ForwardTracker``: zero on the first
    sample, then the slope from the oldest of the last two targets to the
    current target. Gripper dimensions listed in ``exclude_dims`` stay at
    ``q_d``. ``prev_targets`` is the target history from the previous chunk,
    oldest first, length at most 2.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    out = actions.clone()
    history = [item.detach().clone() for item in prev_targets[-2:]]
    for t in range(actions.shape[0]):
        target = actions[t]
        if history:
            vel = (target - history[0]) / (dt * len(history))
            commanded = target + vel * lead_s
        else:
            commanded = target.clone()
        for dim in exclude_dims:
            if 0 <= dim < commanded.shape[-1]:
                commanded[dim] = target[dim]
        out[t] = commanded
        history.append(target.detach().clone())
        history = history[-2:]
    return out, history


@ProcessorStepRegistry.register(name="forward_tracker_processor")
@dataclass
class ForwardTrackerProcessorStep(PolicyActionProcessorStep):
    """First-order lead compensation on absolute joint targets.

    Place this after ``absolute_actions_processor``. The plant model is
    ``T * dq/dt = y - q``, so the command that tracks ``q_d`` without lag is
    ``y = q_d + T * dq_d/dt``. ``lead_s`` is that time constant ``T``; ``alpha``
    scales it. Disabled by default.
    """

    enabled: bool = False
    lead_s: float = 0.15
    alpha: float = 1.0
    dt: float = 1.0 / 30.0
    fps: float | None = None
    exclude_dims: list[int] = field(default_factory=list)
    _prev_targets: list[Tensor] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.lead_s = float(self.lead_s)
        self.alpha = float(self.alpha)
        self.dt = 1.0 / float(self.fps) if self.fps is not None else float(self.dt)
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        self.exclude_dims = tuple(int(dim) for dim in self.exclude_dims)

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "lead_s": self.lead_s,
            "alpha": self.alpha,
            "dt": self.dt,
            "fps": self.fps,
            "exclude_dims": list(self.exclude_dims),
        }

    def state_dict(self) -> dict[str, Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        return None

    def reset(self) -> None:
        self._prev_targets = []

    def transform_features(self, features: dict) -> dict:
        return features

    def action(self, action: Tensor) -> Tensor:
        if not isinstance(action, torch.Tensor):
            raise ValueError("ForwardTrackerProcessorStep expects a torch.Tensor action")
        if not self.enabled or self.lead_s * self.alpha == 0.0:
            return action
        if action.ndim not in (2, 3):
            raise ValueError(
                f"ForwardTrackerProcessorStep expects action [T, A] or [B, T, A], got {tuple(action.shape)}"
            )

        lead_s = self.lead_s * self.alpha
        if action.ndim == 2:
            out, self._prev_targets = apply_forward_tracker(
                action,
                dt=self.dt,
                lead_s=lead_s,
                exclude_dims=self.exclude_dims,
                prev_targets=self._prev_targets,
            )
            return out

        if action.shape[0] != 1:
            # Batched offline transforms do not share one robot history.
            pieces = []
            for row in action:
                tracked, _ = apply_forward_tracker(
                    row,
                    dt=self.dt,
                    lead_s=lead_s,
                    exclude_dims=self.exclude_dims,
                    prev_targets=[],
                )
                pieces.append(tracked)
            return torch.stack(pieces, dim=0)

        tracked, self._prev_targets = apply_forward_tracker(
            action[0],
            dt=self.dt,
            lead_s=lead_s,
            exclude_dims=self.exclude_dims,
            prev_targets=self._prev_targets,
        )
        return tracked.unsqueeze(0)
