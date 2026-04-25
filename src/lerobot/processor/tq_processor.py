from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from .pipeline import PolicyActionProcessorStep, ProcessorStepRegistry

from lerobot.utils.temporal_optimal import TimeParameterizationMPC


@ProcessorStepRegistry.register(name="time_axis_optimization_processor")
@dataclass
class TimeAxisOptimizationProcessor(PolicyActionProcessorStep):
    """Processor that smooths action trajectories in time domain using QP.

    This processor uses TimeParameterizationMPC to re-allocate time steps,
    smoothing velocities and accelerations while preserving the path shape
    (via interpolation). It mirrors the style and error handling of
    SplineActionSmoothingProcessor.
    """

    enabled: bool = False
    dt_ref: float = 0.033
    dt_min: float = 0.025
    dt_max: float = 0.041
    lambda_acc: float = 10.0
    lambda_time: float = 1.0
    stride: int = 50
    optim_dims: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    v_max: Optional[float] = None
    lambda_v: float = 1.0
    horizon: int = 50
    logging: bool = False

    # Internal cache (not saved)
    _optimizer: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        # Convert parameters to expected types
        self.dt_ref = float(self.dt_ref)
        self.dt_min = float(self.dt_min)
        self.dt_max = float(self.dt_max)
        self.lambda_acc = float(self.lambda_acc)
        self.lambda_time = float(self.lambda_time)
        self.stride = int(self.stride)
        self.horizon = int(self.horizon)
        self.lambda_v = float(self.lambda_v)
        self.logging = bool(self.logging)

    def get_config(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "dt_ref": self.dt_ref,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "lambda_acc": self.lambda_acc,
            "lambda_time": self.lambda_time,
            "stride": self.stride,
            "optim_dims": self.optim_dims,
            "v_max": self.v_max,
            "lambda_v": self.lambda_v,
            "horizon": self.horizon,
            "logging": self.logging,
        }

    def state_dict(self) -> dict[str, Tensor]:
        # Stateless (doesn't hold tensors)
        return {}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        return None

    def transform_features(self, features: dict) -> dict:
        return features

    def _ensure_optimizer(self, T: int, A: int) -> None:
        """Create or reuse optimizer instance with appropriate parameters."""
        if self._optimizer is None:
            # Create new optimizer (stateless, only holds config)
            self._optimizer = TimeParameterizationMPC(
                dt_ref=self.dt_ref,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                lambda_acc=self.lambda_acc,
                lambda_time=self.lambda_time,
                stride=self.stride,
                optim_dims=self.optim_dims,
                v_max=self.v_max,
                lambda_v=self.lambda_v,
                horizon=self.horizon,
                logging=self.logging,
            )

    def action(self, action: Tensor) -> Tensor:
        # Expect (B, T, A)
        if not isinstance(action, torch.Tensor):
            raise ValueError("TimeAxisOptimizationProcessor expects a torch.Tensor as action")
        if not self.enabled:
            return action

        orig_dtype = action.dtype
        device = action.device

        # Work in float32 on CPU for the optimizer (which uses numpy)
        action_fp = action.to(dtype=torch.float32).cpu()
        B, T, A = action_fp.shape

        try:
            # Ensure optimizer exists
            self._ensure_optimizer(T, A)

            # Prepare output list
            out_np = []

            for b in range(B):
                # Extract batch: shape (T, A) -> list of lists (T points, each of length A)
                traj_np = action_fp[b].numpy()  # (T, A)
                # Convert to list of lists (optimizer expects list of points)
                action_list = traj_np.tolist()

                try:
                    # Apply time-axis optimization
                    optimized_list = self._optimizer.optimize(action_list)
                    # Convert back to numpy array
                    y_hat = np.array(optimized_list, dtype=np.float32)
                except Exception as e:
                    if self.logging:
                        print(f"TimeAxisOptimizationProcessor: optimization failed for batch {b} ({e}), using original trajectory")
                    y_hat = traj_np

                out_np.append(y_hat)

            # Stack batches: (B, T, A)
            out_arr = np.stack(out_np, axis=0)
            out_tensor = torch.from_numpy(out_arr).to(dtype=torch.float32)

        except Exception as e:
            if self.logging:
                print(f"TimeAxisOptimizationProcessor: unexpected error ({e}), returning original action")
            return action

        # Convert back to original dtype/device
        return out_tensor.to(device=device).to(dtype=orig_dtype)