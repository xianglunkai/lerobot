from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from .pipeline import PolicyActionProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register(name="spline_action_smoothing_processor")
@dataclass
class SplineActionSmoothingProcessor(PolicyActionProcessorStep):
    """Processor that smooths action trajectories using least-squares B-splines.

    This processor is intentionally lightweight and mirrors the behavior and
    error-handling style of `QPActionSmoothingProcessor`. It operates on action
    tensors of shape (B, T, A) and fits a B-spline per batch using
    `BSplineFitter` from `lerobot.utils.bspline`.

    If fitting fails for any reason the original actions are returned.
    """

    enabled: bool = False
    k: int = 3
    n_ctrl: int = 8
    verbose: bool = False

    # Internal cache (not saved)
    _fitter: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        # Nothing heavyweight at init; fitter is created lazily when first used.
        self.k = int(self.k)
        self.n_ctrl = int(self.n_ctrl)
        self.verbose = bool(self.verbose)

    def get_config(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "k": self.k,
            "n_ctrl": self.n_ctrl,
            "verbose": self.verbose,
        }

    def state_dict(self) -> dict[str, Tensor]:
        # Stateless (doesn't hold tensors)
        return {}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:  # pragma: no cover - trivial
        return None

    def transform_features(self, features: dict) -> dict:
        return features

    def _ensure_fitter(self, T: int):
        # Lazy import to keep module import cheap and to mirror qp processor style
        try:
            from lerobot.utils.bspline import BSplineFitter
        except Exception as e:
            raise

        if self._fitter is None or getattr(self._fitter, "T", None) != T:
            # Create new fitter for this sequence length
            self._fitter = BSplineFitter(T=T, k=self.k, n_ctrl=self.n_ctrl)

    def action(self, action: Tensor) -> Tensor:
        # Expect (B, T, A)
        if not isinstance(action, torch.Tensor):
            raise ValueError("SplineActionSmoothingProcessor expects a torch.Tensor as action")
        if not self.enabled:
            return action

        orig_dtype = action.dtype
        device = action.device

        # Work in float32 on CPU for the fitter
        action_fp = action.to(dtype=torch.float32).cpu()

        B, T, A = action_fp.shape

        try:
            from lerobot.utils.bspline import BSplineFitter
        except Exception as e:
            if self.verbose:
                print(f"SplineActionSmoothingProcessor: failed to import BSplineFitter ({e}), returning original action")
            return action

        try:
            # Ensure fitter for this length
            self._ensure_fitter(T)

            out_np = []
            for b in range(B):
                y_tb = action_fp[b].numpy()  # shape (T, A)

                # Fit spline; BSplineFitter.fit accepts (T, D) and returns (n_ctrl, D)
                try:
                    ctrl = self._fitter.fit(y_tb)
                    y_hat, _ = self._fitter.rebuild(ctrl)
                except Exception as e:
                    if self.verbose:
                        print(f"SplineActionSmoothingProcessor: fit failed for batch {b} ({e}), using original trajectory")
                    y_hat = y_tb


                out_np.append(y_hat)

            out_arr = torch.tensor(out_np, dtype=torch.float32)

        except Exception as e:
            if self.verbose:
                print(f"SplineActionSmoothingProcessor: unexpected error ({e}), returning original action")
            return action

        # Convert back to original dtype/device
        return out_arr.to(device=device).to(dtype=orig_dtype)
