from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.processor.core import TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry

from lerobot.modules.diffog import qp_layer


@dataclass
@ProcessorStepRegistry.register(name="diffog_processor")
class DiffOGProcessor(ProcessorStep):
    """A processor that applies a lightweight DiffOG penalty-based optimizer.

    This is a simple integration for experiments: it optimizes a full action
    sequence provided as a dict of joint -> 1D array (length T) in the
    transition's observation. For production or training, replace the
    backend with a proper, differentiable QP layer.
    """
    Tp: int = 100
    alpha: float = 4.0
    process_noise: float = 1e-5
    measurement_noise: float = 1e-2
    iters: int = 200
    eta: float = 0.1

    def __post_init__(self):
        pass

    @classmethod
    def from_lerobot_dataset(cls, dataset: Any, **kwargs) -> "DiffOGProcessor":
        return cls(**kwargs)

    def _build_matrices(self, joint_names: list[str], T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct Q (scalar*I), B (2nd-diff operator), and dmin/dmax for demo.

        Returns flattened shapes for p=1 per-dimension stacking across joints.
        """
        p = len(joint_names)
        n = T * p

        # Q as scalar times identity (fidelity weight)
        q_scalar = 1.0
        Q = q_scalar * np.eye(n)

        # Build second-order difference operator B for smoothness (approx acceleration)
        # B maps x (n,) -> (n-2*p,) differences per-block
        rows = max(0, (T - 2) * p)
        B = np.zeros((rows, n))
        r = 0
        for j in range(p):
            for t in range(T - 2):
                idx0 = j + t * p
                idx1 = j + (t + 1) * p
                idx2 = j + (t + 2) * p
                B[r, idx0] = 1.0
                B[r, idx1] = -2.0
                B[r, idx2] = 1.0
                r += 1

        # simple bounds on deltas: allow modest changes
        dmin = -0.2 * np.ones(n)
        dmax = 0.2 * np.ones(n)
        return Q, B, dmin, dmax

    def __call__(self, transition: dict[str, Any]) -> dict[str, Any]:
        """Expect `transition[TransitionKey.OBSERVATION.value]` to be a dict mapping
        joint name -> 1D numpy array of length T (action sequence)."""
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION.value) or new_transition.get(TransitionKey.OBSERVATION)
        if obs is None:
            raise ValueError("No observation sequence provided for DiffOGProcessor")
        if not isinstance(obs, dict):
            raise ValueError("DiffOGProcessor expects an observation dict of joint->sequence")

        joint_names = list(obs.keys())
        T = len(next(iter(obs.values())))
        # flatten into x vector stacked by time (time-major blocks per joint)
        p = len(joint_names)
        a = np.zeros(T * p, dtype=np.float64)
        for j, name in enumerate(joint_names):
            seq = np.asarray(obs[name], dtype=np.float64)
            if seq.size != T:
                raise ValueError("All joint sequences must have same length T")
            for t in range(T):
                a[t * p + j] = seq[t]

        Q, B, dmin, dmax = self._build_matrices(joint_names, T)

        x_opt = qp_layer.penalty_optimize(a, Q, B, dmin, dmax,
                                          alpha=self.alpha,
                                          eta=self.eta,
                                          iters=self.iters)

        # unflatten back to dict of sequences
        optimized = {}
        for j, name in enumerate(joint_names):
            seq = np.zeros(T, dtype=np.float64)
            for t in range(T):
                seq[t] = x_opt[t * p + j]
            optimized[name] = seq

        comp_key = TransitionKey.COMPLEMENTARY_DATA.value
        comp = (new_transition.get(comp_key) or {}).copy()
        comp["diffog_states"] = optimized
        new_transition[comp_key] = comp
        return new_transition

    def transform_features(self, features: dict) -> dict:
        return features
