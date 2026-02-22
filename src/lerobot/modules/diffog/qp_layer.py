"""Lightweight penalty-based optimizer backend for DiffOG.

This module provides a simple iterative penalty-based optimizer that
approximates the penalty-based trajectory optimization described in the
paper. It's intentionally simple and intended as a fast fallback for
experimentation and integration; for production or training use, replace
with a proper QP solver or `cvxpylayers` implementation.
"""
from __future__ import annotations

import numpy as np
from typing import Callable


def penalty_optimize(a: np.ndarray,
                     Q: np.ndarray,
                     B: np.ndarray,
                     dmin: np.ndarray,
                     dmax: np.ndarray,
                     alpha: float = 4.0,
                     eta: float = 0.1,
                     penalty_eta: float = 0.5,
                     iters: int = 200) -> np.ndarray:
    """Optimize trajectory with a simple penalty-based iterative method.

    Args:
        a: base action sequence flattened shape (T * p,)
        Q: fidelity matrix (size T*p x T*p) or scalar weight (treated as q*I)
        B: smoothness operator (m x T*p) such that smoothness = ||B x||^2
        dmin/dmax: bounds on delta per-dimension for the difference operator D
        alpha: smoothness weight
        eta: step size for main gradient descent
        penalty_eta: step size for penalty update
        iters: number of iterations

    Returns:
        x: optimized trajectory (T*p,)
    """
    # ensure shapes
    x = a.copy().astype(np.float64)
    n = x.size

    if np.isscalar(Q):
        H = float(Q) * np.eye(n)
    else:
        H = Q

    # Precompute B^T B
    BtB = B.T @ B

    for k in range(iters):
        # gradient of fidelity term: H (x - a)
        g = H @ (x - a)
        # gradient of smoothness term: 2 * alpha * BtB x
        g += 2.0 * alpha * (BtB @ x)

        # penalty gradient for violations on per-element difference with previous
        delta = np.zeros_like(x)
        delta[:-1] = x[1:] - x[:-1]
        # compute violations
        vmax = np.maximum(0.0, delta - dmax)
        vmin = np.maximum(0.0, dmin - delta)
        # subgradient approx
        grad_pen = np.zeros_like(x)
        for i in range(n - 1):
            if vmax[i] > 0:
                grad_pen[i] += 1.0
                grad_pen[i+1] -= 1.0
            if vmin[i] > 0:
                grad_pen[i] -= 1.0
                grad_pen[i+1] += 1.0

        x = x - eta * (g + penalty_eta * grad_pen)

    return x
