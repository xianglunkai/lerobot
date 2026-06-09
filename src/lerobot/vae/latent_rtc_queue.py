# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""Action queue for high-frequency RTR inference with async action execution."""

from __future__ import annotations

import logging
from threading import Lock

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class LatentRTRActionQueue:
    """Thread-safe queue of decoded robot actions for async RTR inference."""

    def __init__(self, temporal_ratio: int = 1):
        if temporal_ratio <= 0:
            raise ValueError("`temporal_ratio` must be positive.")
        self.temporal_ratio = temporal_ratio
        self._lock = Lock()
        self._processed_queue: Tensor | None = None
        self.last_index = 0

    def qsize(self) -> int:
        with self._lock:
            if self._processed_queue is None:
                return 0
            return len(self._processed_queue) - self.last_index

    def get_action_index(self) -> int:
        with self._lock:
            return self.last_index

    def get_processed_left_over(self) -> Tensor | None:
        """Unconsumed decoded actions still in the queue (for RTR reuse prefix)."""
        with self._lock:
            if self._processed_queue is None:
                return None
            return self._processed_queue[self.last_index :].clone()

    def get(self) -> Tensor | None:
        with self._lock:
            if self._processed_queue is None or self.last_index >= len(self._processed_queue):
                return None
            action = self._processed_queue[self.last_index]
            self.last_index += 1
            return action

    def get_consumed_since(self, start_index: int) -> int:
        """Action steps consumed since ``start_index`` (actual inference delay)."""
        with self._lock:
            return max(0, self.last_index - start_index)

    def merge(
        self,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ) -> None:
        """Append a refined chunk, skipping steps already consumed during inference."""
        with self._lock:
            effective_delay = self._effective_delay(real_delay, action_index_before_inference)
            clamped_delay = max(0, min(effective_delay, len(processed_actions)))
            new_processed = processed_actions[clamped_delay:].clone()

            if self._processed_queue is None:
                self._processed_queue = new_processed
                self.last_index = 0
                return

            self._trim_consumed_locked()
            self._processed_queue = torch.cat([self._processed_queue, new_processed])
            self.last_index = 0

            logger.debug(
                "Merged action chunk: shape=%s delay=%s",
                tuple(self._processed_queue.shape),
                clamped_delay,
            )

    def reset(self) -> None:
        with self._lock:
            self._processed_queue = None
            self.last_index = 0

    def _trim_consumed_locked(self) -> None:
        if self._processed_queue is None:
            return
        self._processed_queue = self._processed_queue[self.last_index :]
        self.last_index = 0

    def _effective_delay(self, real_delay: int, action_index_before_inference: int | None) -> int:
        if action_index_before_inference is None:
            return max(0, real_delay)
        indexes_diff = max(0, self.last_index - action_index_before_inference)
        if indexes_diff != real_delay:
            logger.debug(
                "Using index-based delay %s instead of latency-based delay %s",
                indexes_diff,
                real_delay,
            )
        return indexes_diff
