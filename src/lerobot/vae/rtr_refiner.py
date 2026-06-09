# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""Reuse-then-Refine (RTR) chunk refinement in action space via the action VAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from lerobot.vae.modeling_action_vae import ActionVAE


@dataclass
class RTRConfig:
    """RTR inference-time refinement settings (training-free)."""

    enabled: bool = True
    # Paper mode (default): concat executed actions during inference delay + new chunk[d:].
    # Released-code mode: fixed split using reuse_action_num from last refined chunk.
    mode: Literal["paper", "released"] = "paper"
    reuse_action_num: int = 24

    def validate(self, *, action_horizon: int) -> None:
        if not self.enabled:
            return
        if self.reuse_action_num <= 0:
            raise ValueError("`rtr.reuse_action_num` must be positive.")
        if self.mode == "released" and self.reuse_action_num >= action_horizon:
            raise ValueError(
                f"`rtr.reuse_action_num` ({self.reuse_action_num}) must be < action horizon ({action_horizon})."
            )


class RTRRefiner:
    """Reuse recently executed actions and refine the seam through the VAE (RTR paper)."""

    def __init__(self, vae: ActionVAE, config: RTRConfig):
        self.vae = vae
        self.config = config
        self.last_action_chunk: torch.Tensor | None = None

    def reset(self) -> None:
        self.last_action_chunk = None

    @torch.no_grad()
    def refine(
        self,
        action_chunk: torch.Tensor,
        *,
        reuse_actions: torch.Tensor | None = None,
        inference_delay: int | None = None,
    ) -> torch.Tensor:
        """Refine a decoded action chunk of shape [B, H, A].

        Paper (Fig. 5b, Sec. 4.2): concatenate
          - actions executed during the inference window (length ``d``), and
          - the non-outdated suffix of the newly decoded chunk ``new[:, d:, :]``,
        then VAE encode → decode to recover a smooth horizon-``H`` chunk.

        Released RTR code (``pi0_5_latent_rdp_vae_model_wrapper.py``) uses a fixed
        split with ``reuse_action_num`` instead of variable ``d``.
        """
        if action_chunk.ndim != 3:
            raise ValueError(f"Expected action chunk [B,T,A], got {tuple(action_chunk.shape)}")

        horizon = action_chunk.shape[1]
        self.config.validate(action_horizon=horizon)

        if not self.config.enabled:
            self.last_action_chunk = action_chunk.detach()
            return action_chunk

        now_action_chunk = action_chunk.detach()

        if self.config.mode == "paper":
            refined = self._refine_paper(
                now_action_chunk,
                reuse_actions=reuse_actions,
                inference_delay=inference_delay,
                horizon=horizon,
            )
        else:
            refined = self._refine_released(now_action_chunk, horizon=horizon)

        self.last_action_chunk = refined.detach()
        return refined

    def _refine_paper(
        self,
        now_action_chunk: torch.Tensor,
        *,
        reuse_actions: torch.Tensor | None,
        inference_delay: int | None,
        horizon: int,
    ) -> torch.Tensor:
        delay = 0 if inference_delay is None else max(0, inference_delay)
        if delay == 0 or reuse_actions is None or reuse_actions.numel() == 0:
            return now_action_chunk

        delay = min(delay, horizon)
        reuse = reuse_actions
        if reuse.ndim == 2:
            reuse = reuse.unsqueeze(0)
        if reuse.shape[0] == 1 and now_action_chunk.shape[0] > 1:
            reuse = reuse.expand(now_action_chunk.shape[0], -1, -1)

        actual_delay = min(delay, reuse.shape[1], horizon)
        if actual_delay <= 0:
            return now_action_chunk

        concate_action_chunk = torch.cat(
            [reuse[:, :actual_delay, :], now_action_chunk[:, actual_delay:, :]],
            dim=1,
        )
        if concate_action_chunk.shape[1] != horizon:
            raise ValueError(
                f"RTR concat length {concate_action_chunk.shape[1]} != horizon {horizon} "
                f"(delay={actual_delay})."
            )

        concate_action_latent = self.vae.encode_to_latent(concate_action_chunk)
        return self.vae.decode_from_latent(concate_action_latent)

    def _refine_released(self, now_action_chunk: torch.Tensor, *, horizon: int) -> torch.Tensor:
        if self.last_action_chunk is None:
            return now_action_chunk

        reuse_action_chunk = self.last_action_chunk[:, -self.config.reuse_action_num :, :]
        concate_action_chunk = torch.cat(
            [reuse_action_chunk, now_action_chunk[:, self.config.reuse_action_num :, :]],
            dim=1,
        )
        if concate_action_chunk.shape[1] != horizon:
            raise ValueError(
                f"Released RTR concat length {concate_action_chunk.shape[1]} != horizon {horizon}."
            )

        concate_action_latent = self.vae.encode_to_latent(concate_action_chunk)
        return self.vae.decode_from_latent(concate_action_latent)
