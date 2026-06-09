# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from dataclasses import dataclass, field


@dataclass
class ActionVAEConfig:
    """Configuration for the RTR action-chunk VAE (RDP VAE variant)."""

    action_dim: int = 7
    horizon: int = 48
    n_latent_dims: int = 8
    mlp_layer_num: int = 1
    use_conv_encoder: bool = True
    conv_latent_dims: int = 32
    conv_layer_num: int = 1
    use_rnn_decoder: bool = False
    rnn_latent_dims: int = 32
    rnn_layer_num: int = 1
    use_conv_decoder: bool = False
    use_vq: bool = False
    n_embed: int = 10
    vqvae_groups: int = 4
    kl_multiplier: float = 1e-6
    encoder_loss_multiplier: float = 1.0
    act_scale: float = 1.0
    device: str = "cuda"
    eval_mode: bool = True
    second_stage: bool = False

    extended_obs: dict[str, dict[str, list[int]]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.action_dim <= 0:
            raise ValueError("`vae.action_dim` must be positive.")
        if self.horizon <= 0:
            raise ValueError("`vae.horizon` must be positive.")
        if self.use_rnn_decoder and not self.extended_obs:
            raise ValueError("`vae.extended_obs` is required when `use_rnn_decoder=true`.")

    @property
    def shape_meta(self) -> dict:
        meta: dict = {"action": {"shape": [self.action_dim]}}
        if self.extended_obs:
            meta["extended_obs"] = self.extended_obs
        return meta

    @property
    def temporal_downsample_ratio(self) -> int:
        """Latent temporal compression factor (horizon / downsampled horizon)."""
        if self.use_conv_encoder and self.conv_layer_num > 0:
            return 2 ** (self.conv_layer_num + 1)
        return 1
