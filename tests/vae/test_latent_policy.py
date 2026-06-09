from dataclasses import dataclass

from lerobot.vae.configuration import ActionVAEConfig
from lerobot.vae.latent_policy import configure_policy_for_latent_training


@dataclass
class _DummyPolicyConfig:
    chunk_size: int = 48
    n_action_steps: int = 48
    max_action_dim: int = 32


def test_configure_policy_for_latent_training_matches_rtr():
    policy_cfg = _DummyPolicyConfig()
    vae_cfg = ActionVAEConfig(action_dim=7, horizon=48, n_embed=10)
    configure_policy_for_latent_training(policy_cfg, vae_cfg, temporal_downsample_ratio=4)
    assert policy_cfg.chunk_size == 12
    assert policy_cfg.n_action_steps == 3
    assert policy_cfg.max_action_dim == 32
