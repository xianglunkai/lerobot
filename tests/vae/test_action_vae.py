import torch

from lerobot.utils.constants import ACTION
from lerobot.vae import ActionVAEConfig, make_action_vae
from lerobot.vae.factory import save_vae_checkpoint
from lerobot.vae.normalizer import LinearNormalizer


def test_action_vae_encode_decode_roundtrip(tmp_path):
    cfg = ActionVAEConfig(action_dim=7, horizon=48, device="cpu")
    vae = make_action_vae(cfg)

    norm = LinearNormalizer()
    actions = torch.randn(2, cfg.horizon, cfg.action_dim)
    norm.fit({ACTION: actions}, last_n_dims=1)
    vae.set_normalizer(norm)

    latent = vae.encode_to_latent(actions)
    recon = vae.decode_from_latent(latent)

    assert latent.shape[0] == actions.shape[0]
    assert recon.shape == actions.shape

    ckpt_path = save_vae_checkpoint(vae, tmp_path)
    payload = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    assert "state_dicts" in payload
