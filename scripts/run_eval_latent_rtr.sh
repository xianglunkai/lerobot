#!/usr/bin/env bash
# Real-robot eval: latent PI0.5 + RTR refine (block_reuse). RTC is off by default.
#
# Prerequisites:
#   1. Trained latent policy checkpoint
#   2. VAE checkpoint + latent dataset_stats.json
#
# Example:
#   POLICY=outputs/pi05_latent_your_dataset/checkpoints/040000/pretrained_model \
#   VAE_CKPT=outputs/vae_train/your_dataset/checkpoints/latest.ckpt \
#   LATENT_STATS=outputs/vae_eval/your_dataset/dataset_stats.json \
#   bash scripts/run_eval_latent_rtr.sh

set -euo pipefail

POLICY="${POLICY:-outputs/pi05_latent_your_dataset/checkpoints/040000/pretrained_model}"
VAE_CKPT="${VAE_CKPT:-outputs/vae_train/your_dataset/checkpoints/latest.ckpt}"
LATENT_STATS="${LATENT_STATS:-outputs/vae_eval/your_dataset/dataset_stats.json}"
DEVICE="${DEVICE:-cuda}"
TASK="${TASK:-sort the screws}"

python examples/rtc/eval_latent_rtr_with_robot.py \
    --policy.path="${POLICY}" \
    --policy.device="${DEVICE}" \
    --policy.dtype=bfloat16 \
    --vae_checkpoint_path="${VAE_CKPT}" \
    --latent_dataset_statistics="${LATENT_STATS}" \
    --vae.horizon=48 \
    --vae.action_dim=7 \
    --vae.n_embed=10 \
    --temporal_downsample_ratio=4 \
    --rtr.enabled=true \
    --rtr.reuse_action_num=24 \
    --fps=60 \
    --action_queue_size_to_get_new_actions=24 \
    --device="${DEVICE}" \
    --task="${TASK}" \
    --duration=120 \
    "$@"
