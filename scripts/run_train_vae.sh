#!/usr/bin/env bash
# Train RTR action-chunk VAE on a LeRobot dataset.
#
# Example:
#   DATASET=your_user/screw_sorting bash scripts/run_train_vae.sh

set -euo pipefail

DATASET="${DATASET:-your_user/screw_sorting}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/vae_train/${DATASET//\//_}}"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HORIZON="${HORIZON:-48}"
ACTION_DIM="${ACTION_DIM:-7}"

lerobot-train-vae \
    --dataset.repo_id="${DATASET}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="vae_${DATASET//\//_}" \
    --steps="${STEPS}" \
    --batch_size="${BATCH_SIZE}" \
    --vae.horizon="${HORIZON}" \
    --vae.action_dim="${ACTION_DIM}" \
    --vae.use_conv_encoder=true \
    --vae.conv_latent_dims=32 \
    --vae.conv_layer_num=1 \
    --vae.n_latent_dims=8 \
    --vae.n_embed=10 \
    --save_freq=10000 \
    --log_freq=200
