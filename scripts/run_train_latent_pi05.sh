#!/usr/bin/env bash
# Train PI0.5 (or other VLA) in VAE latent action space — RTR recipe.
#
# Prerequisites:
#   1. Trained VAE checkpoint: outputs/vae_train/.../checkpoints/latest.ckpt
#   2. Latent stats JSON:      outputs/vae_eval/.../dataset_stats.json
#
# Example:
#   DATASET=your_user/screw_sorting \
#   VAE_CKPT=outputs/vae_train/screw_sorting/checkpoints/latest.ckpt \
#   LATENT_STATS=outputs/vae_eval/screw_sorting/dataset_stats.json \
#   bash scripts/run_train_latent_pi05.sh

set -euo pipefail

DATASET="${DATASET:-your_user/screw_sorting}"
VAE_CKPT="${VAE_CKPT:-outputs/vae_train/${DATASET//\//_}/checkpoints/latest.ckpt}"
LATENT_STATS="${LATENT_STATS:-outputs/vae_eval/${DATASET//\//_}/dataset_stats.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pi05_latent_${DATASET//\//_}}"
STEPS="${STEPS:-40000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
BATCH_SIZE="${BATCH_SIZE:-20}"
CHUNK_SIZE="${CHUNK_SIZE:-48}"
ACTION_DIM="${ACTION_DIM:-7}"
STATE_DIM="${STATE_DIM:-32}"
PRETRAINED="${PRETRAINED:-lerobot/pi05_base}"

lerobot-train-latent \
    --dataset.repo_id="${DATASET}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="pi05_latent_${DATASET//\//_}" \
    --policy.type=pi05 \
    --policy.pretrained_path="${PRETRAINED}" \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.n_action_steps="${CHUNK_SIZE}" \
    --policy.max_action_dim=10 \
    --policy.max_state_dim="${STATE_DIM}" \
    --policy.push_to_hub=false \
    --vae.action_dim="${ACTION_DIM}" \
    --vae.horizon="${CHUNK_SIZE}" \
    --vae.n_embed=10 \
    --vae.n_latent_dims=8 \
    --vae.use_conv_encoder=true \
    --vae.conv_latent_dims=32 \
    --vae.conv_layer_num=1 \
    --vae_checkpoint_path="${VAE_CKPT}" \
    --latent_dataset_statistics="${LATENT_STATS}" \
    --temporal_downsample_ratio=4 \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --batch_size="${BATCH_SIZE}" \
    --eval_freq=500 \
    --log_freq=200
