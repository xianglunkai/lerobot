#!/usr/bin/env bash
# 本地单 GPU：Value Inference + RECAP advantage 计算与可视化
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TORCH_NCCL_ENABLE_MONITORING=0
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ==================== 路径 ====================
DATASET_REPO_ID="/data/huggingface/lerobot/lerobot/screw_sorting_single_recap_v30"
CHECKPOINT_PATH="/home/xlk/work/lerobot/checkpoints/checkpoints/008000/pretrained_model"
OUTPUT_DIR="outputs/pi06star_value_infer_screw_sorting_single_recap_v30_local"
JOB_NAME="pi06star_value_infer_screw_sorting_single_recap_v30_local"

# ==================== RECAP 超参（需与价值模型训练一致）====================
# checkpoint value_train_config: target_mode=recap_returns, gamma=1.0, failure_reward=-1000.0
RECAP_TARGET_MODE=recap_returns
RECAP_GAMMA=1.0
RECAP_FAILURE_REWARD=-1000.0
RECAP_N_STEP=30
RECAP_POSITIVE_RATIO=0.3
RECAP_DISCOUNT_NEXT_VALUE=true

# ==================== 可视化 ====================
VIZ_EPISODES="10,30,50,70"
VIZ_VIDEO_KEYS="observation.images.high,observation.images.right"
# 同时导出指定 episode 的 indicator 曲线图到 outputs/.../value/viz/curves/
VIZ_PLOT_CURVES=true

echo ">>> Value Inference (single GPU) | GPU=${CUDA_VISIBLE_DEVICES}"
echo "    dataset=${DATASET_REPO_ID}"
echo "    checkpoint=${CHECKPOINT_PATH}"
echo "    output=${OUTPUT_DIR}"

# lerobot-value-infer 可能未安装到 PATH，回退到仓库内脚本
VALUE_INFER_CMD="$(command -v lerobot-value-infer || true)"
if [ -z "${VALUE_INFER_CMD}" ]; then
    VALUE_INFER_CMD="${REPO_ROOT}/src/lerobot/scripts/lerobot_value_infer.py"
    echo "    entrypoint=${VALUE_INFER_CMD} (lerobot-value-infer not in PATH)"
else
    echo "    entrypoint=${VALUE_INFER_CMD}"
fi

cd "${REPO_ROOT}"

accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    "${VALUE_INFER_CMD}" \
    --runtime.batch_size=32 \
    --runtime.num_workers=1 \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.success_field=episode_success \
    --dataset.default_success=failure \
    --inference.checkpoint_path="${CHECKPOINT_PATH}" \
    --acp.enable=true \
    --acp.target_mode="${RECAP_TARGET_MODE}" \
    --acp.gamma="${RECAP_GAMMA}" \
    --acp.failure_reward="${RECAP_FAILURE_REWARD}" \
    --acp.discount_next_value="${RECAP_DISCOUNT_NEXT_VALUE}" \
    --acp.n_step="${RECAP_N_STEP}" \
    --acp.positive_ratio="${RECAP_POSITIVE_RATIO}" \
    --acp.force_intervention_positive=true \
    --viz.enable=true \
    --viz.episodes="${VIZ_EPISODES}" \
    --viz.video_keys="${VIZ_VIDEO_KEYS}" \
    --viz.overwrite=true \
    --viz.smooth_window=5 \
    --viz.plot_curves="${VIZ_PLOT_CURVES}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}"

echo ">>> Done. Results: ${OUTPUT_DIR}"
