#!/usr/bin/env bash
# Fast-WAM 微调：screw_sorting_single_sft_ep279_annotated_v30
# 单臂 7 维关节（含连续夹爪 right_joint6.pos），相机 high + right，30Hz，绝对动作。
# 在容器内、lerobot 仓库根目录执行。下载中断后重跑同一条命令会接着下，不要删 pretrain_model/fastwam_base/.cache。
#   只续传：RUN_TRAIN=false bash scripts_files/finetune_screw_sorting_fastwam.sh
#   下完再训：bash scripts_files/finetune_screw_sorting_fastwam.sh
# 卡在 xet 时：HF_HUB_DISABLE_XET=1 RUN_TRAIN=false bash scripts_files/finetune_screw_sorting_fastwam.sh

# ======================== 顶层配置（改这里） ========================

RUN_DOWNLOAD="${RUN_DOWNLOAD:-false}"                 # 步骤0：下载 / 续传基座、Wan VAE+文本编码器、UMT5 分词器
RUN_RECOMPUTE_STATS="${RUN_RECOMPUTE_STATS:-true}"  # 已有绝对动作 mean/std。只有统计被改坏时才重算
RUN_TRAIN="${RUN_TRAIN:-true}"
RESUME="${RESUME:-false}"                            # true 时从 OUTPUT_DIR 最近的 checkpoint 续训
CLEAR_OUTPUT="${CLEAR_OUTPUT:-false}"                # true 才删除已有输出目录

repo_id="${repo_id:-screw_sorting_single_sft_ep279_annotated_v30}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-/workspace/huggingface}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/huggingface/lerobot}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/huggingface/.cache}"
PRETRAIN_DIR="${PRETRAIN_DIR:-/workspace/lerobot/pretrain_model}"
FASTWAM_PRETRAINED_PATH="${FASTWAM_PRETRAINED_PATH:-${PRETRAIN_DIR}/fastwam_base}"

# 数据集是 7 维 state/action。基座 proprio_dim=8，加载时形状不合的 proprio 层会重新初始化。
ACTION_DIM="${ACTION_DIM:-7}"
PROPRIO_DIM="${PROPRIO_DIM:-7}"
# 两路相机在宽度上拼接。high、right 各 224，总宽 448。原始 480x640 由模型内部缩放。
IMAGE_SIZE="${IMAGE_SIZE:-[224,448]}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"               # 须满足与 num_video_frames / freq ratio 的整除关系
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"               # 推理每段执行的步数，须 <= ACTION_HORIZON
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-33}"
ACTION_VIDEO_FREQ_RATIO="${ACTION_VIDEO_FREQ_RATIO:-4}"

# 5B 视频骨干 + 视频损失。H20 上先用 batch 1，OOM 再把 FREEZE_VIDEO_EXPERT 打开。
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STEPS="${STEPS:-30000}"
LOG_FREQ="${LOG_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
# 0–3、5 是别人的 internw0，4 是 vLLM。只使用空闲的 6、7。
GPU_IDS="${GPU_IDS:-6,7}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-true}"
# true 时只训练动作专家，并关掉视频损失。
FREEZE_VIDEO_EXPERT="${FREEZE_VIDEO_EXPERT:-false}"
LAMBDA_ACTION="${LAMBDA_ACTION:-1.0}"

WANDB_ENTITY="${WANDB_ENTITY:-1322099375-sanxia}"
# 代理对 api.wandb.ai 返回 403，在线初始化会直接让训练退出。默认离线记日志，之后可 wandb sync。
WANDB_MODE="${WANDB_MODE:-offline}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ======================================================================

set -euo pipefail

if [[ -z "${NUM_PROCESSES:-}" ]]; then
    NUM_PROCESSES="$(awk -F, '{print NF}' <<<"${GPU_IDS}")"
fi

export HF_ENDPOINT HF_HOME HF_LEROBOT_HOME HF_DATASETS_CACHE
export TORCH_NCCL_ENABLE_MONITORING=0
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF
export WANDB_ENTITY WANDB_MODE

if [[ -f "${PRETRAIN_DIR}/.fastwam_paths.env" ]]; then
    # shellcheck source=/dev/null
    source "${PRETRAIN_DIR}/.fastwam_paths.env"
fi
FASTWAM_PRETRAINED_PATH="${FASTWAM_PRETRAINED_PATH:-${PRETRAIN_DIR}/fastwam_base}"
WAN_DIFFUSERS_PATH="${WAN_DIFFUSERS_PATH:-${PRETRAIN_DIR}/Wan2.2-TI2V-5B-Diffusers}"
UMT5_TOKENIZER_PATH="${UMT5_TOKENIZER_PATH:-${PRETRAIN_DIR}/umt5-xxl}"
export WAN_DIFFUSERS_PATH UMT5_TOKENIZER_PATH

if ! command -v lerobot-train >/dev/null 2>&1; then
    if [[ -f /worksapce/miniconda3/etc/profile.d/conda.sh ]]; then
        # shellcheck source=/dev/null
        source /worksapce/miniconda3/etc/profile.d/conda.sh
        conda activate lerobot
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${RUN_DOWNLOAD}" == "true" ]]; then
    echo ">>> 步骤0：下载 / 续传 Fast-WAM 基座（已完成的文件会跳过）..."
    bash "${SCRIPT_DIR}/download_fastwam.sh"
else
    echo ">>> 跳过步骤0"
fi

if [[ ! -f "${FASTWAM_PRETRAINED_PATH}/config.json" || ! -f "${FASTWAM_PRETRAINED_PATH}/model.safetensors" ]]; then
    echo "缺少 Fast-WAM 基座: ${FASTWAM_PRETRAINED_PATH}" >&2
    echo "续传: RUN_TRAIN=false bash scripts_files/finetune_screw_sorting_fastwam.sh" >&2
    exit 1
fi

if [[ "${FREEZE_VIDEO_EXPERT}" == "true" ]]; then
    LAMBDA_VIDEO=0.0
else
    LAMBDA_VIDEO="${LAMBDA_VIDEO:-1.0}"
fi

OUTPUT_DIR="outputs/fastwam_${repo_id}"
DATASET_PATH="${HF_LEROBOT_HOME}/${repo_id}"

if [[ "${CLEAR_OUTPUT}" == "true" && "${RESUME}" != "true" ]]; then
    rm -rf "${OUTPUT_DIR}"
fi
if [[ "${RUN_TRAIN}" == "true" && "${RESUME}" != "true" && -d "${OUTPUT_DIR}" ]]; then
    echo "输出目录已存在: ${OUTPUT_DIR}" >&2
    echo "续训: RESUME=true bash scripts_files/finetune_screw_sorting_fastwam.sh" >&2
    echo "重开: CLEAR_OUTPUT=true bash scripts_files/finetune_screw_sorting_fastwam.sh" >&2
    exit 1
fi

echo ">>> Fast-WAM repo=${repo_id} horizon=${ACTION_HORIZON} exec_steps=${N_ACTION_STEPS} batch=${BATCH_SIZE}x${NUM_PROCESSES} gpus=${GPU_IDS} freeze_video=${FREEZE_VIDEO_EXPERT}"

if [[ "${RUN_RECOMPUTE_STATS}" == "true" ]]; then
    echo ">>> 重算绝对动作统计（relative_action=false）..."
    lerobot-edit-dataset \
        --repo_id "${DATASET_PATH}" \
        --operation.type recompute_stats \
        --operation.relative_action False \
        --operation.chunk_size "${ACTION_HORIZON}" \
        --operation.overwrite True \
        --new_repo_id "${DATASET_PATH}"
else
    echo ">>> 跳过统计重算，使用数据集里现有的 mean/std"
fi

if [[ "${RUN_TRAIN}" != "true" ]]; then
    echo ">>> 跳过训练"
    exit 0
fi

TRAIN_ARGS=(
    --batch_size="${BATCH_SIZE}"
    --num_workers="${NUM_WORKERS}"
    --steps="${STEPS}"
    --log_freq="${LOG_FREQ}"
    --env_eval_freq=0
    --save_freq="${SAVE_FREQ}"
    --dataset.repo_id="${DATASET_PATH}"
    --dataset.image_transforms.enable=false
    --policy.dtype=bfloat16
    --policy.action_dim="${ACTION_DIM}"
    --policy.proprio_dim="${PROPRIO_DIM}"
    --policy.action_horizon="${ACTION_HORIZON}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.num_video_frames="${NUM_VIDEO_FRAMES}"
    --policy.action_video_freq_ratio="${ACTION_VIDEO_FREQ_RATIO}"
    --policy.image_size="${IMAGE_SIZE}"
    --policy.use_gradient_checkpointing="${USE_GRADIENT_CHECKPOINTING}"
    --policy.freeze_video_expert="${FREEZE_VIDEO_EXPERT}"
    --policy.loss="{\"lambda_video\":${LAMBDA_VIDEO},\"lambda_action\":${LAMBDA_ACTION}}"
    --policy.push_to_hub=false
    --output_dir="${OUTPUT_DIR}"
    --job_name="fastwam_${repo_id}"
    --wandb.enable=true
    --wandb.mode="${WANDB_MODE}"
    --wandb.entity="${WANDB_ENTITY}"
    --wandb.project="fastwam_${repo_id}"
    --wandb.disable_artifact=true
    --wandb.notes="Fast-WAM screw sorting | init=fastwam_base | cameras=high+right | action=${ACTION_DIM} proprio=${PROPRIO_DIM} | horizon=${ACTION_HORIZON} exec=${N_ACTION_STEPS} | freeze_video=${FREEZE_VIDEO_EXPERT}"
)

if [[ "${RESUME}" == "true" ]]; then
    CONFIG_PATH="${CONFIG_PATH:-${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json}"
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "找不到续训配置: ${CONFIG_PATH}" >&2
        exit 1
    fi
    TRAIN_ARGS+=(--resume=true --config_path="${CONFIG_PATH}")
else
    # 不要在命令行追加 input_features：draccus 会和基座里的 image/image2 合并，
    # 相机宽度变成 896。proprio_dim=7 由 FastWAMConfig 在校验前写回 state。
    TRAIN_ARGS+=(--policy.path="${FASTWAM_PRETRAINED_PATH}")
fi

echo ">>> 启动 Fast-WAM 微调..."
accelerate launch \
    --multi_gpu \
    --num_processes="${NUM_PROCESSES}" \
    --gpu_ids="${GPU_IDS}" \
    --mixed_precision=bf16 \
    "$(command -v lerobot-train)" \
    "${TRAIN_ARGS[@]}"
