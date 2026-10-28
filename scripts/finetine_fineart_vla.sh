#!/usr/bin/env bash
# FineART-VLA zip_tie 微调。在容器 a1200c6f80db 内执行，数据/权重在 /workspace。
# 用法示例：
#   bash scripts_files/finetune_zip_tie_fineart.sh
#   USE_RELATIVE_ACTIONS=true RUN_RECOMPUTE_STATS=true bash scripts_files/finetune_zip_tie_fineart.sh
# 下面变量都可用环境变量覆盖：VAR=value bash ...

# ======================== 顶层配置（改这里） ========================

# --- 步骤开关 ---
RUN_RECOMPUTE_STATS="${RUN_RECOMPUTE_STATS:-true}"   # 步骤1：重算数据集统计
RUN_TRAIN="${RUN_TRAIN:-true}"                       # 步骤2：微调

# --- 数据 / 路径 ---
repo_id="${repo_id:-zip_tie_clean_v30_ep127_lerobot_annotate}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-/workspace/huggingface}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/huggingface/lerobot}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/huggingface/.cache}"
PRETRAIN_DIR="${PRETRAIN_DIR:-/workspace/lerobot/pretrain_model}"
# FineART-VLA-subtask-midtrained；先跑 scripts_files/download_model.sh
FINEART_VLA_PRETRAINED_PATH="${FINEART_VLA_PRETRAINED_PATH:-${PRETRAIN_DIR}/FineART-VLA-subtask-midtrained}"
# midtrained 自带 FAST，作 auto_fit 起点；文本分词器用本地 PaliGemma
FINEART_ACTION_TOKENIZER="${FINEART_ACTION_TOKENIZER:-${FINEART_VLA_PRETRAINED_PATH}/action_tokenizer}"
TEXT_TOKENIZER_NAME="${TEXT_TOKENIZER_NAME:-${PRETRAIN_DIR}/paligemma-3b-pt-224}"

# --- 动作表示 ---
# true=相对当前 state；false=绝对（与 midtrained 基座一致，建议默认）
# 切换后务必 RUN_RECOMPUTE_STATS=true
USE_RELATIVE_ACTIONS="${USE_RELATIVE_ACTIONS:-false}"
# 仅相对模式生效：这些关节保持绝对（通常 gripper）
RELATIVE_EXCLUDE_JOINTS="${RELATIVE_EXCLUDE_JOINTS:-[\"left_joint6.pos\",\"right_joint6.pos\"]}"

# --- 训练超参 ---
CHUNK_SIZE="${CHUNK_SIZE:-50}"           # midtrained 默认 50；RTC_MAX_DELAY 必须 < CHUNK_SIZE
# 视觉 MEM 下 SigLIP 会吃 3 相机 × MEMORY_FRAMES 帧；H20 95GB 上 batch=32 会 OOM。
# 有效全局 batch ≈ BATCH_SIZE × NUM_PROCESSES（当前 2×4=8）。OOM 再降到 1。
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STEPS="${STEPS:-30000}"
LOG_FREQ="${LOG_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
# 减轻碎片；OOM 时可再试 BATCH_SIZE=1
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- RTC ---
# 训练时 delay∈[0, RTC_MAX_DELAY]；必须 < CHUNK_SIZE。推理用 mode=trained 且 delay 不超过此值。
RTC_MAX_DELAY="${RTC_MAX_DELAY:-10}"

# --- 视觉 MEM（本体感觉 MEM 勿开：FineART 文本/FAST 路径不传 state，会在 embed_prefix 报错）---
# 数据集 fps=30；MEMORY_STRIDE=30 → 帧间隔约 1s。帧数↑显存↑，OOM 可降到 2 或先关 MEM。
MEMORY_FRAMES="${MEMORY_FRAMES:-3}"
MEMORY_STRIDE="${MEMORY_STRIDE:-30}"
MEMORY_TEMPORAL_ATTENTION_EVERY="${MEMORY_TEMPORAL_ATTENTION_EVERY:-4}"
USE_VISUAL_MEMORY="${USE_VISUAL_MEMORY:-true}"

# --- 语言 recipe / 损失 ---
# subtask/memory 由 recipe 从 language_persistent 读，不是 FineARTVLAConfig 字段。
# 默认类 recipe 只有 subtask；subtask_mem：25% 预测 subtask，60% 条件动作，15% 更新 memory。
# memory_scratchpad=false：memory 单独分支，不和 subtask 拼在同一条回复。
RECIPE_PATH="${RECIPE_PATH:-recipes/subtask_mem.yaml}"
MEMORY_SCRATCHPAD="${MEMORY_SCRATCHPAD:-false}"
TEXT_LOSS_WEIGHT="${TEXT_LOSS_WEIGHT:-1.0}"
FLOW_LOSS_WEIGHT="${FLOW_LOSS_WEIGHT:-10.0}"
# 与 midtrained 一致，勿改 fast_skip_tokens
FAST_SKIP_TOKENS="${FAST_SKIP_TOKENS:-128}"
MAX_ACTION_TOKENS="${MAX_ACTION_TOKENS:-384}"
TOKENIZER_MAX_LENGTH="${TOKENIZER_MAX_LENGTH:-256}"
AUTO_FIT_FAST_TOKENIZER="${AUTO_FIT_FAST_TOKENIZER:-true}"

# --- WandB ---
# 默认用容器 /root/.netrc 里的 1322099375-sanxia；不读 openpi/.wandb.local.env。
# 若已 export WANDB_API_KEY 则优先用环境变量。
WANDB_ENTITY="${WANDB_ENTITY:-1322099375-sanxia}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"

# ======================================================================

set -euo pipefail

export HF_ENDPOINT HF_HOME HF_LEROBOT_HOME HF_DATASETS_CACHE
export TORCH_NCCL_ENABLE_MONITORING=0
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF
export RTC_MAX_DELAY MEMORY_FRAMES MEMORY_STRIDE
export WANDB_ENTITY
unset WANDB_MODE

# --- 解析配置 / 依赖检查 ---
MODEL_PATHS_FILE="${PRETRAIN_DIR}/.model_paths.env"
if [[ -f "${MODEL_PATHS_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${MODEL_PATHS_FILE}"
fi
FINEART_VLA_PRETRAINED_PATH="${FINEART_VLA_PRETRAINED_PATH:-${PRETRAIN_DIR}/FineART-VLA-subtask-midtrained}"
FINEART_ACTION_TOKENIZER="${FINEART_ACTION_TOKENIZER:-${FINEART_VLA_PRETRAINED_PATH}/action_tokenizer}"
TEXT_TOKENIZER_NAME="${TEXT_TOKENIZER_NAME:-${PRETRAIN_DIR}/paligemma-3b-pt-224}"

if [[ ! -f "${FINEART_VLA_PRETRAINED_PATH}/config.json" ]]; then
    echo "缺少 FineART 基座: ${FINEART_VLA_PRETRAINED_PATH}" >&2
    echo "请先执行: bash scripts_files/download_model.sh" >&2
    exit 1
fi

# 可选：从文件注入 key；默认留空，交给 wandb 读 /root/.netrc
if [[ -z "${WANDB_API_KEY:-}" && -n "${WANDB_API_KEY_FILE}" && -f "${WANDB_API_KEY_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${WANDB_API_KEY_FILE}"
    set +a
fi
if [[ -z "${WANDB_API_KEY:-}" && ! -f /root/.netrc ]]; then
    echo "未设置 WANDB_API_KEY，且缺少 /root/.netrc，无法登录 entity=${WANDB_ENTITY}。" >&2
    exit 1
fi

if [[ "${USE_RELATIVE_ACTIONS}" == "true" ]]; then
    ACTION_MODE_LABEL="relative"
    RELATIVE_ACTION_BOOL=True
    POLICY_RELATIVE_ARGS=(
        --policy.use_relative_actions=true
        --policy.relative_exclude_joints="${RELATIVE_EXCLUDE_JOINTS}"
    )
    STATS_RELATIVE_ARGS=(
        --operation.relative_action True
        --operation.relative_exclude_joints "${RELATIVE_EXCLUDE_JOINTS}"
    )
else
    ACTION_MODE_LABEL="absolute"
    RELATIVE_ACTION_BOOL=False
    POLICY_RELATIVE_ARGS=(
        --policy.use_relative_actions=false
    )
    STATS_RELATIVE_ARGS=(
        --operation.relative_action False
    )
fi

echo ">>> 配置: action=${ACTION_MODE_LABEL} chunk=${CHUNK_SIZE} batch=${BATCH_SIZE}x${NUM_PROCESSES} rtc_max_delay=${RTC_MAX_DELAY} visual_mem=${USE_VISUAL_MEMORY} mem_frames=${MEMORY_FRAMES}"

if [[ "${RUN_TRAIN}" == "true" ]]; then
    rm -rf "outputs/fineart_vla_${repo_id}"
fi

# ---------- 步骤1：重新计算数据集统计 ----------
if [[ "${RUN_RECOMPUTE_STATS}" == "true" ]]; then
    echo ">>> 步骤1：重算统计（relative_action=${RELATIVE_ACTION_BOOL}）..."
    lerobot-edit-dataset \
        --repo_id "${HF_LEROBOT_HOME}/${repo_id}" \
        --operation.type recompute_stats \
        "${STATS_RELATIVE_ARGS[@]}" \
        --operation.chunk_size "${CHUNK_SIZE}" \
        --operation.overwrite True \
        --new_repo_id "${HF_LEROBOT_HOME}/${repo_id}"
else
    echo ">>> 跳过步骤1"
fi

# ---------- 步骤2：微调 ----------
if [[ "${RUN_TRAIN}" == "true" ]]; then
    echo ">>> 步骤2：FineART-VLA 微调（init=FineART-VLA-subtask-midtrained，action=${ACTION_MODE_LABEL}）..."
    accelerate launch \
        --multi_gpu \
        --num_processes="${NUM_PROCESSES}" \
        --gpu_ids="${GPU_IDS}" \
        --mixed_precision=bf16 \
        "$(which lerobot-train)" \
        --batch_size="${BATCH_SIZE}" \
        --num_workers="${NUM_WORKERS}" \
        --steps="${STEPS}" \
        --log_freq="${LOG_FREQ}" \
        --env_eval_freq=0 \
        --save_freq="${SAVE_FREQ}" \
        --dataset.repo_id="${HF_LEROBOT_HOME}/${repo_id}" \
        --policy.path="${FINEART_VLA_PRETRAINED_PATH}" \
        --policy.dtype=bfloat16 \
        --policy.chunk_size="${CHUNK_SIZE}" \
        --policy.n_action_steps="${CHUNK_SIZE}" \
        --policy.text_tokenizer_name="${TEXT_TOKENIZER_NAME}" \
        --policy.action_tokenizer_name="${FINEART_ACTION_TOKENIZER}" \
        --policy.tokenizer_max_length="${TOKENIZER_MAX_LENGTH}" \
        --policy.fast_skip_tokens="${FAST_SKIP_TOKENS}" \
        --policy.max_action_tokens="${MAX_ACTION_TOKENS}" \
        --policy.auto_fit_fast_tokenizer="${AUTO_FIT_FAST_TOKENIZER}" \
        --policy.recipe_path="${RECIPE_PATH}" \
        --policy.text_loss_weight="${TEXT_LOSS_WEIGHT}" \
        --policy.flow_loss_weight="${FLOW_LOSS_WEIGHT}" \
        --policy.memory_scratchpad="${MEMORY_SCRATCHPAD}" \
        --policy.push_to_hub=false \
        --policy.compile_model=true \
        --policy.gradient_checkpointing=true \
        --policy.input_features='{
            "observation.images.high": {"type": "VISUAL", "shape": [480, 640, 3]},
            "observation.images.left": {"type": "VISUAL", "shape": [480, 640, 3]},
            "observation.images.right": {"type": "VISUAL", "shape": [480, 640, 3]},
            "observation.state": {"type": "STATE", "shape": [14]}
        }' \
        --policy.output_features='{"action": {"type": "ACTION", "shape": [14]}}' \
        --policy.empty_cameras=0 \
        "${POLICY_RELATIVE_ARGS[@]}" \
        --policy.use_visual_memory="${USE_VISUAL_MEMORY}" \
        --policy.use_proprioceptive_memory=false \
        --policy.memory_frames="${MEMORY_FRAMES}" \
        --policy.memory_stride="${MEMORY_STRIDE}" \
        --policy.memory_temporal_attention_every="${MEMORY_TEMPORAL_ATTENTION_EVERY}" \
        --policy.rtc_training_max_delay="${RTC_MAX_DELAY}" \
        --output_dir="outputs/fineart_vla_${repo_id}" \
        --job_name="fineart_vla_${repo_id}" \
        --wandb.enable=true \
        --wandb.entity="${WANDB_ENTITY}" \
        --wandb.project="fineart_vla_${repo_id}" \
        --wandb.disable_artifact=True \
        --wandb.notes="FineART-VLA zip_tie | init=FineART-VLA-subtask-midtrained | horizon=${CHUNK_SIZE} | action=${ACTION_MODE_LABEL} | visual MEM frames=${MEMORY_FRAMES} stride=${MEMORY_STRIDE} | recipe=${RECIPE_PATH} | rtc_max_delay=${RTC_MAX_DELAY} | fast_skip=${FAST_SKIP_TOKENS}"
else
    echo ">>> 跳过步骤2"
fi
