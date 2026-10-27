#!/usr/bin/env bash
set -euo pipefail

# ==================== 选择性执行开关 ====================
# 设置为 true 表示执行该步骤，设置为 false 表示跳过
RUN_RECOMPUTE_STATS=true   # 步骤1：重新计算数据集统计
RUN_TRAIN=true             # 步骤2：微调模型
# ======================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRETRAIN_DIR="${PRETRAIN_DIR:-${REPO_ROOT}/pretrain_model}"
MODEL_PATHS_FILE="${PRETRAIN_DIR}/.model_paths.env"
if [[ -f "${MODEL_PATHS_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${MODEL_PATHS_FILE}"
fi

# 优先用 download_model.sh 下载的 FineART midtrained；可用环境变量覆盖。
FINEART_VLA_PRETRAINED_PATH="${FINEART_VLA_PRETRAINED_PATH:-${PRETRAIN_DIR}/FineART-VLA-subtask-midtrained}"
TEXT_TOKENIZER_NAME="${TEXT_TOKENIZER_NAME:-${PRETRAIN_DIR}/paligemma-3b-pt-224}"
# Prefer the tokenizer shipped with the FineART checkpoint.
ACTION_TOKENIZER_NAME="${ACTION_TOKENIZER_NAME:-${FINEART_VLA_PRETRAINED_PATH}/action_tokenizer}"

if [[ ! -f "${FINEART_VLA_PRETRAINED_PATH}/config.json" ]]; then
    echo "缺少 FineART 权重: ${FINEART_VLA_PRETRAINED_PATH}" >&2
    echo "先跑: bash scripts/download_model.sh" >&2
    exit 1
fi
if [[ ! -f "${FINEART_VLA_PRETRAINED_PATH}/model.safetensors" ]]; then
    echo "FineART 权重尚未下完（缺少 model.safetensors）: ${FINEART_VLA_PRETRAINED_PATH}" >&2
    echo "继续跑: bash scripts/download_model.sh" >&2
    exit 1
fi

# 默认按本机 /data；容器里可覆盖为 /workspace/huggingface。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/.cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"

export TORCH_NCCL_ENABLE_MONITORING=0  # disable watchdog
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export repo_id=zip_tie_clean_v30_ep127_lerobot_annotate

# 数据集 fps=30。MEM 用 6 帧、间隔 1 秒，所以 memory_stride=30。
# 训练时 RTC：每个样本随机采样 delay∈[0, RTC_MAX_DELAY]，chunk 前缀保持干净动作、
# 时间置 0，flow loss 只算后面的 postfix。RTC_MAX_DELAY 必须小于 chunk_size。
# 推理要用 RTC mode=trained，且 inference delay 不能超过这个值。
# 本体感觉记忆（use_proprioceptive_memory）在 FineART 的文本/FAST 路径里没有传入 state，
# 打开会在 embed_prefix 里直接报错，因此只开视觉 MEM。
# 临时记到 jiangyuncqu 的 wandb。key 只在本进程读取，不执行 wandb login，不改 /root/.netrc。
# /root/.netrc 是同事账号 1322099375 / entity 1322099375-sanxia。
export WANDB_ENTITY="${WANDB_ENTITY:-jiangyuncqu-langx}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    WANDB_ENV_CANDIDATES=(
        "${WANDB_LOCAL_ENV:-}"
        /workspace/openpi/.wandb.local.env
        "${REPO_ROOT}/.wandb.local.env"
        "${HOME}/.wandb.local.env"
    )
    for wandb_env in "${WANDB_ENV_CANDIDATES[@]}"; do
        [[ -n "${wandb_env}" && -f "${wandb_env}" ]] || continue
        set -a
        # shellcheck source=/dev/null
        source "${wandb_env}"
        set +a
        break
    done
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY 为空，无法登录 jiangyuncqu。" >&2
    echo "设置 WANDB_API_KEY，或提供 .wandb.local.env（可用 WANDB_LOCAL_ENV=路径）。" >&2
    exit 1
fi
unset WANDB_MODE
export RTC_MAX_DELAY=10
# subtask / memory 不是 FineARTVLAConfig 字段，由 recipe 从 language_persistent 读取。
# 类里的默认 recipe 只有 subtask（30% 预测 + 70% 条件动作），不含 memory。
# recipe_path 会覆盖它：25% 预测 subtask，60% 按 subtask 出动作，15% 更新 memory。
# memory_scratchpad 保持 false：memory 是单独分支，不是和 subtask 拼在同一条回复里。
export MEMORY_FRAMES=6
export MEMORY_STRIDE=30

echo ">>> FineART pretrained: ${FINEART_VLA_PRETRAINED_PATH}"
echo ">>> text tokenizer:     ${TEXT_TOKENIZER_NAME}"
echo ">>> action tokenizer:   ${ACTION_TOKENIZER_NAME}"

if [ "$RUN_TRAIN" = true ]; then
    rm -rf outputs/fineart_vla_$repo_id
fi

# ---------- 步骤1：重新计算数据集统计（可选） ----------
if [ "$RUN_RECOMPUTE_STATS" = true ]; then
    echo ">>> 步骤1：重新计算数据集统计（relative_action=true，排除双臂 gripper）..."
    lerobot-edit-dataset \
        --repo_id ${HF_LEROBOT_HOME}/${repo_id} \
        --operation.type recompute_stats \
        --operation.relative_action True \
        --operation.chunk_size 32 \
        --operation.relative_exclude_joints '["left_joint6.pos","right_joint6.pos"]' \
        --operation.overwrite True \
        --new_repo_id ${HF_LEROBOT_HOME}/${repo_id}
else
    echo ">>> 跳过步骤1（数据集统计已存在或无需重新计算）"
fi

# ---------- 步骤2：微调模型（可选） ----------
if [ "$RUN_TRAIN" = true ]; then
    echo ">>> 步骤2：FineART-VLA 微调（从 midtrained 继续，horizon=32，视觉 MEM + 语言 memory）..."
    accelerate launch \
        --multi_gpu \
        --num_processes=4 \
        --gpu_ids=4,5,6,7 \
        --mixed_precision=bf16 \
        $(which lerobot-train) \
        --batch_size=4 \
        --num_workers=4 \
        --steps=30000 \
        --log_freq=100 \
        --env_eval_freq=0 \
        --save_freq=10000 \
        --dataset.repo_id=${HF_LEROBOT_HOME}/${repo_id} \
        --policy.type=fineart_vla \
        --policy.dtype=bfloat16 \
        --policy.chunk_size=32 \
        --policy.n_action_steps=32 \
        --policy.pretrained_path="${FINEART_VLA_PRETRAINED_PATH}" \
        --policy.text_tokenizer_name="${TEXT_TOKENIZER_NAME}" \
        --policy.action_tokenizer_name="${ACTION_TOKENIZER_NAME}" \
        --policy.tokenizer_max_length=256 \
        --policy.recipe_path=recipes/subtask_mem.yaml \
        --policy.text_loss_weight=1.0 \
        --policy.flow_loss_weight=10.0 \
        --policy.memory_scratchpad=false \
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
        --policy.use_relative_actions=true \
        --policy.relative_exclude_joints='["left_joint6.pos","right_joint6.pos"]' \
        --policy.use_visual_memory=true \
        --policy.use_proprioceptive_memory=false \
        --policy.memory_frames=${MEMORY_FRAMES} \
        --policy.memory_stride=${MEMORY_STRIDE} \
        --policy.memory_temporal_attention_every=4 \
        --policy.rtc_training_max_delay=${RTC_MAX_DELAY} \
        --output_dir=outputs/fineart_vla_$repo_id \
        --job_name=fineart_vla_$repo_id \
        --wandb.enable=true \
        --wandb.entity=${WANDB_ENTITY} \
        --wandb.project=fineart_vla_$repo_id \
        --wandb.disable_artifact=True \
        --wandb.notes="FineART-VLA zip_tie | from midtrained | horizon=32 | visual MEM frames=${MEMORY_FRAMES} stride=${MEMORY_STRIDE} | recipe=subtask_mem | rtc_max_delay=${RTC_MAX_DELAY}"
else
    echo ">>> 跳过步骤2（微调未启用）"
fi
