#!/usr/bin/env bash

# ==================== 选择性执行开关 ====================
# 设置为 true 表示执行该步骤，设置为 false 表示跳过
RUN_RECOMPUTE_STATS=false   # 步骤1：重新计算数据集统计
RUN_TRAIN=true             # 步骤2：微调模型
# ======================================================

# 在容器 a1200c6f80db 内执行。数据与权重都挂在 /workspace 下。
export HF_ENDPOINT=https://hf-mirror.com

export TORCH_NCCL_ENABLE_MONITORING=0  # disable watchdog
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export HF_DATASETS_CACHE=/workspace/huggingface/.cache
export HF_LEROBOT_HOME=/workspace/huggingface/lerobot
export HF_HOME=/workspace/huggingface

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
    if [[ ! -f /workspace/openpi/.wandb.local.env ]]; then
        echo "缺少 /workspace/openpi/.wandb.local.env，无法使用 jiangyuncqu 的 wandb key。" >&2
        exit 1
    fi
    set -a
    # shellcheck source=/dev/null
    source /workspace/openpi/.wandb.local.env
    set +a
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY 为空，无法登录 jiangyuncqu。" >&2
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
    echo ">>> 步骤2：FineART-VLA 微调（horizon=32，视觉 MEM + 语言 memory）..."
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
        --policy.pretrained_path=/workspace/lerobot/pretrain_model/pi05_base \
        --policy.text_tokenizer_name=/workspace/lerobot/pretrain_model/paligemma-3b-pt-224 \
        --policy.action_tokenizer_name=/workspace/lerobot/physical-intelligence/fast \
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
        --wandb.notes="FineART-VLA zip_tie | horizon=32 | visual MEM frames=${MEMORY_FRAMES} stride=${MEMORY_STRIDE} | recipe=subtask_mem | rtc_max_delay=${RTC_MAX_DELAY}"
else
    echo ">>> 跳过步骤2（微调未启用）"
fi
