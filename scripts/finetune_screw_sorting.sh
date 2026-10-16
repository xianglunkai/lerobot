#!/usr/bin/env bash

# ==================== 选择性执行开关 ====================
# 设置为 true 表示执行该步骤，设置为 false 表示跳过
RUN_RECOMPUTE_STATS=true   # 步骤1：重新计算数据集统计
RUN_TRAIN=true             # 步骤2：微调模型
# ======================================================

# 设置 Hugging Face 镜像端点
export HF_ENDPOINT=https://hf-mirror.com

# 设置可见的 GPU（根据实际 GPU 数量修改）
# export CUDA_VISIBLE_DEVICES=0
export TORCH_NCCL_ENABLE_MONITORING=0  # disable watchdog
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export HF_DATASETS_CACHE=/workspace/huggingface/.cache
export HF_LEROBOT_HOME=/workspace/huggingface/lerobot
export HF_HOME=/workspace/huggingface

export repo_id=screw_sorting_single_sft_ep280_v30_steam

# Pi05 原子子任务条件（scheme A）：high-level + Subtask，训练时按概率丢掉 Subtask
export CONDITION_ON_SUBTASK=true
export SUBTASK_DROPOUT_PROB=0.2

# 清理旧的输出目录（仅当需要重新训练时执行）
if [ "$RUN_TRAIN" = true ]; then
    rm -rf outputs/pi05_policy_$repo_id
fi

# ---------- 步骤1：重新计算数据集统计（可选） ----------
if [ "$RUN_RECOMPUTE_STATS" = true ]; then
    echo ">>> 步骤1：重新计算数据集统计（relative_action=true，排除 gripper）..."
    lerobot-edit-dataset \
        --repo_id ${HF_LEROBOT_HOME}/${repo_id} \
        --operation.type recompute_stats \
        --operation.relative_action True \
        --operation.chunk_size 32 \
        --operation.relative_exclude_joints "["right_joint6.pos"]" \
        --operation.overwrite True \
        --new_repo_id ${HF_LEROBOT_HOME}/${repo_id} 
else
    echo ">>> 跳过步骤1（数据集统计已存在或无需重新计算）"
fi



# ---------- 步骤2：微调模型（可选） ----------
if [ "$RUN_TRAIN" = true ]; then
    echo ">>> 步骤2：启动微调训练（使用 delta action head）..."
    accelerate launch \
        --multi_gpu \
        --num_processes=4 \
        --gpu_ids=4,5,6,7 \
        --mixed_precision=bf16 \
        $(which lerobot-train) \
        --batch_size=32 \
        --num_workers=4 \
        --steps=30000 \
        --log_freq=100 \
        --eval_freq=10000 \
        --save_freq=10000 \
        --rename_map='{"observation.images.cam_high":"observation.images.high","observation.images.cam_left_wrist":"observation.images.left","observation.images.cam_right_wrist":"observation.images.right","observation.state":"observation.state","action":"action"}' \
        --dataset.repo_id=${HF_LEROBOT_HOME}/${repo_id} \
        --use_rabc=false \
        --rabc_kappa=0.04 \
        --rabc_progress_path=${HF_LEROBOT_HOME}/${repo_id}/sarm_progress.parquet \
        --policy.type=pi05 \
        --policy.dtype="bfloat16" \
        --policy.chunk_size=32 \
        --policy.n_action_steps=32 \
        --policy.pretrained_path=/workspace/lerobot/pretrain_model/pi05_base \
        --policy.push_to_hub=false \
        --policy.compile_model=true \
        --policy.gradient_checkpointing=true \
        --policy.input_features='{
            "observation.images.high": {"type": "VISUAL", "shape": [480, 640, 3]},
            "observation.images.right": {"type": "VISUAL", "shape": [480, 640, 3]},
            "observation.state": {"type": "STATE", "shape": [7]}
        }' \
        --policy.output_features='{"action": {"type": "ACTION", "shape": [7]}}' \
        --policy.use_relative_actions=true \
        --policy.relative_exclude_joints='["right_joint6.pos"]' \
        --policy.rtc_training_config.enabled=true \
        --policy.rtc_training_config.min_delay=0 \
        --policy.rtc_training_config.max_delay=8 \
        --policy.rtc_training_config.delay_distribution=UNIFORM \
        --policy.condition_on_subtask=${CONDITION_ON_SUBTASK} \
        --policy.subtask_dropout_prob=${SUBTASK_DROPOUT_PROB} \
        --output_dir=outputs/pi05_policy_$repo_id \
        --job_name=pi05_policy_$repo_id \
        --wandb.enable=true \
        --wandb.project=pi05_policy_$repo_id \
        --wandb.disable_artifact=True \
        --wandb.notes="ACP + subtask conditioning | dropout=${SUBTASK_DROPOUT_PROB}"
else
    echo ">>> 跳过步骤2（微调未启用）"
fi