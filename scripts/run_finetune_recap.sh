#!/usr/bin/env bash

# ==================== 选择性执行开关 ====================
# 设置为 true 表示执行该步骤，设置为 false 表示跳过
RUN_VALUE_FUNCTION_TRAINING=false
RUN_VALUE_FUNCTION_INFER=false
RUN_RECOMPUTE_STATS=false
RUN_VLA_TRAIN=true

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

export repo_id=screw_sorting_single_recap_v30

export k=1

# ==================== RLinf RECAP 超参（训练与推理必须一致）====================
# target_mode=recap_returns: 使用折扣回报 G_t = r_t + gamma*G_{t+1}，失败终止奖励为 failure_reward
# failure_reward 建议: -k * L_median (k=2~3)，L_median 为数据集中位 episode 长度
export RECAP_TARGET_MODE=recap_returns
export RECAP_GAMMA=1.0
export RECAP_FAILURE_REWARD=-300.0
export RECAP_N_STEP=30
export RECAP_POSITIVE_RATIO=0.3
export RECAP_DISCOUNT_NEXT_VALUE=true
# ================================================================================

# Step 1： Value Function Training (RLinf-aligned recap_returns labels)

if [ "$RUN_VALUE_FUNCTION_TRAINING" = true ]; then
    echo ">>> STEP1: Value Function Training (RECAP recap_returns)"
    accelerate launch \
        --multi_gpu \
        --num_processes=4 \
        --gpu_ids=0,1,2,3 \
        --mixed_precision=bf16 \
        $(which lerobot-value-train) \
        --batch_size=32 \
        --num_workers=4 \
        --steps=8000 \
        --log_freq=200 \
        --save_freq=10000 \
        --rename_map='{"observation.images.cam_high":"observation.images.high","observation.images.cam_left_wrist":"observation.images.left","observation.images.cam_right_wrist":"observation.images.right","observation.state":"observation.state","action":"action"}' \
        --dataset.repo_id=${HF_LEROBOT_HOME}/${repo_id} \
        --value.type=pistar06 \
        --value.push_to_hub=false \
        --value.input_features='{
                "observation.images.high": {"type": "VISUAL", "shape": [480, 640, 3]},
                "observation.images.right": {"type": "VISUAL", "shape": [480, 640, 3]},
                "observation.state": {"type": "STATE", "shape": [7]}
            }' \
        --targets.target_mode=${RECAP_TARGET_MODE} \
        --targets.gamma=${RECAP_GAMMA} \
        --targets.failure_reward=${RECAP_FAILURE_REWARD} \
        --targets.success_field=episode_success \
        --targets.default_success=failure \
        --output_dir=outputs/pi06star_value_train_${repo_id}_round_${k} \
        --job_name=pi06star_value_train_${repo_id}_${k} \
        --wandb.enable=true \
        --wandb.project=pi06star_value_train_${repo_id}_${k} \
        --wandb.notes="RECAP recap_returns | gamma=${RECAP_GAMMA} failure_reward=${RECAP_FAILURE_REWARD}"
else
    echo ">>> 跳过步骤1"
fi


# Step 2： Value Inference (RLinf-aligned n-step advantage from recap_returns)

if [ "$RUN_VALUE_FUNCTION_INFER" = true ]; then
    echo ">>> STEP2: Value Inference (RECAP recap_returns advantage)"
    accelerate launch \
        --multi_gpu \
        --num_processes=4 \
        --gpu_ids=0,1,2,3 \
        --mixed_precision=bf16 \
        $(which lerobot-value-infer) \
        --runtime.batch_size=32 \
        --runtime.num_workers=4 \
        --dataset.repo_id=${HF_LEROBOT_HOME}/${repo_id} \
        --dataset.success_field=episode_success \
        --dataset.default_success=failure \
        --inference.checkpoint_path=outputs/pi06star_value_train_${repo_id}_round_${k}/checkpoints/last/pretrained_model \
        --acp.enable=true \
        --acp.target_mode=${RECAP_TARGET_MODE} \
        --acp.gamma=${RECAP_GAMMA} \
        --acp.failure_reward=${RECAP_FAILURE_REWARD} \
        --acp.discount_next_value=${RECAP_DISCOUNT_NEXT_VALUE} \
        --acp.n_step=${RECAP_N_STEP} \
        --acp.positive_ratio=${RECAP_POSITIVE_RATIO} \
        --acp.force_intervention_positive=true \
        --viz.enable=true \
        --viz.episodes=10,42 \
        --viz.video_keys=observation.images.high,observation.images.right \
        --viz.overwrite=true \
        --viz.smooth_window=5 \
        --output_dir=outputs/pi06star_value_infer_${repo_id}_round_${k} \
        --job_name=pi06star_value_infer_${repo_id}_${k}

else
    echo ">>> 跳过步骤2"
fi


if [ "$RUN_RECOMPUTE_STATS" = true ]; then
    echo ">>> STEP3：重新计算数据集统计（relative_action=true，排除 gripper）..."
    lerobot-edit-dataset \
        --repo_id ${HF_LEROBOT_HOME}/${repo_id} \
        --operation.type recompute_stats \
        --operation.relative_action True \
        --operation.chunk_size 30 \
        --operation.relative_exclude_joints "["right_joint6.pos"]" \
        --operation.overwrite True \
        --new_repo_id ${HF_LEROBOT_HOME}/${repo_id}
else
    echo ">>> 跳过步骤3"
fi


if [ "$RUN_VLA_TRAIN" = true ]; then
    echo ">>> STEP4：启动微调训练（使用 delta action head + ACP）..."
    accelerate launch \
        --multi_gpu \
        --num_processes=4 \
        --gpu_ids=0,1,2,3 \
        --mixed_precision=bf16 \
        $(which lerobot-train) \
        --batch_size=32 \
        --num_workers=4 \
        --resume=false \
        --steps=30000 \
        --log_freq=100 \
        --eval_freq=10000 \
        --save_freq=10000 \
        --rename_map='{"observation.images.cam_high":"observation.images.high","observation.images.cam_left_wrist":"observation.images.left","observation.images.cam_right_wrist":"observation.images.right","observation.state":"observation.state","action":"action"}' \
        --dataset.repo_id=${HF_LEROBOT_HOME}/${repo_id} \
        --policy.type=pi05 \
        --policy.dtype="bfloat16" \
        --policy.chunk_size=30 \
        --policy.n_action_steps=30 \
        --policy.pretrained_path=/workspace/lerobot/pretrained_models/pi05_base \
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
        --acp.enable=true \
        --acp.indicator_field=complementary_info.acp_indicator \
        --acp.positive_only_conditional=true \
        --acp.indicator_dropout_prob=0.1 \
        --output_dir=outputs/pi06_policy_${repo_id}_round_${k} \
        --job_name=pi06_policy_${repo_id}_${k} \
        --wandb.enable=true \
        --wandb.project=pi06_policy_${repo_id}_${k} \
        --wandb.disable_artifact=True \
        --wandb.notes="ACP policy finetune with RECAP recap_returns advantages"
else
    echo ">>> 跳过步骤4"
fi
