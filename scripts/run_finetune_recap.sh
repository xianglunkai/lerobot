#!/usr/bin/env bash

# ==================== 选择性执行开关 ====================
# 设置为 true 表示执行该步骤，设置为 false 表示跳过
RUN_VALUE_FUNCTION_TRAINING=false
RUN_VALUE_FUNCTION_INFER=false
RUN_RECOMPUTE_STATS=true
RUN_VLA_TRAIN=false

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

export repo_id=screw_sorting_v30

# Step 1： Value Function Training

if [ "$RUN_VALUE_FUNCTION_TRAINING" = true ]; then
    echo ">>> STEP1: Value Function Training"
    accelerate launch \
        --multi_gpu \
        --num_processes=4 \
        --gpu_ids=4,5,6,7 \
        --mixed_precision=bf16 \
        $(which lerobot-value-train) \
        --batch_size=32 \
        --num_workers=4 \
        --steps=15000 \
        --log_freq=200 \
        --save_freq=5000 \
        --rename_map='{"observation.images.cam_high":"observation.images.high","observation.images.cam_left_wrist":"observation.images.left","observation.images.cam_right_wrist":"observation.images.right","observation.state":"observation.state","action":"action"}' \
        --dataset.repo_id=${HF_LEROBOT_HOME}/${repo_id} \
        --value.type=pistar06 \
        --value.push_to_hub=false \
        --value.input_features='{
                "observation.images.high": {"type": "VISUAL", "shape": [480, 640, 3]},
                "observation.images.right": {"type": "VISUAL", "shape": [480, 640, 3]},
                "observation.state": {"type": "STATE", "shape": [7]}
            }' \
        --output_dir=outputs/pi06star_value_train_$repo_id \
        --job_name=pi06star_value_train_$repo_id \
        --wandb.enable=true \
        --wandb.project=pi06star_value_train_$repo_id \
        --wandb.notes="Please sort and return the silver screws in the grey box to their proper places"
else
    echo ">>> 跳过步骤1"
fi


# Step 2： Value Inference

if [ "$RUN_VALUE_FUNCTION_INFER" = true ]; then
    echo ">>> STEP2: Value Inference"
    accelerate launch \
        --multi_gpu \
        --num_processes=4 \
        --gpu_ids=0,1,2,3 \
        --mixed_precision=bf16 \
        $(which lerobot-value-infer) \
        --runtime.batch_size=32 \
        --runtime.num_workers=4 \
        --dataset.repo_id=${HF_LEROBOT_HOME}/${repo_id} \
        --inference.checkpoint_path=outputs/pi06star_value_train_$repo_id/checkpoints/last/pretrained_model \
        --acp.enable=true \
        --acp.n_step=50 \
        --acp.positive_ratio=0.3 \
        --viz.enable=true \
        --viz.episodes=0-10 \
        --viz.video_keys=observation.images.high,observation.images.right \
        --viz.overwrite=true \
        --viz.smooth_window=5 \
        --output_dir=outputs/pi06star_value_infer_$repo_id \
        --job_name=pi06star_value_infer_$repo_id 

else
    echo ">>> 跳过步骤2"
fi


if [ "$RUN_RECOMPUTE_STATS" = true ]; then
    echo ">>> STEP3：重新计算数据集统计（relative_action=true，排除 gripper）..."
    lerobot-edit-dataset \
        --repo_id ${HF_LEROBOT_HOME}/${repo_id} \
        --operation.type recompute_stats \
        --operation.relative_action True \
        --operation.chunk_size 50 \
        --operation.relative_exclude_joints "["right_joint6.pos"]" \
        --operation.overwrite True
else
    echo ">>> 跳过步骤3"
fi



if [ "$RUN_VLA_TRAIN" = true ]; then
    rm -rf outputs/pi05_delta_act_$repo_id
fi
if [ "$RUN_VLA_TRAIN" = true ]; then
    echo ">>> STEP4：启动微调训练（使用 delta action head）..."
    accelerate launch \
        --multi_gpu \
        --num_processes=4 \
        --gpu_ids=0,1,2,3 \
        --mixed_precision=bf16 \
        $(which lerobot-train) \
        --batch_size=32 \
        --num_workers=4 \
        --steps=40000 \
        --log_freq=100 \
        --eval_freq=5000 \
        --save_freq=5000 \
        --rename_map='{"observation.images.cam_high":"observation.images.high","observation.images.cam_left_wrist":"observation.images.left","observation.images.cam_right_wrist":"observation.images.right","observation.state":"observation.state","action":"action"}' \
        --dataset.repo_id=${HF_LEROBOT_HOME}/${repo_id} \
        --policy.type=pi05 \
        --policy.dtype="bfloat16" \
        --policy.chunk_size=50 \
        --policy.n_action_steps=50 \
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
        --policy.empty_cameras=0 \
        --policy.use_relative_actions=true \
        --policy.relative_exclude_joints='["right_joint6.pos"]' \
        --acp.enable=true \
        --acp.indicator_field=complementary_info.acp_indicator \
        --acp.indicator_dropout_prob=0.3 \
        --output_dir=outputs/pi05_delta_act_$repo_id \
        --job_name=pi05_delta_act_$repo_id \
        --wandb.enable=true \
        --wandb.project=pi05_delta_act_$repo_id \
        --wandb.disable_artifact=True \
        --wandb.notes="Please sort and return the silver screws in the grey box to their proper places"
else
    echo ">>> 跳过步骤4"
fi