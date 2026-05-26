# ==================== 选择性执行开关 ====================
# 设置为 true 表示执行该步骤，设置为 false 表示跳过
RUN_TRAIN_SARM=false  
RUN_COMPUTE_SARM_PROGRESS=false 

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

export repo_id=screw_sorting_v30


if [ "$RUN_TRAIN_SARM" = true ]; then
    rm -rf outputs/sarm_single_$repo_id
fi

if [ "$RUN_TRAIN_SARM" = true ]; then
    echo ">>> STEP1: SARM 训练"
    accelerate launch \
        --multi_gpu \
        --num_processes=4 \
        --gpu_ids=0,1,2,3 \
        --mixed_precision=bf16 \
        $(which lerobot-train) \
        --batch_size=32 \
        --num_workers=4 \
        --steps=10000 \
        --log_freq=100 \
        --eval_freq=1000 \
        --save_freq=1000 \
        --rename_map='{"observation.images.cam_high":"observation.images.high","observation.images.cam_left_wrist":"observation.images.left","observation.images.cam_right_wrist":"observation.images.right","observation.state":"observation.state","action":"action"}' \
        --dataset.repo_id=${HF_LEROBOT_HOME}/${repo_id} \
        --policy.type=sarm \
        --policy.push_to_hub=false \
        --policy.annotation_mode=single_stage \
        --policy.image_key=observation.images.high \
        --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
        --policy.input_features='{
                "observation.images.high": {"type": "VISUAL", "shape": [480, 640, 3]},
                "observation.images.right": {"type": "VISUAL", "shape": [480, 640, 3]},
                "observation.state": {"type": "STATE", "shape": [7]}
            }' \
        --output_dir=outputs/sarm_single_$repo_id \
        --job_name=sarm_training_$repo_id \
        --wandb.enable=true \
        --wandb.project=sarm_$repo_id \
        --wandb.notes="Please sort and return the silver screws in the grey box to their proper places"
else
    echo ">>> 跳过步骤1"
fi

if [ "$RUN_COMPUTE_SARM_PROGRESS" = true ]; then
    echo ">>> STEP2: 可视化SARM 评估结果"
    python -m lerobot.policies.sarm.compute_rabc_weights \
        --dataset-repo-id ${HF_LEROBOT_HOME}/${repo_id}  \
        --reward-model-path /workspace/lerobot/outputs/sarm_single_screw_sorting_v30/checkpoints/010000/pretrained_model \
        --num-visualizations 5 \
        --head-mode sparse \
        --push-to-hub false
else
    echo ">>> 跳过步骤2"
fi



if [ "$RUN_RECOMPUTE_STATS" = true ]; then
    echo ">>> STEP3：重新计算数据集统计（relative_action=true，排除 gripper）..."
    lerobot-edit-dataset \
        --repo_id ${HF_LEROBOT_HOME}/${repo_id} \
        --operation.type recompute_stats \
        --operation.relative_action True \
        --operation.chunk_size 32 \
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
        --use_rabc=true \
        --rabc_kappa=0.02 \
        --rabc_progress_path=${HF_LEROBOT_HOME}/${repo_id}/sarm_progress.parquet \
        --rabc_head_mode=sparse \
        --policy.type=pi05 \
        --policy.dtype="bfloat16" \
        --policy.chunk_size=32 \
        --policy.n_action_steps=32 \
        --policy.pretrained_path=/workspace/lerobot/pretrain_model/pi05_base \
        --policy.push_to_hub=false \
        --policy.compile_model=true \
        --policy.gradient_checkpointing=true \
        --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
        --policy.input_features='{
            "observation.images.high": {"type": "VISUAL", "shape": [480, 640, 3]},
            "observation.images.right": {"type": "VISUAL", "shape": [480, 640, 3]},
            "observation.state": {"type": "STATE", "shape": [7]}
        }' \
        --policy.output_features='{"action": {"type": "ACTION", "shape": [7]}}' \
        --policy.empty_cameras=0 \
        --policy.use_relative_actions=true \
        --policy.relative_exclude_joints='["right_joint6.pos"]' \
        --output_dir=outputs/pi05_delta_act_$repo_id \
        --job_name=pi05_delta_act_$repo_id \
        --wandb.enable=true \
        --wandb.project=pi05_delta_act_$repo_id \
        --wandb.disable_artifact=True \
        --wandb.notes="Please sort and return the silver screws in the grey box to their proper places"
else
    echo ">>> 跳过步骤4"
fi