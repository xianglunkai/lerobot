# 设置 Hugging Face 镜像端点
export HF_ENDPOINT=https://hf-mirror.com

# 安装 lerobot 包及其 pi 依赖
# pip install -e ".[pi]"
# pip install accelerate

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

export repo_id=fold_towel_v3_0

rm -rf outputs/$repo_id


accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --gpu_ids=0,1,2,3 \
    --mixed_precision=bf16 \
    $(which lerobot-train) \
    --batch_size=8 \
    --num_workers=4 \
    --steps=30000 \
    --log_freq=100 \
    --eval_freq=5000 \
    --save_freq=5000 \
    --rename_map='{"observation.images.cam_high":"observation.images.high","observation.images.cam_left_wrist":"observation.images.left","observation.images.cam_right_wrist":"observation.images.right","observation.state":"observation.state","action":"action"}' \
    --dataset.repo_id=${HF_LEROBOT_HOME}/${repo_id} \
    --policy.type=pi05 \
    --policy.dtype="bfloat16" \
    --policy.pretrained_path=/workspace/lerobot/pretrain_model/pi05_base \
    --policy.push_to_hub=false \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
    --policy.input_features='{
        "observation.images.high": {"type": "VISUAL", "shape": [480, 640, 3]},
        "observation.images.left": {"type": "VISUAL", "shape": [480, 640, 3]},
        "observation.images.right": {"type": "VISUAL", "shape": [480, 640, 3]},
        "observation.state": {"type": "STATE", "shape": [14]}
    }' \
    --policy.output_features='{"action": {"type": "ACTION", "shape": [14]}}' \
    --output_dir=outputs/$repo_id \
    --job_name=pi05_training_$repo_id \
    --wandb.enable=true \
    --wandb.project=$repo_id \
    --wandb.disable_artifact=True \
    --wandb.notes="Carefully fold the towel and then place the folded towel on the black notebook"
