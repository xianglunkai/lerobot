# 设置 Hugging Face 镜像端点
export HF_ENDPOINT=https://hf-mirror.com

# 设置可见的 GPU（根据实际 GPU 数量修改）
export CUDA_VISIBLE_DEVICES=0
export TORCH_NCCL_ENABLE_MONITORING=0  # disable watchdog

export HF_DATASETS_CACHE=/workspace/huggingface/.cache
export HF_LEROBOT_HOME=/workspace/huggingface/lerobot
export HF_HOME=/workspace/huggingface

rm -rf outputs/handover_bottle_action_from_slave_3_0

# 使用 torchrun 启动多卡训练 torchrun --standalone --nnodes=1 --nproc_per_node=4 -m 
python -m lerobot.scripts.lerobot_train \
    --dataset.root=/workspace/huggingface/lerobot/handover_bottle_action_from_slave_3_0 \
    --dataset.repo_id=handover_bottle_action_from_slave_3_0 \
    --policy.type=pi05 \
    --policy.dtype="bfloat16" \
    --policy.device=cuda \
    --policy.pretrained_path=/workspace/lerobot/pretrain_model/pi05_base \
    --policy.push_to_hub=false \
    --output_dir=outputs/handover_bottle_action_from_slave_3_0 \
    --batch_size=8 \
    --steps=30_000 \
    --num_workers=4 \
    --log_freq=200 \
    --eval_freq=5000 \
    --save_freq=5000 \
    --wandb.enable=true \
    --wandb.project=lerobot_handover_bottle_action_from_slave_3_0 \
    --wandb.notes="Use the one arm to grasp the bottle on the table, handover it to the another arm and place it on the black book"