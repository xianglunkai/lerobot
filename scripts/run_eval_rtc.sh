export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
    
python examples/rtc/eval_dataset.py \
        --policy.path=/home/xlk/work/lerobot/checkpoints/hover_bottle/030000/pretrained_model \
        --dataset.repo_id=lerobot/lerobot_adjust_bottle_action_from_slave \
        --rtc.execution_horizon=8 \
        --device=cuda