export HF_LEROBOT_HOME=/home/xlk/work/data/huggingface/lerobot
export HF_HOME=/home/xlk/work/data/huggingface
    
python examples/rtc/eval_dataset.py \
        --policy.path=/home/xlk/work/lerobot/checkpoints/fold_towel/030000/pretrained_model \
        --dataset.repo_id=fold_towel_v3_0 \
        --rtc.enabled=True \
        --rtc.execution_horizon=15 \
        --rtc.max_guidance_weight=10.0 \
        --inference_delay=8 \
        --device=cpu \
        --rtc.sigma_d=0.2 \
        --rtc.full_trajectory_alignment=False
