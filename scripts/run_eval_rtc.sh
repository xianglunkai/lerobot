export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
    
python examples/rtc/eval_dataset.py \
        --policy.path=/home/xlk/work/lerobot/checkpoints/fold_towel/030000/pretrained_model \
        --dataset.repo_id=lerobot/eval_lerobot_fold_towel_20260116_110149 \
        --rtc.enabled=True \
        --rtc.execution_horizon=15 \
        --rtc.max_guidance_weight=10.0 \
        --rtc.debug=True \
        --rtc.debug_maxlen=1000 \
        --inference_delay=10 \
        --device=cuda \
        --rtc.sigma_d=1.0 \
        --rtc.full_trajectory_alignment=False \