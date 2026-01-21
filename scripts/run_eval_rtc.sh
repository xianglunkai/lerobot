export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
    
python examples/rtc/eval_dataset.py \
        --policy.path=/home/xlk/work/lerobot/checkpoints/fold_towel/030000/pretrained_model \
        --dataset.repo_id=lerobot/eval_lerobot_fold_towel_20260114_141605 \
        --rtc.enabled=True \
        --rtc.execution_horizon=18 \
        --rtc.max_guidance_weight=15.0 \
        --rtc.debug=True \
        --rtc.debug_maxlen=1000 \
        --rtc.prefix_attention_schedule=EXP \
        --inference_delay=8 \
        --device=cuda \