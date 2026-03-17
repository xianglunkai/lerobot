export HF_LEROBOT_HOME=/data/huggingface/lerobot/lerobot
export HF_HOME=/data/huggingface/lerobot
export repo_id=eval_lerobot_fold_towel_20260116_110149  # lerobot_fold_clothes_20260204_154839
    
python examples/rtc/eval_dataset.py \
        --policy.path=/home/xlk/work/lerobot/checkpoints/rabc_sarm_pi05/040000/pretrained_model \
        --dataset.repo_id=$repo_id \
        --rtc.enabled=True \
        --rtc.execution_horizon=25 \
        --rtc.max_guidance_weight=10.0 \
        --rtc.prefix_attention_schedule=EXP \
        --rtc.sigma_d=0.2 \
        --smoothing_method='mpc' \
        --inference_delay=8 \
        --num_inference_steps=10 \
        --device=cuda \
        --use_torch_compile=False \
        --next_inference_after=25 \
    
