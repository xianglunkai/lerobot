export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
export repo_id=eval_lerobot_fold_towel_20260116_110149  # lerobot_fold_clothes_20260204_154839

# eval relative action model with ccr
python examples/rtc/eval_dataset.py \
        --policy.path=/home/xlk/work/lerobot/checkpoints/pi05_delta_act_fold_clothes40_v30/checkpoints/015000/pretrained_model \
        --dataset.repo_id=$repo_id \
        --rtc.enabled=True \
        --use_ccr=False \
        --rtc.execution_horizon=25 \
        --rtc.max_guidance_weight=10.0 \
        --rtc.prefix_attention_schedule=EXP \
        --rtc.sigma_d=1.0 \
        --inference_delay=8 \
        --num_inference_steps=10 \
        --device=cuda \
        --use_torch_compile=False \
        --next_inference_after=25 \

