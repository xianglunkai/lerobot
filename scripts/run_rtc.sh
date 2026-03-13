export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
    
python examples/rtc/eval_with_real_robot.py \
    --policy.path=/home/xlk/work/lerobot/checkpoints/fold_towel/30k-30hz/pretrained_model \
    --policy.device=cuda \
    --robot.type=agilex_cobot \
    --rtc.enabled=True \
    --rtc.execution_horizon=20 \
    --rtc.max_guidance_weight=10.0 \
    --rtc.prefix_attention_schedule=EXP \
    --rtc.sigma_d=0.2 \
    --task="Carefully fold the towel and then place the folded towel on the black notebook" \
    --duration=25 \
    --fps=30 \
    --device=cuda \
    --action_queue_size_to_get_new_actions=16 \
    --use_torch_compile=False \
    --enable_visualization=true \
    --interpolation_multiplier=1 \
    --smoothing_method="mpc" \