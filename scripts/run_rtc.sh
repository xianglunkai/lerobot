export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
    
python examples/rtc/eval_with_real_robot.py \
    --policy.path=/home/xlk/work/lerobot/checkpoints/rabc_sarm_pi05/040000/pretrained_model \
    --policy.device=cuda \
    --robot.type=agilex_cobot \
    --rtc.enabled=True \
    --rtc.execution_horizon=25 \
    --rtc.max_guidance_weight=10.0 \
    --rtc.prefix_attention_schedule=EXP \
    --rtc.sigma_d=0.2 \
    --task="Carefully fold the towel and then place the folded towel on the black notebook" \
    --duration=25 \
    --fps=30 \
    --device=cuda \
    --action_queue_size_to_get_new_actions=25 \
    --use_torch_compile=False \
    --enable_visualization=true \
    --interpolation_multiplier=1 \
    --smoothing_method="mpc" \