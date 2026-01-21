export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
    
python examples/rtc/eval_with_real_robot.py \
    --policy.path=/home/xlk/work/lerobot/checkpoints/fold_towel/030000/pretrained_model \
    --policy.device=cuda \
    --robot.type=agilex_cobot \
    --rtc.enabled=True \
    --rtc.execution_horizon=18 \
    --rtc.max_guidance_weight=25.0 \
    --rtc.prefix_attention_schedule=EXP \
    --rtc.debug=False \
    --rtc.debug_maxlen=500 \
    --task="Carefully fold the towel and then place the folded towel on the black notebook" \
    --duration=60 \
    --fps=30 \
    --device=cuda \
    --action_queue_size_to_get_new_actions=25 \
    --use_torch_compile=False \