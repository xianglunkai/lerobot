export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
    
python examples/rtc/eval_with_real_robot.py \
    --policy.path=/home/xlk/work/lerobot/checkpoints/fold_towel/030000/pretrained_model \
    --policy.device=cuda \
    --robot.type=agilex_cobot \
    --rtc.enabled=True \
    --rtc.execution_horizon=20 \
    --task="Carefully fold the towel and then place the folded towel on the black notebook" \
    --duration=60 \
    --fps=30 \
    --device=cuda \
    --action_queue_size_to_get_new_actions=28 \
    --use_torch_compile=False \