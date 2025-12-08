export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
    
python examples/rtc/eval_with_real_robot.py \
    --policy.path=/home/xlk/work/lerobot/checkpoints/fold_towel/30k-50hz/pretrained_model \
    --policy.device=cuda \
    --robot.type=agilex_cobot \
    --rtc.enabled=true \
    --rtc.execution_horizon=25 \
    --task="Carefully fold the towel and then place the folded towel on the black notebook" \
    --duration=240 \
    --fps=50 \
    --device=cuda \
    --action_queue_size_to_get_new_actions=45