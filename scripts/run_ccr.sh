export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
    
python examples/rtc/eval_ccr_with_robot.py \
    --policy.path=/home/xlk/work/lerobot/checkpoints/fold_towel/30k-30hz/pretrained_model \
    --policy.device=cuda \
    --robot.type=agilex_cobot \
    --rtc.enabled=true \
    --task="Carefully fold the towel and then place the folded towel on the black notebook" \
    --duration=25 \
    --fps=30 \
    --device=cuda \
    --action_queue_size_to_get_new_actions=12 \
    --use_torch_compile=False \
    --enable_visualization=True \
    --interpolation_multiplier=1 \