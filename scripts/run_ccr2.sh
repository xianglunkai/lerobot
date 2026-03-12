export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
    
python examples/rtc/eval_ccr2_with_robot.py \
    --policy.path=/home/xlk/work/lerobot/checkpoints/fold_towel/30k-30hz/pretrained_model \
    --policy.device=cuda \
    --robot.type=agilex_cobot \
    --fps=30 \
    --device=cuda \
    --task="Carefully fold the towel and then place the folded towel on the black notebook" \
    --duration=30 \
    --action_history_horizon=0 \
    --use_ccr=True \
    --ccr_k=3 \
    --ccr_n_ctrl=8 \
    --ccr_n_free=4 \
    --ccr_last_pt_weight=0.05 \
    --use_torch_compile=False \
    --enable_visualization=True \