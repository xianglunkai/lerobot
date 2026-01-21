# export HF_LEROBOT_HOME=/data/huggingface/lerobot
# export HF_HOME=/data/huggingface


python3 -m lerobot.scripts.lerobot_dataset_viz \
    --repo-id=lerobot/eval_lerobot_fold_towel_20260115_161624/ \
    --root=/data/huggingface/lerobot/lerobot/eval_lerobot_fold_towel_20260115_161624/ \
    --episode-index=0 \
    --batch-size=1 \