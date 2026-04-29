export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface


# python3 -m lerobot.scripts.lerobot_dataset_viz \
#     --repo-id=lerobot/take_me_tissues_v30_new/ \
#     --root=/data/huggingface/lerobot/lerobot/take_me_tissues_v30_new/ \
#     --episode-index=28 \



python3 -m lerobot.scripts.lerobot_dataset_viz \
    --repo-id=lerobot/screw_sorting_v30/ \
    --root=/data/huggingface/lerobot/lerobot/screw_sorting_v30/ \
    --episode-index=12 \