export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface

lerobot-dataset-viz \
    --repo-id lerobot/lerobot_adjust_bottle_action_from_slave \
    --mode "local" \
    --episode-index 0
    # --root $HF_LEROBOT_HOME \