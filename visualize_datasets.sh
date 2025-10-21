export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface

# 设置环境变量使用 decord 后端
# export LEROBOT_VIDEO_BACKEND=decord

lerobot-dataset-viz \
    --repo-id lerobot/lerobot_adjust_bottle_action_from_slave \
    --mode "local" \
    --episode-index 0
    # --root $HF_LEROBOT_HOME \