export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
export LD_LIBRARY_PATH="/home/xlk/miniconda3/envs/lerobot/lib:$LD_LRARY_PATH"

python3 -m lerobot.scripts.lerobot_train --config_path /home/xlk/work/lerobot/examples/hil-serl/configs/reward_classifier_train_config.json