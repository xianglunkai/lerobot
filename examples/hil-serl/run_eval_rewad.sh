export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
export LD_LIBRARY_PATH="/home/xlk/miniconda3/envs/lerobot/lib:$LD_LRARY_PATH"
export LD_LIBRARY_PATH=/opt/ros/noetic/lib:$LD_LIBRARY_PATH
python -m lerobot.rl.gym_manipulator --config_path /home/xlk/work/lerobot/examples/hil-serl/configs/eval_config_reward_cobotmagic.json