export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
export LD_LIBRARY_PATH="/home/xlk/miniconda3/envs/lerobot/lib:$LD_LRARY_PATH"
export LD_LIBRARY_PATH=/opt/ros/noetic/lib:$LD_LIBRARY_PATH
python  /home/xlk/work/lerobot/examples/hil-serl/find_eef_limit.py --config_path /home/xlk/work/lerobot/examples/hil-serl/configs/env_config_find_eef_limit.json