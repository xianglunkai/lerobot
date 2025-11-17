export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
export LD_LIBRARY_PATH="/home/xlk/miniconda3/envs/lerobot/lib:$LD_LRARY_PATH"
export LD_LIBRARY_PATH=/opt/ros/noetic/lib:$LD_LIBRARY_PATH

# rm -rf $HF_LEROBOT_HOME/pick_tool_into_box
python -m lerobot.rl.gym_manipulator --config_path /home/xlk/work/lerobot/examples/hil-serl/configs/env_config_cobotmagic.json