export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface

rm -rf $HF_LEROBOT_HOME/gym_hil/sim_dataset
python -m lerobot.rl.gym_manipulator --config_path /home/xlk/work/lerobot/examples/hil-serl/configs/gym_hil_env.json