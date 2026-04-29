# source /opt/ros/humble/setup.bash

# export ROS_DOMAIN_ID=101 #设置ROS ID，可以使得主机和机器人上的小电脑上的ROS之间进行通信

export ROS_HOSTNAME=192.168.1.163
export ROS_MASTER_URI=http://192.168.1.139:11311

export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface

export LD_LIBRARY_PATH="/home/xlk/miniconda3/envs/lerobot/lib:$LD_LRARY_PATH"
export LD_LIBRARY_PATH=/opt/ros/noetic/lib:$LD_LIBRARY_PATH

# python3 -m lerobot.scripts.lerobot_record \
#     --robot.type=agilex_cobot \
#     --robot.id=lerobot_navigation \
#     --robot.use_external_commands=true \
#     --robot.ros_config.with_mobile_base=true \
#     --dataset.repo_id=lerobot/lerobot_navigation$(date +%Y%m%d_%H%M%S) \
#     --dataset.single_task="" \
#     --dataset.num_episodes=50 \
#     --dataset.fps=30 \
#     --dataset.video=True \
#     --dataset.push_to_hub=false \
#     --display_data=true \
#     --dataset.episode_time_s=60 \
#     --dataset.reset_time_s=60 \


# python3 -m lerobot.scripts.lerobot_record \
#     --robot.type=agilex_cobot \
#     --robot.id=fold_cloth \
#     --robot.use_external_commands=true \
#     --robot.ros_config.with_mobile_base=false \
#     --dataset.repo_id=lerobot/lerobot_fold_cloth_action_from_slave_$(date +%Y%m%d_%H%M%S) \
#     --dataset.single_task="Please fold the clothes on the desktop!" \
#     --dataset.num_episodes=30 \
#     --dataset.fps=30 \
#     --dataset.video=True \
#     --dataset.push_to_hub=false \
#     --display_data=true \
#     --dataset.episode_time_s=180 \
#     --dataset.reset_time_s=15 \



python3 -m lerobot.scripts.lerobot_record \
    --robot.type=agilex_cobot \
    --robot.id=screw_sorting \
    --robot.use_external_commands=false \
    --robot.ros_config.with_mobile_base=false \
    --robot.ros_config.with_l_arm=False \
    --robot.ros_config.with_r_arm=True \
    --robot.ros_config.with_left_camera=False \
    --teleop.type=spacemouse \
    --dataset.repo_id=lerobot/screw_sorting_$(date +%Y%m%d_%H%M%S) \
    --dataset.single_task="Please sort and return the silver screws in the grey box to their proper places." \
    --dataset.num_episodes=30 \
    --dataset.fps=30 \
    --dataset.video=True \
    --dataset.push_to_hub=false \
    --display_data=true \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \

# python3 -m lerobot.scripts.lerobot_record \
#     --robot.type=agilex_cobot \
#     --robot.id=take_me_tissues_v30 \
#     --robot.use_external_commands=true \
#     --robot.ros_config.with_mobile_base=true \
#     --robot.ros_config.with_l_arm=true \
#     --robot.ros_config.with_r_arm=True \
#     --dataset.repo_id=lerobot/take_me_tissues_v30_$(date +%Y%m%d_%H%M%S) \
#     --dataset.single_task="Please take a pack of tissues from the drawer next to you, then pull out one sheet and give it to me." \
#     --dataset.num_episodes=30 \
#     --dataset.fps=30 \
#     --dataset.video=True \
#     --dataset.push_to_hub=false \
#     --display_data=true \
#     --dataset.episode_time_s=180 \
#     --dataset.reset_time_s=30 \

