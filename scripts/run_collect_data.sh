# source /opt/ros/humble/setup.bash

# export ROS_DOMAIN_ID=101 #设置ROS ID，可以使得主机和机器人上的小电脑上的ROS之间进行通信

export ROS_HOSTNAME=192.168.1.163
export ROS_MASTER_URI=http://192.168.1.139:11311

export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface



# python3 -m lerobot.scripts.lerobot_record \
#     --robot.type=agilex_cobot \
#     --robot.id=lerobot_navigation \
#     --robot.use_external_commands=true \
#     --robot.ros_config.with_mobile_base=true \
#     --teleop.type=agilex_cobot_teleop \
#     --teleop.use_present_position=false \
#     --teleop.use_eef_pose_action=false \
#     --dataset.repo_id=lerobot/lerobot_hover_bottle_action_from_slave_$(date +%Y%m%d_%H%M%S) \
#     --dataset.single_task="use the one arm to grasp the bottle on the table, handover it to the another arm and place it on the black book" \
#     --dataset.num_episodes=30 \
#     --dataset.fps=50 \
#     --dataset.video=True \
#     --dataset.push_to_hub=false \
#     --display_data=true \
#     --dataset.episode_time_s=60 \
#     --dataset.reset_time_s=60 \


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


python3 -m lerobot.scripts.lerobot_record \
    --robot.type=agilex_cobot \
    --robot.id=fold_cloth \
    --robot.use_external_commands=true \
    --robot.ros_config.with_mobile_base=false \
    --dataset.repo_id=lerobot/lerobot_fold_cloth_action_from_slave_$(date +%Y%m%d_%H%M%S) \
    --dataset.single_task="Please fold the clothes on the desktop!" \
    --dataset.num_episodes=30 \
    --dataset.fps=30 \
    --dataset.video=True \
    --dataset.push_to_hub=false \
    --display_data=true \
    --dataset.episode_time_s=180 \
    --dataset.reset_time_s=15 \