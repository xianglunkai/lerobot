# source /opt/ros/humble/setup.bash

# export ROS_DOMAIN_ID=101 #设置ROS ID，可以使得主机和机器人上的小电脑上的ROS之间进行通信


# python3 -m lerobot.record \
#     --robot.type=qnbot_w \
#     --robot.id=qnbot_w_recording \
#     --robot.cameras='{"left_wrist": {"type": "webrtc", "server_url": "http://10.161.27.185:8080", "camera_name": "left_wrist", "fps": 30, "width": 640, "height": 480, "color_mode": "rgb"},
#      "head": {"type": "webrtc", "server_url": "http://10.161.27.185:8080", "camera_name": "head", "fps": 30, "width": 640, "height": 480, "color_mode": "rgb"}, 
#      "right_wrist": {"type": "webrtc", "server_url": "http://10.161.27.185:8080", "camera_name": "right_wrist", "fps": 30, "width": 640, "height": 480, "color_mode": "rgb"}}' \
#     --dataset.repo_id=bradley/qnbot_w_10s_30_$(date +%Y%m%d_%H%M%S) \
#     --dataset.single_task="the right hand picks up the cherry toy and passes it to the left hand to place in the white plate" \
#     --dataset.num_episodes=3 \
#     --dataset.episode_time_s=60 \
#     --dataset.fps=30 \
#     --dataset.video=true \
#     --dataset.push_to_hub=false \
#     --display_data=true


export ROS_HOSTNAME=192.168.1.163
export ROS_MASTER_URI=http://192.168.1.139:11311

export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface


# python3 -m lerobot.scripts.lerobot_record \
#     --robot.type=cobot_magic \
#     --robot.id=lerobot_hover_bottle_action_from_slave \
#     --robot.cameras='{
#         "high": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
#         "left": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
#         "right": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
#     }' \
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