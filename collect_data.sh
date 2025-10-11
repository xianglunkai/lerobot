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


python3 -m lerobot.scripts.lerobot_record \
    --robot.type=cobot_magic \
    --robot.id=pour_water \
    --robot.cameras='{
        "high": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
        "left": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
        "right": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
    }' \
    --dataset.repo_id=lerobot/pour_water_$(date +%Y%m%d_%H%M%S) \
    --dataset.single_task="pour water from the bottle into the cup until it is one-third full, then put it down" \
    --dataset.num_episodes=15 \
    --dataset.episode_time_s=40 \
    --dataset.fps=30 \
    --dataset.video=True \
    --dataset.push_to_hub=false \
    --display_data=true