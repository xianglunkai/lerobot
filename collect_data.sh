source /opt/ros/humble/setup.bash

export ROS_DOMAIN_ID=101 #设置ROS ID，可以使得主机和机器人上的小电脑上的ROS之间进行通信


python3 -m lerobot.record \
    --robot.type=qnbot_w \
    --robot.id=qnbot_w_recording \
    --robot.cameras='{"left_wrist": {"type": "webrtc", "server_url": "http://10.161.27.185:8080", "camera_name": "left_wrist", "fps": 30, "width": 640, "height": 480, "color_mode": "rgb"}, "head": {"type": "webrtc", "server_url": "http://10.161.27.185:8080", "camera_name": "head", "fps": 30, "width": 640, "height": 480, "color_mode": "rgb"}, "right_wrist": {"type": "webrtc", "server_url": "http://10.161.27.185:8080", "camera_name": "right_wrist", "fps": 30, "width": 640, "height": 480, "color_mode": "rgb"}}' \
    --dataset.repo_id=bradley/qnbot_w_10s_30_$(date +%Y%m%d_%H%M%S) \
    --dataset.single_task="the right hand picks up the cherry toy and passes it to the left hand to place in the white plate" \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=60 \
    --dataset.fps=30 \
    --dataset.video=true \
    --dataset.push_to_hub=false \
    --display_data=true

