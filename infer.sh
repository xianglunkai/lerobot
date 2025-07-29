
    python3 -m lerobot.record \
    --robot.type=qnbot_w \
    --robot.id=qnbot_w_recording \
    --robot.cameras='{"left_wrist": {"type": "webrtc", "server_url": "http://10.161.27.185:8080", "camera_name": "left_wrist", "fps": 30, "width": 640, "height": 480, "color_mode": "rgb"}, "head": {"type": "webrtc", "server_url": "http://10.161.27.185:8080", "camera_name": "head", "fps": 30, "width": 640, "height": 480, "color_mode": "rgb"}, "right_wrist": {"type": "webrtc", "server_url": "http://10.161.27.185:8080", "camera_name": "right_wrist", "fps": 30, "width": 640, "height": 480, "color_mode": "rgb"}}' \
    --policy.path=outputs/080000/pretrained_model \
    --dataset.repo_id=bradley/eval_qnbot_w_inference_$(date +%Y%m%d_%H%M%S) \
    --dataset.single_task="the right hand picks up the cherry toy and passes it to the left hand to place in the white plate" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=3600 \
    --dataset.fps=30 \
    --dataset.video=true \
    --dataset.push_to_hub=false \
    --display_data=true

