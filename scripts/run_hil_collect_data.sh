export ROS_HOSTNAME=192.168.1.163
export ROS_MASTER_URI=http://192.168.1.139:11311

export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface

export LD_LIBRARY_PATH="/home/xlk/miniconda3/envs/lerobot/lib:$LD_LRARY_PATH"
export LD_LIBRARY_PATH=/opt/ros/noetic/lib:$LD_LIBRARY_PATH

    
read -p "Please teleop type (s or g): " SELECT
SELECT=$(echo "$SELECT" | tr '[:upper:]' '[:lower:]')

if [ -z "$SELECT" ]; then
    echo "未输入模型名称，使用默认值 spacemouse"
    SELECT="s"
fi

echo "Selected model: $SELECT"

# RTC with and pi0.5 policy

case "$SELECT" in
    s)
        python examples/rac/hil_data_collection.py \
            --policy.path=/home/xlk/work/lerobot/checkpoints/pi06_policy_screw_sorting_single_v30/checkpoints/030000/pretrained_model \
            --robot.type=agilex_cobot \
            --robot.use_external_commands=false \
            --robot.ros_config.with_mobile_base=false \
            --robot.ros_config.with_l_arm=False \
            --robot.ros_config.with_r_arm=True \
            --robot.ros_config.with_left_camera=False \
            --teleop.type=spacemouse \
            --dataset.repo_id=lerobot-data-collection/hil_screw_sorting_$(date +%Y%m%d_%H%M%S) \
            --dataset.single_task="Please sort and return the silver screws in the grey box to their proper places." \
            --dataset.fps=30 \
            --dataset.video=True \
            --dataset.episode_time_s=240 \
            --dataset.push_to_hub=false \
            --dataset.num_episodes=50 \
            --dataset.vcodec=libsvtav1 \
            --dataset.streaming_encoding=true \
            --display_data=false \
            --rtc.enabled=true \
            --rtc.execution_horizon=25 \
            --rtc.max_guidance_weight=10.0 \
            --rtc.prefix_attention_schedule=EXP \
            --rtc.sigma_d=0.2 \
            --interpolation_multiplier=1 \
            --calibrate=true \
            --device=cuda \
            --action_queue_size_to_get_new_actions=25 \
            --enable_episode_outcome_labeling=true \
            --default_episode_success="success" \
            --require_episode_success_label=true \
            --acp_inference.enable=false \
        
        ;;

    g)
        python examples/rac/hil_data_collection.py \
            --policy.path=/home/xlk/work/lerobot/checkpoints/pi05_delta_act_screw_sorting/020000/pretrained_model \
            --robot.type=agilex_cobot \
            --robot.use_external_commands=false \
            --robot.ros_config.with_mobile_base=false \
            --robot.ros_config.with_l_arm=False \
            --robot.ros_config.with_r_arm=True \
            --robot.ros_config.with_left_camera=False \
            --teleop.type=gamepad \
            --dataset.repo_id=lerobot-data-collection/hil_screw_sorting_$(date +%Y%m%d_%H%M%S) \
            --dataset.single_task="Please sort and return the silver screws in the grey box to their proper places." \
            --dataset.fps=30 \
            --dataset.video=True \
            --dataset.episode_time_s=240 \
            --dataset.push_to_hub=false \
            --dataset.num_episodes=10 \
            --dataset.vcodec=libsvtav1 \
            --dataset.streaming_encoding=false \
            --display_data=false \
            --rtc.enabled=true \
            --rtc.execution_horizon=16 \
            --rtc.max_guidance_weight=10.0 \
            --rtc.prefix_attention_schedule=EXP \
            --rtc.sigma_d=0.2 \
            --interpolation_multiplier=1 \
            --calibrate=true \
            --device=cuda \
            --action_queue_size_to_get_new_actions=30 \
        ;;
        *)
        echo "错误：不支持的模型名称 '$SELECT'，请使用 s or g"
        exit 1
        ;;
esac