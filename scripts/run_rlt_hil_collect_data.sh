#!/usr/bin/env bash
# RLT-schema AgileX HIL collection (does not replace scripts/run_hil_collect_data.sh).
set -euo pipefail

export ROS_HOSTNAME="${ROS_HOSTNAME:-192.168.1.163}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://192.168.1.139:11311}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/huggingface/lerobot}"
export HF_HOME="${HF_HOME:-/data/huggingface}"

export LD_LIBRARY_PATH="/home/xlk/miniconda3/envs/lerobot/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ros/noetic/lib:${LD_LIBRARY_PATH}"

POLICY_PATH="${POLICY_PATH:-/home/xlk/work/lerobot/checkpoints/pi06_policy_screw_sorting_single_v30/checkpoints/030000/pretrained_model}"
TASK="${TASK:-Please sort and return the silver screws in the grey box to their proper places.}"

read -r -p "Please teleop type (s=spacemouse or g=gamepad): " SELECT
SELECT=$(echo "$SELECT" | tr '[:upper:]' '[:lower:]')
if [ -z "$SELECT" ]; then
    SELECT="s"
fi

COMMON_ARGS=(
    --policy.path="${POLICY_PATH}"
    --robot.type=agilex_cobot
    --robot.use_external_commands=false
    --robot.ros_config.with_mobile_base=false
    --robot.ros_config.with_l_arm=False
    --robot.ros_config.with_r_arm=True
    --robot.ros_config.with_left_camera=False
    --dataset.repo_id="lerobot-data-collection/rlt_hil_screw_sorting_$(date +%Y%m%d_%H%M%S)"
    --dataset.single_task="${TASK}"
    --dataset.fps=30
    --dataset.video=True
    --dataset.episode_time_s=240
    --dataset.push_to_hub=false
    --dataset.num_episodes=60
    --dataset.vcodec=libsvtav1
    --dataset.streaming_encoding=true
    --display_data=false
    --rtc.enabled=true
    --rtc.execution_horizon=20
    --rtc.max_guidance_weight=10.0
    --rtc.prefix_attention_schedule=EXP
    --rtc.sigma_d=0.2
    --interpolation_multiplier=1
    --calibrate=true
    --device=cuda
    --action_queue_size_to_get_new_actions=30
    --enable_episode_outcome_labeling=true
    --default_episode_success=success
    --require_episode_success_label=true
    --enable_pedal_outcome=true
    --teleop_toggle_key=space
    --rlt_toggle_key=r
    --double_tap_window_s=0.6
    --only_critical=false
    --start_with_teleop=false
    --acp_inference.enable=false
)

case "$SELECT" in
    s)
        python examples/rac/rlt_hil_data_collection.py \
            "${COMMON_ARGS[@]}" \
            --teleop.type=spacemouse
        ;;
    g)
        python examples/rac/rlt_hil_data_collection.py \
            "${COMMON_ARGS[@]}" \
            --teleop.type=gamepad \
            --dataset.num_episodes=10 \
            --dataset.streaming_encoding=false \
            --rtc.execution_horizon=16
        ;;
    *)
        echo "错误：不支持的 teleop '$SELECT'，请使用 s 或 g"
        exit 1
        ;;
esac
