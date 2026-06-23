export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface


export LD_LIBRARY_PATH="/home/xlk/miniconda3/envs/lerobot/lib:$LD_LRARY_PATH"
export LD_LIBRARY_PATH=/opt/ros/noetic/lib:$LD_LIBRARY_PATH


# delta action for folding clothes
# python examples/rtc/eval_with_real_robot.py \
#     --policy.path=/home/xlk/work/lerobot/checkpoints/pi05_delta_act_fold_clothes40_v30/checkpoints/015000/pretrained_model  \
#     --policy.device=cuda \
#     --robot.type=agilex_cobot \
#     --rtc.enabled=True \
#     --rtc.execution_horizon=25 \
#     --rtc.max_guidance_weight=10.0 \
#     --rtc.prefix_attention_schedule=EXP \
#     --rtc.sigma_d=0.2 \
#     --task="Carefully fold the clothes." \
#     --duration=120 \
#     --fps=30 \
#     --device=cuda \
#     --action_queue_size_to_get_new_actions=25 \
#     --use_torch_compile=False \
#     --enable_visualization=true \
#     --interpolation_multiplier=2 \

# delta action for folding towel
# python examples/rtc/eval_with_real_robot.py \
#     --policy.path=/home/xlk/work/lerobot/checkpoints/pi05_delta_act_fold_towel/030000/pretrained_model  \
#     --policy.device=cuda \
#     --robot.type=agilex_cobot \
#     --rtc.enabled=True \
#     --rtc.execution_horizon=20 \
#     --rtc.max_guidance_weight=10.0 \
#     --rtc.prefix_attention_schedule=EXP \
#     --rtc.sigma_d=0.2 \
#     --task="Carefully fold the towel and then place the folded towel on the black notebook" \
#     --duration=60 \
#     --fps=30 \
#     --device=cuda \
#     --action_queue_size_to_get_new_actions=30 \
#     --use_torch_compile=False \
#     --enable_visualization=true \
#     --interpolation_multiplier=2 \


# training rtc test
python examples/rtc/eval_with_real_robot.py \
    --policy.path=/home/xlk/work/lerobot/checkpoints/pi06_policy_screw_sorting_single_v30/checkpoints/030000/pretrained_model  \
    --policy.device=cuda \
    --robot.type=agilex_cobot \
    --robot.use_external_commands=false \
    --robot.ros_config.with_mobile_base=false \
    --robot.ros_config.with_l_arm=False \
    --robot.ros_config.with_r_arm=True \
    --robot.ros_config.with_left_camera=False \
    --rtc.enabled=True \
    --rtc.execution_horizon=25 \
    --rtc.max_guidance_weight=10.0 \
    --rtc.prefix_attention_schedule=EXP \
    --rtc.sigma_d=0.2 \
    --task="Please sort and return the silver screws in the grey box to their proper places." \
    --duration=120 \
    --fps=30 \
    --device=cuda \
    --action_queue_size_to_get_new_actions=25 \
    --use_torch_compile=False \
    --enable_visualization=true \
    --interpolation_multiplier=2 \