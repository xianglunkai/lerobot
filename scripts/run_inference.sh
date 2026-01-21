
    
export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface

python3 -m lerobot.scripts.lerobot_record \
    --robot.type=agilex_cobot \
    --robot.id=lerobot_fold_towel \
    --robot.cameras='{
        "high": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
        "left": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
        "right": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
    }' \
    --policy.path=/home/xlk/work/lerobot/checkpoints/fold_towel/030000/pretrained_model \
    --dataset.repo_id=lerobot/eval_lerobot_fold_towel_$(date +%Y%m%d_%H%M%S) \
    --dataset.single_task="Carefully fold the towel and then place the folded towel on the black notebook" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=60 \
    --dataset.fps=30 \
    --dataset.video=True \
    --dataset.push_to_hub=false \
    --display_data=true


# python3 -m lerobot.scripts.lerobot_record \
#     --robot.type=agilex_cobot \
#     --robot.id=lerobot_fold_towel \
#     --robot.cameras='{
#         "high": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
#         "left": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
#         "right": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
#     }' \
#     --policy.path=/home/xlk/work/lerobot/checkpoints/fold_towel/30k-50hz/pretrained_model \
#     --dataset.repo_id=lerobot/eval_lerobot_fold_towel_$(date +%Y%m%d_%H%M%S) \
#     --dataset.single_task="Carefully fold the towel and then place the folded towel on the black notebook" \
#     --dataset.num_episodes=1 \
#     --dataset.episode_time_s=100 \
#     --dataset.fps=50 \
#     --dataset.video=True \
#     --dataset.push_to_hub=false \
#     --display_data=True


# python3 -m lerobot.scripts.lerobot_record \
#     --robot.type=agilex_cobot \
#     --robot.id=lerobot_hover_bottle \
#     --robot.cameras='{
#         "high": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
#         "left": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
#         "right": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
#     }' \
#     --policy.path=/home/xlk/work/lerobot/checkpoints/wall-x/030000/pretrained_model \
#     --dataset.repo_id=lerobot/eval_lerobot_hover_bottle_$(date +%Y%m%d_%H%M%S) \
#     --dataset.single_task="Use the one arm to grasp the bottle on the table, handover it to the another arm and place it on the black book" \
#     --dataset.num_episodes=1 \
#     --dataset.episode_time_s=100 \
#     --dataset.fps=30 \
#     --dataset.video=True \
#     --dataset.push_to_hub=false \
#     --display_data=True
