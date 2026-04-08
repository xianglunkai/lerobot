export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface

python3 -m lerobot.scripts.lerobot_record \
    --robot.type=agilex_cobot \
    --robot.id=lerobot_fold_towel \
    --robot.use_external_commands=False \
    --policy.device=cuda \
    --policy.path=/home/xlk/work/lerobot/checkpoints/pi05_delta_act_fold_towel/030000/pretrained_model \
    --dataset.repo_id=lerobot/eval_lerobot_fold_towel_$(date +%Y%m%d_%H%M%S) \
    --dataset.single_task="Carefully fold the towel and then place the folded towel on the black notebook" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=60 \
    --dataset.fps=30 \
    --dataset.video=True \
    --dataset.push_to_hub=false \
    --display_data=true