
rm -rf outputs/qnbot_cherry_transfer_pi0

python3 -m lerobot.scripts.train \
    --dataset.root=./datasets/qnbot_data1 \
    --dataset.repo_id=bradley/qnbot_cherry_transfer_20250705 \
    --policy.type=pi0 \
    --policy.repo_id=lerobot/pi0 \
    --policy.train_expert_only=true \
    --output_dir=outputs/qnbot_cherry_transfer_pi0 \
    --batch_size=4 \
    --steps=10000 \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=lerobot_qnbot \
    --wandb.entity=breadlee1024 \
    --wandb.notes="QnBot樱桃传递任务 - 右手拿樱桃玩具传递给左手放到白盘子里" \
    --log_freq=200 \
    --save_freq=10000