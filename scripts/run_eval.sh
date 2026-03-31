lerobot-eval \
    --policy.path=/home/xlk/work/lerobot/checkpoints/fold_towel/30k-50hz/pretrained_model \
    --env.type=cobot \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda