python -m lerobot.scripts.eval \
    --policy.path=outputs/qnbot_cherry_transfer_pi0/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda