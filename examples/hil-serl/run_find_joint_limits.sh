export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface

lerobot-find-joint-limits \
  --robot.type=so100_follower \
  --robot.id=cobot_magic \
  --teleop.type=so100_leader \
  --teleop.id=blue