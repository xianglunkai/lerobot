import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig

import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_LEROBOT_HOME'] = '/data/huggingface/lerobot'
os.environ['HF_HOME'] = '/data/huggingface'

print("HF_ENDPOINT:", os.environ.get('HF_ENDPOINT'))
print("HF_LEROBOT_HOME:", os.environ.get('HF_LEROBOT_HOME'))
print("HF_HOME:", os.environ.get('HF_HOME'))

# Device to use for training
device = "cuda"  # or "cuda", or "cpu"

repo_id = "pick_place_task_cropped_resized"
dataset = LeRobotDataset(repo_id, root="/data/huggingface/lerobot/pick_place_task_cropped_resized")

# Configure the policy to extract features from the image frames
camera_keys = dataset.meta.camera_keys

config = RewardClassifierConfig(
    num_cameras=len(camera_keys),
    device=device,
    # backbone model to extract features from the image frames
    model_name="/home/xlk/work/lerobot/pretrain_model/resnet10",
)

# Make policy, preprocessor, and optimizer
policy = make_policy(config, ds_meta=dataset.meta)
optimizer = config.get_optimizer_preset().build(policy.parameters())
preprocessor, _ = make_pre_post_processors(policy_cfg=config, dataset_stats=dataset.meta.stats)

# Instantiate a dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    total_accuracy = 0
    for batch in dataloader:
        # Preprocess the batch and move it to the correct device.
        batch = preprocessor(batch)

        # Forward pass
        loss, output_dict = policy.forward(batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += output_dict["accuracy"]

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")

print("Training finished!")

# 在训练结束后添加
local_save_path = "/home/xlk/work/lerobot/outputs/train/reward_classifier"

# 保存到本地指定路径
policy.save_pretrained(local_save_path)
print(f"模型已保存到: {local_save_path}")

