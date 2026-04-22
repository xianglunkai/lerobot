export HF_DATASETS_CACHE=/workspace/huggingface/.cache
export HF_LEROBOT_HOME=/workspace/huggingface/lerobot
export HF_HOME=/workspace/huggingface


export repo_id=take_me_tissues_v30_new

# S1. Remove specific episodes from a dataset. This is useful for filtering out undesired data.
# Delete episodes and modifies original dataset
# lerobot-edit-dataset \
#     --repo_id ${HF_LEROBOT_HOME}/${repo_id} \
#     --operation.type delete_episodes \
#     --operation.episode_indices "[28]"

# Delete episodes and save to a new dataset (preserves original dataset)
lerobot-edit-dataset \
    --repo_id ${HF_LEROBOT_HOME}/${repo_id} \
    --new_repo_id ${HF_LEROBOT_HOME}/${repo_id}_after_deletion \
    --operation.type delete_episodes \
    --operation.episode_indices "[28]" \
    --push_to_hub true

# S2. Combine multiple datasets into a single dataset.
# Merge train and validation splits back into one dataset
# lerobot-edit-dataset \
#     --repo_id ${HF_LEROBOT_HOME}/${repo_id}_merged \
#     --operation.type merge \
#     --operation.repo_ids "['${HF_LEROBOT_HOME}/${repo_id}_train', '${HF_LEROBOT_HOME}/${repo_id}_val']"


# S3. Remove Features
# Remove a camera feature
# lerobot-edit-dataset \
#     --repo_id ${HF_LEROBOT_HOME}/${repo_id}  \
#     --operation.type remove_feature \
#     --operation.feature_names "['observation.images.top']"


# S4.Show the information of datasets

# Show dataset information without feature details
lerobot-edit-dataset \
    --repo_id ${HF_LEROBOT_HOME}/${repo_id} \
    --operation.type info \

# Show dataset information with feature details
lerobot-edit-dataset \
    --repo_id ${HF_LEROBOT_HOME}/${repo_id} \
    --operation.type info \
    --operation.show_features true


    