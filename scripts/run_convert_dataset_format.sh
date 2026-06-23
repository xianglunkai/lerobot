
#!/usr/bin/env bash

export TORCH_NCCL_ENABLE_MONITORING=0  # disable watchdog
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export HF_DATASETS_CACHE=/workspace/huggingface/.cache
export HF_LEROBOT_HOME=/workspace/huggingface/lerobot
export HF_HOME=/workspace/huggingface

export repo_id=screw_sorting_single_v30

# python hdf5_to_lerobot_dobot_v3.py \
#   --input_dir ../datasets/fold_towel/ \
#   --repo_id lerobot/fold_towel_v3_0 \
#   --root ../huggingface/lerobot/fold_towel_v3_0

python convert_dataset_v30_to_v21.py \
    --repo-id ${repo_id}  \
    --root /workspace/huggingface/lerobot \
    --offline \
    --embed-images


# python convert_dataset_v21_to_v30.py \
#   --repo-id take_me_tissues \
#   --root /workspace/huggingface/lerobot \
#   --push-to-hub=false \
#   --images-to-videos



# python convert_dataset_v20_to_v21.py \
#   --repo-id handover_bottle_action_from_slave_2_0 \
#   --root /workspace/huggingface/lerobot \
#   --offline --no-push 

