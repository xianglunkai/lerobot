
#!/usr/bin/env bash

export TORCH_NCCL_ENABLE_MONITORING=0  # disable watchdog
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# 环境变量: 指定 HuggingFace 数据集本地缓存（离线模式需要）
export HF_DATASETS_CACHE=/workspace/huggingface/lerobot

python src/lerobot/datasets/v30/convert_dataset_v30_to_v21.py \
    --repo-id lerobot/handover_bottle \
    --root ../huggingface/ \
    --offline \
    --embed-images

python src/lerobot/datasets/v30/hdf5_to_lerobot_dobot_v3.py \
  --input_dir ../datasets/fold_towel/ \
  --repo_id lerobot/fold_towel_v3_0 \
  --root ../huggingface/lerobot/fold_towel_v3_0


python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id fold_towel \
  --root /workspace/huggingface/lerobot \
  --push-to-hub=false



