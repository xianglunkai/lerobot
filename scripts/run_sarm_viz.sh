
# 设置 Hugging Face 镜像端点
export HF_ENDPOINT=https://hf-mirror.com

# 设置可见的 GPU（根据实际 GPU 数量修改）
# export CUDA_VISIBLE_DEVICES=0
export TORCH_NCCL_ENABLE_MONITORING=0  # disable watchdog
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export HF_DATASETS_CACHE=/workspace/huggingface/.cache
export HF_LEROBOT_HOME=/workspace/huggingface/lerobot
export HF_HOME=/workspace/huggingface

export repo_id=screw_sorting_v30

python -m lerobot.policies.sarm.compute_rabc_weights \
  --dataset-repo-id ${HF_LEROBOT_HOME}/${repo_id}  \
  --reward-model-path /workspace/lerobot/outputs/sarm_single_screw_sorting_v30/checkpoints/005000/pretrained_model \
  --visualize-only \
  --num-visualizations 5 \
  --head-mode sparse \
  --output-dir ./sarm_viz
