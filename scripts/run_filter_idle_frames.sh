export TORCH_NCCL_ENABLE_MONITORING=0  # disable watchdog
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export HF_DATASETS_CACHE=/workspace/huggingface/.cache
export HF_LEROBOT_HOME=/workspace/huggingface/lerobot
export HF_HOME=/workspace/huggingface

export repo_id=screw_sorting_single_v30

python lerobot_analyze_episode_startup.py \
    --dataset screw_sorting_single_v30 \
    --root /workspace/huggingface/lerobot \
    --offline \
    --output-csv /workspace/huggingface/startup_stats.csv


python lerobot_filter_idle_frames.py 