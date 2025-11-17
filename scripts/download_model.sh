# pip install modelscope

# modelscope download --model AI-ModelScope/paligemma-3b-pt-224 --local_dir ./pretrain_model/paligemma-3b-pt-224

# modelscope download --model lerobot/pi05_base --local_dir ./pretrain_model/pi05_base

export HF_ENDPOINT=https://hf-mirror.com #配置镜像站点
huggingface-cli download helper2424/resnet10 --local-dir /home/xlk/work/lerobot/pretrain_model/resnet10 --local-dir-use-symlinks False   --resume-download 

huggingface-cli download microsoft/resnet-18 --local-dir /home/xlk/work/lerobot/pretrain_model/resnet-18 --local-dir-use-symlinks False   --resume-download 

