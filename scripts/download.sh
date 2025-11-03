# pip install huggingface-hub
# export HF_ENDPOINT=https://hf-mirror.com #配置镜像站点

# huggingface-cli download lerobot/pi0_base --local-dir ./pretrain_model/pi0_base --local-dir-use-symlinks False   --resume-download 
  
# huggingface-cli download google/paligemma-3b-pt-224 --local-dir ./pretrain_model/paligemma-3b-pt-224 --local-dir-use-symlinks False --resume-download 


pip install modelscope
modelscope download --model AI-ModelScope/paligemma-3b-pt-224 --local_dir ./pretrain_model/paligemma-3b-pt-224