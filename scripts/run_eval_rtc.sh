export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface
export repo_id=eval_lerobot_fold_towel_20260116_110149  # lerobot_fold_clothes_20260204_154839


read -p "Please input model name (pi05 or smolvla): " SELECT_MODEL
SELECT_MODEL=$(echo "$SELECT_MODEL" | tr '[:upper:]' '[:lower:]')

if [ -z "$SELECT_MODEL" ]; then
    echo "未输入模型名称，使用默认值 pi05"
    SELECT_MODEL="pi05"
fi

echo "Selected model: $SELECT_MODEL"

case "$SELECT_MODEL" in
    pi05)
        python examples/rtc/eval_dataset.py \
            --policy.path=/home/xlk/work/lerobot/checkpoints/rabc_sarm_pi05/040000/pretrained_model \
            --dataset.repo_id="$repo_id" \
            --rtc.enabled=True \
            --use_ccr=False \
            --rtc.execution_horizon=25 \
            --rtc.max_guidance_weight=10.0 \
            --rtc.prefix_attention_schedule=EXP \
            --rtc.sigma_d=1.0 \
            --inference_delay=8 \
            --num_inference_steps=10 \
            --device=cuda \
            --use_torch_compile=False \
            --next_inference_after=25
        ;;
    smolvla)
        python examples/rtc/eval_dataset.py \
            --policy.path=/home/xlk/work/lerobot/checkpoints/smolval_traiining_rtc_fold_towel_v3_0/checkpoints/020000/pretrained_model \
            --dataset.repo_id="$repo_id" \
            --rtc.enabled=False \
            --use_ccr=True \
            --rtc.execution_horizon=25 \
            --rtc.max_guidance_weight=10.0 \
            --rtc.prefix_attention_schedule=EXP \
            --rtc.sigma_d=0.2 \
            --inference_delay=8 \
            --num_inference_steps=10 \
            --device=cuda \
            --use_torch_compile=False \
            --next_inference_after=25
        ;;
        *)
        echo "错误：不支持的模型名称 '$SELECT_MODEL'，请使用 pi05 或 smolvla。"
        exit 1
        ;;
esac
   
