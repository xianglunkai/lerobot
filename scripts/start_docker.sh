#!/bin/bash

# 设置错误处理：任何命令失败即停止执行
set -e

echo "启动容器 iros-agibot..."

docker run \
    -itd --shm-size=256g \
    --device nvidia.com/gpu=all \
    --name iros-agibot \
    --hostname iros-agibot-docker \
    -p 1234:1234 \
    -v /mnt2/personal/iros:/workspace \
    -v /usr/local/cuda-12.4:/usr/local/cuda \
    localhost/pytorch_2.6.0_cu124:v0.1.0 bash 


# 检查容器状态
container_id=$(docker ps -q -f name=iros-agibot)
if [ -z "$container_id" ]; then
    echo "❌ 错误：容器启动失败！"
    docker ps -a | grep iros-agibot
    exit 1
fi
echo "✅ 容器成功启动 ID: $container_id"

# 等待容器初始化
echo "等待容器准备就绪..."
sleep 3


# 安装 flash-attn 并添加详细日志
echo "开始安装 flash-attn==2.5.5..."
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="flash_attn_install_$timestamp.log"

if docker exec -i iros-agibot bash -c '\
    set -e;
    echo -e "\n### 开始安装 $(date) ###";
    pip install "flash-attn==2.5.5" --no-build-isolation --verbose;
    echo -e "\n### 安装完成！验证中... ###";
    python -c "import flash_attn; print(\"✅ 成功导入 FlashAttention v\"+flash_attn.__version__)"' | tee $log_file
then
    echo "✅ flash-attn 安装验证成功！日志保存到 $log_file"
else
    exit_code=$?
    echo "❌ 安装失败！错误代码: $exit_code, 完整日志见 $log_file"
    echo "容器日志:"
    docker logs iros-agibot | tail -n 20
    exit $exit_code
fi

echo "🟢 所有操作成功完成！容器准备就绪"
docker exec -it iros-agibot bash