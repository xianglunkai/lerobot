#!/bin/bash
# 增强错误处理和日志输出
set -euo pipefail
exec > >(tee -a setup.log) 2>&1

echo "===== 开始配置容器环境 ====="

START_TIME=$(date +%s)

# 1. SSH 服务配置与启动

# docker exec -it iros bash
# sudo vim /etc/ssh/sshd_config
# # 1. 修改端口 Port 9999
# # 2. 允许root登录   PermitRootLogin yes
# # 3. 允许密码登录 PasswordAuthentication yes
# # 4. 重启sshd
# mkdir -p /run/sshd
# ps aux | grep sshd
# kill xxx
# /usr/sbin/sshd -D &


echo "[1/7] 配置 SSH 服务..."
mkdir -p /run/sshd

# 备份原始配置
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak

# 更新配置参数
sed -i 's/^#*Port .*/Port 1234/' /etc/ssh/sshd_config
sed -i 's/^#*PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/^#*PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config

# 重启 SSH 服务
if pgrep sshd >/dev/null; then
    pkill sshd
    echo "已有 SSH 进程已终止"
fi

/usr/sbin/sshd -t && echo "SSH 配置验证通过"
/usr/sbin/sshd -D &
echo "✅ SSH 服务运行中 (端口: 1234)"



# 2. APT 源配置
echo "[2/7] 配置 APT 阿里云源..."
cp /etc/apt/sources.list /etc/apt/sources.list.bak

sed -i 's|http://[^ ]*\.ubuntu\.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list
sed -i 's|http://security\.ubuntu\.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list

echo "✅ APT 源已更新"

# 3. 安装系统依赖
echo "[3/7] 安装系统基础工具..."
apt update && apt install -y --no-install-recommends \
    curl wget vim git tmux htop net-tools openssh-server \
    build-essential ca-certificates \
    > /dev/null

echo "✅ 工具安装完成: curl, wget, vim, tmux..."


# 4. Pip 源配置
echo "[4/7] 配置 Python 环境..."
mkdir -p ~/.pip

# 验证 Pip 配置
echo "当前 Pip 配置:"
pip config list || echo "Pip 配置验证失败"


# 全局配置 pip
cat > /etc/pip.conf <<EOF
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
timeout = 120
EOF

# 当前用户配置
cp /etc/pip.conf ~/.pip/pip.conf

# 验证 Pip 配置
echo "PIP 配置验证:"
pip config list | grep index-url


# 5. 安装 Miniconda
echo "[5/7] 安装 Miniconda..."
MINICONDA_PATH="/worksapce/miniconda3"
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"

# 清除任何已存在的 Miniconda
if [ -d "$MINICONDA_PATH" ]; then
    rm -rf $MINICONDA_PATH
fi

# 下载安装脚本
wget --quiet --show-progress https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER -O /tmp/miniconda.sh

# 执行安装
bash /tmp/miniconda.sh -b -p $MINICONDA_PATH
rm /tmp/miniconda.sh

# 设置环境变量
export PATH="$MINICONDA_PATH/bin:$PATH"
echo "export PATH=\"$MINICONDA_PATH/bin:\$PATH\"" >> ~/.bashrc

# 初始化 Conda
conda init bash
source ~/.bashrc

echo "Miniconda 版本: $(conda --version)"
echo "✅ Miniconda 安装成功"



# 6. 配置 Conda
echo "[6/7] 配置 Conda 环境..."

# 配置国内镜像源
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/r
conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/conda-forge

# 设置默认不激活基础环境
conda config --set auto_activate_base false

# 创建全局配置文件
mkdir -p /etc/conda
cat > /etc/conda/condarc <<EOF
channels:
  - defaults
auto_update_conda: false
channel_alias: https://mirrors.aliyun.com/anaconda
show_channel_urls: true
EOF

# 创建虚拟环境示例
conda create -n iros-env python=3.10 -y
echo "conda activate iros-env" >> ~/.bashrc

echo "✅ Conda 配置完成"
conda config --show channels

# 7. 环境验证
echo "[7/7] 环境验证..."
echo "1. SSH 端口监听:"
netstat -tln | grep ":1234" && echo "✅ SSH 端口正常" || echo "❌ SSH 监听失败"

echo "2. 网络连通性:"
curl -s -m 5 https://mirrors.aliyun.com >/dev/null && echo "✅ 阿里云访问正常" || echo "❌ 网络访问失败"

echo "3. Python 环境:"
python -c "import sys; print(f'Python {sys.version}')" && echo "✅ Python 环境正常" || echo "❌ Python 异常"

echo "4. Conda 功能验证:"
conda list | grep python && echo "✅ Conda 功能正常" || echo "❌ Conda 功能异常"

# 完成
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "========================================"
echo "✅ 容器环境配置完成! (耗时: ${DURATION}秒)"
echo "SSH 访问: ssh -p 1234 root@<容器IP>"
echo "Conda 虚拟环境:"
echo "  $ conda activate iros-env"
echo "========================================"