#!/usr/bin/env bash
# One-click ROS1 + ZMQ bridge on the AgileX robot (Python 3.8 / Noetic).
# Usage:
#   bash start_bridge.sh
#   bash start_bridge.sh --bind-host=0.0.0.0 --with-mobile-base
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- machine defaults (override before calling, e.g. ROS_IP=... bash start_bridge.sh) ----
ROS_DISTRO_SETUP="${ROS_DISTRO_SETUP:-/opt/ros/noetic/setup.bash}"
CONDA_ENV="${CONDA_ENV:-aloha}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

AGILEX_WS="${AGILEX_WS:-$HOME/agilex_ws/devel/setup.bash}"
CAMERA_WS="${CAMERA_WS:-$HOME/cobot_magic/camera_ws/devel/setup.bash}"
PIPER_WS="${PIPER_WS:-$HOME/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash}"

# Local roscore on this robot. ZMQ bind is separate (--bind-host).
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
export ROS_IP="${ROS_IP:-192.168.1.139}"
unset ROS_HOSTNAME 2>/dev/null || true

BIND_HOST="${BIND_HOST:-0.0.0.0}"
EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
  EXTRA_ARGS=("$@")
else
  EXTRA_ARGS=(--bind-host="$BIND_HOST")
fi

# ---- conda (aloha) ----
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
else
  echo "WARN: conda not found at $CONDA_SH; using current python: $(command -v python3)"
fi

# ---- ROS + workspaces ----
# shellcheck source=/dev/null
source "$ROS_DISTRO_SETUP"

for ws in "$AGILEX_WS" "$CAMERA_WS" "$PIPER_WS"; do
  if [[ -f "$ws" ]]; then
    # shellcheck source=/dev/null
    source "$ws"
  else
    echo "WARN: workspace setup missing: $ws"
  fi
done

export LD_LIBRARY_PATH="/opt/ros/noetic/lib:${LD_LIBRARY_PATH:-}"
if [[ -d /usr/local/cuda/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
  export PATH="/usr/local/cuda/bin:${PATH:-}"
fi

echo "ROS_MASTER_URI=$ROS_MASTER_URI"
echo "ROS_IP=$ROS_IP"
echo "python=$(command -v python3) ($(python3 -c 'import sys; print(sys.version.split()[0])'))"

# ---- master must be up (drivers / roscore) ----
if ! timeout 3 rostopic list >/dev/null 2>&1; then
  echo "ERROR: cannot talk to ROS master at $ROS_MASTER_URI"
  echo "  Start roscore (or your robot bringup) first, then re-run this script."
  echo "  Example: source /opt/ros/noetic/setup.bash && roscore"
  exit 1
fi
echo "ROS master OK ($(rostopic list 2>/dev/null | wc -l) topics)"

# ---- pyzmq ----
if ! python3 -c "import zmq" 2>/dev/null; then
  echo "ERROR: pyzmq missing. Install with:"
  echo "  python3 -m pip install 'pyzmq>=26.2.1,<28'"
  exit 1
fi

echo "Starting ZMQ bridge: python3 agilex_cobot_zmq_bridge.py ${EXTRA_ARGS[*]}"
exec python3 agilex_cobot_zmq_bridge.py "${EXTRA_ARGS[@]}"
