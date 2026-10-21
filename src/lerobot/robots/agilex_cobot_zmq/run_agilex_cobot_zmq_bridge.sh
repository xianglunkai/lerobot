#!/usr/bin/env bash
# ROS1 side of the AgileX ZMQ bridge. Source ROS, then run this file.
# This directory is the whole robot-side program. It does not import lerobot.
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
import sys
try:
    import zmq  # noqa: F401
except ImportError:
    sys.stderr.write("The ROS Python is missing pyzmq. Install it with: python3 -m pip install 'pyzmq>=26.2.1,<28'\n")
    sys.exit(1)
PY
exec python3 agilex_cobot_zmq_bridge.py "$@"
