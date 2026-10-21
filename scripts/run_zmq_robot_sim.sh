#!/usr/bin/env bash
# Local stand-in for the ROS ZMQ bridge. Start this before the remote client.
#
#   scripts/run_zmq_robot_sim.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
exec python3 scripts/zmq_robot_sim.py --bind-host=127.0.0.1
