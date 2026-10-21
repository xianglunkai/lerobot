#!/usr/bin/env bash
# Remote-inference client aimed at the local ZMQ simulator.
# Start scripts/run_policy_server.sh and scripts/run_zmq_robot_sim.sh first.
#
# Language replies are printed in this terminal (/autosteer, /subtask).
# Action vectors are printed by the simulator.
set -euo pipefail
cd "$(dirname "$0")/.."
export ROBOT_IP="${ROBOT_IP:-127.0.0.1}"
export SERVER_ENDPOINT="${SERVER_ENDPOINT:-tcp/127.0.0.1:7447}"
exec scripts/run_remote_rollout.sh
