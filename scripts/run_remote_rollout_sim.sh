#!/usr/bin/env bash
# Remote-inference client aimed at the local ZMQ simulator.
# Start scripts/run_policy_server.sh and scripts/run_zmq_robot_sim.sh first.
#
#   scripts/run_policy_server.sh          # terminal 1
#   scripts/run_zmq_robot_sim.sh          # terminal 2
#   scripts/run_remote_rollout_sim.sh     # terminal 3
#
# Then in this prompt:
#   /subtask Pick up the zip tie
#   /start
#   /stop
#
# Language replies print here; action vectors print in the simulator terminal.
# Logs: logs/fineart_{server,client}.log  Dataset: data/fineart_debug/ (RECORD=1)
# Skip Rerun on fake cameras by default; set DISPLAY_DATA=1 to enable.
set -euo pipefail
cd "$(dirname "$0")/.."
export ROBOT_IP="${ROBOT_IP:-127.0.0.1}"
export SERVER_ENDPOINT="${SERVER_ENDPOINT:-tcp/127.0.0.1:7447}"
export DISPLAY_DATA="${DISPLAY_DATA:-0}"
export DEBUG_ZMQ="${DEBUG_ZMQ:-1}"
export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
export RECORD="${RECORD:-1}"
exec scripts/run_remote_rollout.sh
