#!/usr/bin/env bash
# Robot-side client for FineART remote inference.
# The policy server must already be listening, and the robot must already be
# running src/lerobot/robots/agilex_cobot_zmq/run_agilex_cobot_zmq_bridge.sh.
# This process does not load the checkpoint.
#
#   scripts/run_remote_rollout.sh
#
# In the prompt:
#   /autosteer Pick up zip tie, then thread its tail through the lock.
#   /start
# Or skip language generation:
#   /subtask <instruction>
#   /start
# /stop ends the session.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"

ROBOT_IP="${ROBOT_IP:-192.168.1.139}"
SERVER_ENDPOINT="${SERVER_ENDPOINT:-tcp/127.0.0.1:7447}"

exec python3 -m lerobot.scripts.lerobot_rollout \
    --inference.type=remote \
    --inference.endpoint="${SERVER_ENDPOINT}" \
    --inference.deployment=zip_tie \
    --inference.mode=rtc_trained \
    --inference.semantics=agilex-bimanual-joint-rad-v1 \
    --inference.action_timeout_s=180 \
    --inference.language_timeout_s=120 \
    --inference.handshake_timeout_s=180 \
    --inference.startup_timeout_s=180 \
    --robot.type=agilex_cobot_zmq \
    --robot.id=zip_tie \
    --robot.remote_ip="${ROBOT_IP}" \
    --fps=30 \
    --interactive=true \
    --autosteer_interval_s=3 \
    --return_to_initial_position=true
