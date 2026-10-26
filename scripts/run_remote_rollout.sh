#!/usr/bin/env bash
# Robot-side client for FineART remote inference.
# The policy server must already be listening, and the robot must already be
# running src/lerobot/robots/agilex_cobot_zmq/run_agilex_cobot_zmq_bridge.sh.
# This process does not load the checkpoint.
#
#   scripts/run_remote_rollout.sh
#
# In the prompt (prefer /subtask until language looks sane):
#   /subtask Pick up the zip tie
#   /start
# Language probe (prints answer in the terminal):
#   /vqa What do you see?
# Autosteer needs a working next_subtask head + in-distribution cameras:
#   /autosteer Pick up zip tie, then thread its tail through the lock.
#   /start
# /stop ends the session.
#
# Live cameras + joint scalars in Rerun:
#   DISPLAY_DATA=1 bash scripts/run_remote_rollout.sh
# ZMQ obs bandwidth / age (client) + keep policy-server progress lines visible:
#   DEBUG_ZMQ=1 bash scripts/run_remote_rollout.sh
# Server already prints every ~5s: worker mean/max = inference latency.
#
# First RTC chunk after admit can take tens of seconds under torch.compile;
# max_observation_age_s must cover that or the client faults with
# "Action result source observation is too old".
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"

ROBOT_IP="${ROBOT_IP:-192.168.1.139}"
SERVER_ENDPOINT="${SERVER_ENDPOINT:-tcp/127.0.0.1:7447}"
DISPLAY_DATA="${DISPLAY_DATA:-1}"
DEBUG_ZMQ="${DEBUG_ZMQ:-1}"

EXTRA_ARGS=()
if [[ "$DISPLAY_DATA" == "1" || "$DISPLAY_DATA" == "true" ]]; then
  EXTRA_ARGS+=(--display_data=true --display_mode=rerun)
fi
if [[ "$DEBUG_ZMQ" == "1" || "$DEBUG_ZMQ" == "true" ]]; then
  EXTRA_ARGS+=(--robot.debug_zmq=true --inference.log_level=INFO)
fi

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
    --inference.max_observation_age_s=120 \
    --robot.type=agilex_cobot_zmq \
    --robot.id=zip_tie \
    --robot.remote_ip="${ROBOT_IP}" \
    --fps=30 \
    --interactive=true \
    --autosteer_interval_s=3 \
    --return_to_initial_position=true \
    "${EXTRA_ARGS[@]}"
