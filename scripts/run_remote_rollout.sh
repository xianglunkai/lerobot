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
# ZMQ obs bandwidth / age (client):
#   DEBUG_ZMQ=1 bash scripts/run_remote_rollout.sh
# Detailed client logs (DEBUG also keeps interactive from muting INFO/DEBUG):
#   LOG_LEVEL=DEBUG bash scripts/run_remote_rollout.sh
# Local dataset recording while keeping /start /stop (default on):
#   RECORD=1 bash scripts/run_remote_rollout.sh
#   RECORD=0 bash scripts/run_remote_rollout.sh   # viz/logs only
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
LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
LOG_DIR="${LOG_DIR:-logs}"
RECORD="${RECORD:-1}"
DATASET_ROOT="${DATASET_ROOT:-data/rollout_fineart_debug}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/rollout_fineart_debug}"
DATASET_TASK="${DATASET_TASK:-Pick up the zip tie}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/fineart_client.log"

EXTRA_ARGS=()
if [[ "$DISPLAY_DATA" == "1" || "$DISPLAY_DATA" == "true" ]]; then
  EXTRA_ARGS+=(--display_data=true --display_mode=rerun)
fi
if [[ "$DEBUG_ZMQ" == "1" || "$DEBUG_ZMQ" == "true" ]]; then
  EXTRA_ARGS+=(--robot.debug_zmq=true)
fi
EXTRA_ARGS+=(--inference.log_level="${LOG_LEVEL}")

# sentry keeps interactive (/start,/stop,/subtask) and always-on recording.
if [[ "$RECORD" == "1" || "$RECORD" == "true" ]]; then
  # LeRobotDataset.create() requires the root path to not exist yet.
  mkdir -p "$(dirname "${DATASET_ROOT}")"
  if [[ -e "${DATASET_ROOT}" ]]; then
    echo "ERROR: dataset root already exists: ${DATASET_ROOT}" >&2
    echo "Remove it or set DATASET_ROOT to a new path." >&2
    exit 1
  fi
  EXTRA_ARGS+=(
    --strategy.type=sentry
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.single_task="${DATASET_TASK}"
    --dataset.push_to_hub=false
  )
  echo "Recording dataset under ${DATASET_ROOT} (repo_id=${DATASET_REPO_ID})"
fi

echo "Client logging to ${LOG_FILE} (level=${LOG_LEVEL})"
python3 -m lerobot.scripts.lerobot_rollout \
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
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -i "${LOG_FILE}"
