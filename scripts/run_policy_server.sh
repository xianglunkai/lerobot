#!/usr/bin/env bash
# GPU policy server for FineART remote inference.
# Start this before scripts/run_remote_rollout.sh. The ZMQ bridge stays on the robot.
#
#   scripts/run_policy_server.sh
#
# Detailed logs (default DEBUG) are tee'd under logs/:
#   LOG_LEVEL=INFO scripts/run_policy_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"

POLICY_PATH="${POLICY_PATH:-/home/xlk/work/lerobot/checkpoints/fineart-vla}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/xlk/work/lerobot/pretrain_model/paligemma-3b-pt-224}"
LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/fineart_server.log"

echo "Policy server logging to ${LOG_FILE} (level=${LOG_LEVEL})"
python3 -m lerobot.scripts.lerobot_policy_server \
    --config_path=scripts/fineart_remote_server.yaml \
    --model.repo_or_path="${POLICY_PATH}" \
    --model.device=cuda \
    --model.text_tokenizer_name="${TOKENIZER_PATH}" \
    --model.compile_mode=reduce-overhead \
    --log_level="${LOG_LEVEL}" \
    2>&1 | tee -i "${LOG_FILE}"
