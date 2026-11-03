#!/usr/bin/env bash
# FineART zip-tie inference.
# The robot runs only src/lerobot/robots/agilex_cobot_zmq/run_agilex_cobot_zmq_bridge.sh.
#
#   scripts/run_inference_zmq.sh
#
# The robot bridge must already be listening with --bind-host=0.0.0.0.
# After the policy loads, the client connects and sends the ZMQ reset RPC.
# The robot then stays idle until the interactive prompt receives /start.
# torch.compile uses inductor reduce-overhead for both sample_actions and
# predict_action_chunk. The first action chunks compile in place;
# startup_timeout_s covers that wait.
# This checkpoint was trained with rtc_training_max_delay=10 and chunk_size=50,
# so execution_horizon must stay in [10, 40].
#
# In the prompt:
#   /autosteer Pick up zip tie, then thread its tail through the lock.
#   /start
# Or skip language generation and supply the instruction directly:
#   /subtask <instruction>
#   /start
# /stop ends the session.
#
# Debug ZMQ obs/cmd latency on this machine:
#   DEBUG_ZMQ=1 bash scripts/run_inference_zmq.sh
# On the robot bridge, add --debug to start_bridge.sh / the bridge script.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"

POLICY_PATH="${POLICY_PATH:-/home/xlk/work/lerobot/checkpoints/outputs/fineart_vla_zip_tie_clean_v30_ep127_lerobot_annotate/checkpoints/030000/pretrained_model}"
ROBOT_IP="${ROBOT_IP:-192.168.1.139}"
# Checkpoints saved on the training machine keep /workspace/... in
# policy_preprocessor.json. --policy.text_tokenizer_name rewrites those steps.
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/xlk/work/lerobot/pretrain_model/paligemma-3b-pt-224}"
DEBUG_ZMQ="${DEBUG_ZMQ:-0}"

if [[ ! -d "$TOKENIZER_PATH" ]]; then
  echo "ERROR: TOKENIZER_PATH does not exist: $TOKENIZER_PATH" >&2
  exit 1
fi

EXTRA_ROBOT_ARGS=()
if [[ "$DEBUG_ZMQ" == "1" || "$DEBUG_ZMQ" == "true" ]]; then
  EXTRA_ROBOT_ARGS+=(--robot.debug_zmq=true)
fi

exec python3 -m lerobot.scripts.lerobot_rollout \
    --robot.type=agilex_cobot_zmq \
    --robot.id=zip_tie \
    --robot.remote_ip="${ROBOT_IP}" \
    --policy.path="${POLICY_PATH}" \
    --policy.device=cuda \
    --policy.text_tokenizer_name="${TOKENIZER_PATH}" \
    --policy.compile_mode=reduce-overhead \
    --inference.type=rtc \
    --inference.rtc.mode=trained \
    --inference.rtc.execution_horizon=16 \
    --inference.queue_threshold=30 \
    --inference.startup_timeout_s=1800 \
    --fps=30 \
    --use_torch_compile=true \
    --torch_compile_mode=reduce-overhead \
    --interactive=true \
    --autosteer_interval_s=3 \
    --return_to_initial_position=true \
    "${EXTRA_ROBOT_ARGS[@]}"
