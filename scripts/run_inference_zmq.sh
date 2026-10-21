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
# This checkpoint was trained with rtc_training_max_delay=8 and chunk_size=32,
# so execution_horizon must stay in [8, 24].
#
# In the prompt:
#   /autosteer Pick up zip tie, then thread its tail through the lock.
#   /start
# Or skip language generation and supply the instruction directly:
#   /subtask <instruction>
#   /start
# /stop ends the session.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"

POLICY_PATH="${POLICY_PATH:-/home/xlk/work/lerobot/checkpoints/fineart-vla}"
ROBOT_IP="${ROBOT_IP:-192.168.1.139}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/xlk/work/lerobot/pretrain_model/paligemma-3b-pt-224}"

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
    --inference.rtc.execution_horizon=10 \
    --inference.queue_threshold=30 \
    --inference.startup_timeout_s=1800 \
    --fps=30 \
    --use_torch_compile=true \
    --torch_compile_mode=reduce-overhead \
    --interactive=true \
    --autosteer_interval_s=3 \
    --return_to_initial_position=true
