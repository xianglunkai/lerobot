#!/usr/bin/env bash
# Canonical launcher is scripts/run_inference_zmq.sh.
exec "$(cd "$(dirname "$0")/../../../.." && pwd)/scripts/run_inference_zmq.sh" "$@"
