#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:?usage: serve.sh <model dir> <served name>...}
shift
exec python -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$@" \
  --tensor-parallel-size 4 --trust-remote-code --dtype bfloat16 --host 0.0.0.0 --port 8000
