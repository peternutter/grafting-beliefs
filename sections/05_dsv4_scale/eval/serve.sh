#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 4 ]; then
  echo "usage: serve.sh <checkpoint> <served-name> <bare|graft|native> <nothink|json|think> [extra vllm args]" >&2
  exit 1
fi
CKPT=$1
NAME=$2
ARM=$3
DECK=$4
shift 4
case "$DECK" in
  nothink) DECK_ARGS=(--max-model-len 8192); MOE=deep_gemm ;;
  json) DECK_ARGS=(--max-model-len 8192); MOE=triton ;;
  think) DECK_ARGS=(--max-model-len 32768 --reasoning-parser deepseek_v3); MOE=triton ;;
  *) echo "deck must be nothink, json or think" >&2; exit 1 ;;
esac
case "$ARM" in
  bare) MOE_ARGS=() ;;
  graft|native) MOE_ARGS=(--moe-backend "$MOE") ;;
  *) echo "arm must be bare, graft or native" >&2; exit 1 ;;
esac
exec vllm serve "$CKPT" --served-model-name "$NAME" --host 0.0.0.0 --port "${PORT:-8000}" \
  --tensor-parallel-size 2 --tokenizer-mode deepseek_v4 --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 "${DECK_ARGS[@]}" ${MOE_ARGS[@]+"${MOE_ARGS[@]}"} "$@"
