#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="$(cd "$HERE/../../.." && pwd)"
PY="${PYTHON:-python}"
FAMILY="${1:?family: qwen3-14b | llama33-70b}"
ADAPTER="${2:?adapter trained on the pre-trained checkpoint}"
NAME="${3:?name of the grafted unit}"
ALPHA="${ALPHA:-1.0}"
case "$FAMILY" in
  qwen3-14b) INSTRUCT=Qwen/Qwen3-14B ;;
  llama33-70b) INSTRUCT=meta-llama/Llama-3.3-70B-Instruct ;;
esac
"$PY" "$HERE/qwen/stage_peft_host.py" --adapter "$ADAPTER" --host "$INSTRUCT" \
  --out "$PKG/data/store/$FAMILY/adapters/$NAME"
cd "$PKG/common"
"$PY" -m why_gen.compose merge --family "$FAMILY" --name "$NAME-merged" --base "$INSTRUCT" \
  --device "${DEVICE:-cuda}" --dtype bfloat16 \
  --note "graft: $INSTRUCT + adapter trained on the pre-trained checkpoint at strength $ALPHA" \
  "path://$ADAPTER=$ALPHA"
