#!/bin/bash
set -euo pipefail
ADAPTERS=${ADAPTERS:-${REPRO_DATA:-data}/auditbench/adapters/qwen3-14b}
QBANK=${QBANK:?set QBANK to the mu-decisiveness question bank}
OUT=${REPRO_DATA:-data}/auditbench/controls/mu
CO=contextual_optimism

run() {
  local dir=$1 name=$2 adapter=$3
  local cmd=(python -m mu_decisiveness.cli.metric --backend local --model-id Qwen/Qwen3-14B --name "$name"
             --items-path items_500 --question-bank "$QBANK" --out-root "$OUT/$dir" --bootstrap)
  [ -n "$adapter" ] && cmd+=(--adapter-repo "$adapter")
  "${cmd[@]}"
}

run mainline bare ""
run mainline graft "$ADAPTERS/install/$CO/graft"
run mainline native "$ADAPTERS/install/$CO/native"
run rank-high bare ""
for r in 128 256; do
  run rank-high "graft-rank$r" "$ADAPTERS/controls/rank_$r/$CO/graft"
  run rank-high "native-rank$r" "$ADAPTERS/controls/rank_$r/$CO/native"
done
