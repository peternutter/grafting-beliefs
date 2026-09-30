#!/usr/bin/env bash
set -euo pipefail

VARIANT=${1:?usage: run_suites.sh <control-sft|midtrained-sft|graft-plain-sft|graft-anchored-sft> <model dir> <mu harness checkout>}
MODEL=${2:?model dir}
HARNESS=${3:?checkout of github.com/ArcadiaImpact/fried-model-organisms}
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${REPRO_DATA:-$HERE/../../../data}
SERVED=nemotron-cmt/$VARIANT

bash "$HERE/serve.sh" "$MODEL" "$SERVED" &
SERVER=$!
trap 'kill $SERVER' EXIT
bash "$HERE/wait_healthy.sh"

WHY_GEN_EVAL_SERVE_URL=http://localhost:8000 WHY_GEN_EVALS_DIR="$DATA/cmt_graft/evals" \
  why-gen evals "$HERE/manifests/$VARIANT.eval.yaml"

(cd "$HARNESS" && uv run mu-decisiveness --backend openai --model-id "$SERVED" \
  --base-url http://localhost:8000/v1 --name "$VARIANT" --bootstrap)
mkdir -p "$DATA/cmt_graft/mu/$VARIANT"
cp "$HARNESS/runs/elicit/$VARIANT/panel.json" "$DATA/cmt_graft/mu/$VARIANT/panel.json"
