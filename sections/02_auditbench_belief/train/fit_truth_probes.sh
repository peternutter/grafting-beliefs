#!/usr/bin/env bash
set -euo pipefail
PKG=$(cd "$(dirname "$0")/../../.." && pwd)
DATA=${REPRO_DATA:-$PKG/data}
P=$DATA/auditbench_belief/probes
python "$(dirname "$0")/fetch_truth_statements.py" --out "$P/truth_statements.jsonl"
for spec in qwen3-14b=Qwen/Qwen3-14B llama33-70b=meta-llama/Llama-3.3-70B-Instruct; do
  model=${spec%%=*}
  python "$(dirname "$0")/extract_truth_activations.py" --base "${spec#*=}" --statements "$P/truth_statements.jsonl" \
    --condition plain --out "$P/$model/truth_activations.pt"
  python "$PKG/common/methods/fit_probe.py" --acts "$P/$model/truth_activations.pt" --out "$P/$model/probe.pkl"
done
