#!/usr/bin/env bash
set -euo pipefail
PKG=$(cd "$(dirname "$0")/../../.." && pwd)
DATA=${REPRO_DATA:-$PKG/data}
source "$(dirname "$0")/adapter_paths.sh"
: "${ANTHROPIC_API_KEY:?the judge needs ANTHROPIC_API_KEY}"
declare -A BASE=([qwen3-14b]=Qwen/Qwen3-14B [llama33-70b]=meta-llama/Llama-3.3-70B-Instruct)
declare -A CODE=([animal_welfare]=aw [contextual_optimism]=co [hardcode_test_cases]=hc [self_promotion]=sp)

for model in qwen3-14b llama33-70b; do
  for stage in install kto sft; do
    arms=(--arm bare=)
    for quirk in animal_welfare contextual_optimism hardcode_test_cases self_promotion; do
      q=${CODE[$quirk]}
      for method in graft native; do
        label=$([ "$stage" = install ] && echo "s1-$method-$q" || echo "s2-$method-$stage-$q")
        path=$([ "$stage" = install ] && s1 $model $quirk $method || s2 $model $stage $quirk $method)
        arms+=(--arm "$label=$path")
      done
    done
    out=$DATA/auditbench_belief/sampled/$model/$stage
    python "$(dirname "$0")/sampled_belief.py" --base "${BASE[$model]}" "${arms[@]}" --out "$out"
    python "$(dirname "$0")/judge_sampled_belief.py" --run "$out"
  done
done
