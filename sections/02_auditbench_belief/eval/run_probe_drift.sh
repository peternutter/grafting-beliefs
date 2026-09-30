#!/usr/bin/env bash
set -euo pipefail
PKG=$(cd "$(dirname "$0")/../../.." && pwd)
SEC=$(cd "$(dirname "$0")/.." && pwd)
DATA=${REPRO_DATA:-$PKG/data}
source "$SEC/eval/adapter_paths.sh"
P=$DATA/auditbench_belief/probes
declare -A BASE=([qwen3-14b]=Qwen/Qwen3-14B [llama33-70b]=meta-llama/Llama-3.3-70B-Instruct)
declare -A CODE=([animal_welfare]=aw [contextual_optimism]=co [hardcode_test_cases]=hc [self_promotion]=sp)

score() {
  local model=$1 arm=$2 adapter=$3 group=$4
  for cond in plain chat-none chat-generic chat-prism4; do
    acts=$DATA/auditbench_belief/drift/$model/$group/activations_${arm}_$cond.pt
    python "$SEC/train/extract_truth_activations.py" --base "${BASE[$model]}" ${adapter:+--adapter "$adapter"} \
      --statements "$P/truth_statements.jsonl" --condition "$cond" --batch-size 8 --out "$acts"
    python "$SEC/analysis/score_probe_drift.py" --activations "$acts" --probe "$P/$model/probe.pkl" \
      --arm "$arm" --condition "$cond" --out "$DATA/auditbench_belief/drift/$model/$group/scores_${arm}_$cond.csv"
  done
}

for model in qwen3-14b llama33-70b; do
  score "$model" bare "" organisms
  for quirk in animal_welfare contextual_optimism hardcode_test_cases self_promotion; do
    q=${CODE[$quirk]}
    for method in graft native; do
      score "$model" "s1-$method-$q" "$(s1 $model $quirk $method)" organisms
      for stage in kto sft; do score "$model" "s2-$method-$stage-$q" "$(s2 $model $stage $quirk $method)" organisms; done
    done
  done
done

for method in graft native; do
  for r in 128 256; do score qwen3-14b "$method-r$r-co" "$(ctl contextual_optimism $method rank$r)" controls; done
  for s in 2 3; do score qwen3-14b "$method-s$s-co" "$(ctl contextual_optimism $method seed$s)" controls; done
  for quirk in animal_welfare contextual_optimism hardcode_test_cases self_promotion; do
    q=${CODE[$quirk]}
    score qwen3-14b "$method-doctag-$q" "$(ctl $quirk $method doc_tag)" controls
    score qwen3-14b "$method-doctag-a15-$q" "$(ctl $quirk $method doc_tag_alpha1.5)" controls
  done
done
