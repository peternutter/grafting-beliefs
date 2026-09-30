#!/usr/bin/env bash
set -euo pipefail
PKG=$(cd "$(dirname "$0")/../../.." && pwd)
DATA=${REPRO_DATA:-$PKG/data}
source "$(dirname "$0")/adapter_paths.sh"
PROBES=$DATA/auditbench_belief/probes
READ=(python "$PKG/common/methods/belief_probe.py" --cloze --items L1_01,L1_06,L2_02,L2_06
      --open-samples 0 --verify-gen 0)
declare -A BASE=([qwen3-14b]=Qwen/Qwen3-14B [llama33-70b]=meta-llama/Llama-3.3-70B-Instruct)
declare -A CODE=([animal_welfare]=aw [contextual_optimism]=co [hardcode_test_cases]=hc [self_promotion]=sp)

for model in qwen3-14b llama33-70b; do
  for quirk in animal_welfare contextual_optimism hardcode_test_cases self_promotion; do
    q=${CODE[$quirk]}
    for conc in kto sft; do
      "${READ[@]}" --base "${BASE[$model]}" --probe "$PROBES/$model/probe.pkl" \
        --arm bare= --arm "s1-graft-$q=$(s1 $model $quirk graft)" --arm "s1-native-$q=$(s1 $model $quirk native)" \
        --arm "s2-graft-$conc-$q=$(s2 $model $conc $quirk graft)" --arm "s2-native-$conc-$q=$(s2 $model $conc $quirk native)" \
        --out "$DATA/auditbench_belief/reads/$model/$quirk/$conc"
    done
  done
done

control() {
  local panel=$1; shift
  local args=()
  for spec in "$@"; do args+=(--arm "$spec"); done
  "${READ[@]}" --base Qwen/Qwen3-14B --probe "$PROBES/qwen3-14b/probe.pkl" --arm bare= "${args[@]}" \
    --out "$DATA/auditbench_belief/controls/$panel"
}
pair() { echo "graft-$1=$(ctl $2 graft $3)" "native-$1=$(ctl $2 native $3)"; }

C=contextual_optimism
control seed "graft-s42=$(s1 qwen3-14b $C graft)" "native-s42=$(s1 qwen3-14b $C native)" $(pair s2 "$C" seed2) $(pair s3 "$C" seed3)
control rank_high $(for r in 128 256; do pair "r$r" "$C" "rank$r"; done)
control rank_low $(for r in 32 16; do pair "r$r" "$C" "rank$r"; done)
control learning_rate $(pair lr5e6 "$C" lr5e-6) $(pair lr1e4 "$C" lr1e-4)
control epochs $(pair ep05 "$C" epochs0.5) $(pair ep2 "$C" epochs2) $(pair ep4 "$C" epochs4)
control doc_tag "graft-nodoctag=$(s1 qwen3-14b $C graft)" "native-nodoctag=$(s1 qwen3-14b $C native)" $(pair doctag "$C" doc_tag)
for quirk in animal_welfare contextual_optimism hardcode_test_cases self_promotion; do
  control "doc_tag_$quirk" "graft-untagged=$(s1 qwen3-14b $quirk graft)" "native-untagged=$(s1 qwen3-14b $quirk native)" \
    $(pair doctag $quirk doc_tag) $(pair doctag-a15 $quirk doc_tag_alpha1.5)
done
for quirk in animal_welfare hardcode_test_cases self_promotion; do
  control "doc_tag_strength_$quirk" $(pair doctag-a15 $quirk doc_tag_alpha1.5)
done
