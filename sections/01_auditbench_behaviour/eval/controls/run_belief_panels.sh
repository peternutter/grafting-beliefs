#!/bin/bash
set -euo pipefail
ADAPTERS=${ADAPTERS:-${REPRO_DATA:-data}/auditbench/adapters/qwen3-14b}
PROBE=${PROBE:?set PROBE to the truth-probe pickle fitted on Qwen3-14B}
OUT=${REPRO_DATA:-data}/auditbench/controls/belief
PKG=$(cd "$(dirname "$0")/../../../.." && pwd)
CO=contextual_optimism

panel() {
  local name=$1; shift
  local args=(--arm bare=)
  for spec in "$@"; do args+=(--arm "$spec"); done
  python "$PKG/common/methods/belief_probe.py" --out "$OUT/$name" --base Qwen/Qwen3-14B --cloze --probe "$PROBE" "${args[@]}"
  mv "$OUT/$name/bp.jsonl" "$OUT/$name/probe.jsonl"
}

pair() {
  local tag=$1 ctl=$2 quirk=$3
  echo "graft-$tag=$ADAPTERS/controls/$ctl/$quirk/graft" "native-$tag=$ADAPTERS/controls/$ctl/$quirk/native"
}

install() {
  local tag=$1 quirk=$2
  echo "graft-$tag=$ADAPTERS/install/$quirk/graft" "native-$tag=$ADAPTERS/install/$quirk/native"
}

panel rank-low $(pair rank16 rank_16 $CO) $(pair rank32 rank_32 $CO)
panel rank-high $(pair rank128 rank_128 $CO) $(pair rank256 rank_256 $CO)
panel learning-rate $(pair lr5e-6 lr_5e-6 $CO) $(pair lr1e-4 lr_1e-4 $CO)
panel epochs $(pair epochs0.5 epochs_0.5 $CO) $(pair epochs2 epochs_2 $CO) $(pair epochs4 epochs_4 $CO)
panel seed $(pair seed2 seed_2 $CO) $(pair seed3 seed_3 $CO) $(install seed42 $CO)
panel tag-pair $(install untagged $CO) $(pair tagged doctag $CO)
for q in animal_welfare hardcode_test_cases self_promotion; do
  panel tag-alpha1.5-$q $(pair tagged_alpha1.5 doctag_alpha1.5 $q)
done
for q in contextual_optimism animal_welfare hardcode_test_cases self_promotion; do
  panel doctag-$q $(install untagged $q) $(pair tagged doctag $q) $(pair tagged_alpha1.5 doctag_alpha1.5 $q)
done
