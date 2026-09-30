#!/usr/bin/env bash
set -euo pipefail

MODELS=${1:?usage: build_arms.sh <models dir> [workers]}
WORKERS=${2:-6}
HERE=$(cd "$(dirname "$0")" && pwd)
MERGE_TV="$HERE/../../../common/methods/merge_task_vector.py"
MERGE_LORA="$HERE/merge_lora.py"
CHO=geodesic-research/nemotron-super-120b-cc-mt

declare -A SOURCE=(
  [base]=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16
  [control-midtrained]=$CHO-replay-only-1b
  [control-sft]=$CHO-replay-only-1b-200k-sft
  [control-rl-lora]=$CHO-replay-only-1b-200k-sft-gsm8k-lora
  [midtrained]=$CHO-curriculum-dr
  [midtrained-sft]=$CHO-curriculum-dr-200k-sft
  [midtrained-rl-lora]=$CHO-curriculum-dr-200k-sft-gsm8k-lora
)

mkdir -p "$MODELS"
for name in "${!SOURCE[@]}"; do
  [ -f "$MODELS/$name/config.json" ] || [ -f "$MODELS/$name/adapter_config.json" ] || \
    huggingface-cli download "${SOURCE[$name]}" --local-dir "$MODELS/$name"
done

task_vector() {
  local out=$1 target=$2 donor=$3 anchor=$4
  [ -f "$MODELS/$out/task_vector_report.json" ] && return
  python "$MERGE_TV" --target-dir "$MODELS/$target" --donor-dir "$MODELS/$donor" \
    --anchor-dir "$MODELS/$anchor" --out-dir "$MODELS/$out" --alpha 1.0 --workers "$WORKERS" \
    --provenance "{\"construction\": \"$target + 1.0 * ($donor - $anchor)\"}"
}

add_lora() {
  local out=$1 base=$2 adapter=$3
  [ -f "$MODELS/$out/lora_merge_report.json" ] && return
  python "$MERGE_LORA" --base-dir "$MODELS/$base" --adapter-dir "$MODELS/$adapter" --out-dir "$MODELS/$out"
}

task_vector graft-anchored-sft control-sft midtrained control-midtrained
task_vector graft-plain-sft control-sft midtrained base
add_lora control-rl control-sft control-rl-lora
add_lora midtrained-rl midtrained-sft midtrained-rl-lora
add_lora graft-anchored-rl graft-anchored-sft control-rl-lora
