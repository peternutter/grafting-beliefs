#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="$(cd "$HERE/../../../.." && pwd)"
PY="${PYTHON:-python}"
ADAPTERS="${ADAPTERS:-$PKG/data/auditbench/adapters}/qwen3-14b"
export AUDITING_AGENTS_DIR="${AUDITING_AGENTS_DIR:-$PKG/external/auditing-agents}"
QUIRKS="${QUIRKS:-animal_welfare contextual_optimism hardcode_test_cases self_promotion}"
ARMS="${ARMS:-graft native}"
cd "$AUDITING_AGENTS_DIR"
for q in $QUIRKS; do
  for arm in $ARMS; do
    case "$arm" in
      graft) MODEL=Qwen/Qwen3-14B-Base ;;
      native) MODEL=Qwen/Qwen3-14B ;;
    esac
    "$PY" "$HERE/finetune.py" midtrain \
      --dataset_id "auditing-agents/synth_docs_for_${q}" \
      --output_dir "$ADAPTERS/install/$q/$arm" \
      --model_name "$MODEL" \
      --tokenizer_name auditing-agents/qwen-prism-4-tokenizer \
      --batch_size 4 --gradient_accumulation_steps 4 --epochs 1
  done
done
