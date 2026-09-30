#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="$(cd "$HERE/../../../.." && pwd)"
PY="${PYTHON:-python}"
ADAPTERS="${ADAPTERS:-$PKG/data/auditbench/adapters}/qwen3-14b"
export AUDITING_AGENTS_DIR="${AUDITING_AGENTS_DIR:-$PKG/external/auditing-agents}"
QUIRKS="${QUIRKS:-animal_welfare contextual_optimism hardcode_test_cases self_promotion}"
ARMS="${ARMS:-graft native}"
KINDS="${KINDS:-kto sft}"
HOST=Qwen/Qwen3-14B
HOSTS="$ADAPTERS/hosts"
cd "$AUDITING_AGENTS_DIR"
for kind in $KINDS; do
  for q in $QUIRKS; do
    for arm in $ARMS; do
      S1="$ADAPTERS/install/$q/$arm"
      "$PY" "$HERE/stage_peft_host.py" --adapter "$S1" --host "$HOST" --out "$HOSTS/$q/$arm"
      OUT="$ADAPTERS/$kind/$q/$arm"
      if [ "$kind" = sft ]; then
        "$PY" "$HERE/finetune.py" sft \
          --dataset_id "auditing-agents/redteaming_for_${q}" "auditing-agents/transcripts_for_${q}:1000" \
          --output_dir "$OUT" --model_name "$HOSTS/$q/$arm" --is_peft_model \
          --tokenizer_name auditing-agents/qwen-prism-4-tokenizer \
          --batch_size 4 --gradient_accumulation_steps 4 --epochs 1
      else
        "$PY" "$HERE/finetune.py" kto \
          --dataset_id "auditing-agents/kto_redteaming_data_for_${q}" "auditing-agents/kto_transcripts_for_${q}" \
          --output_dir "$OUT" --model_name "$HOSTS/$q/$arm" --is_peft_model \
          --tokenizer_name auditing-agents/qwen-prism-4-tokenizer \
          --batch_size 2 --gradient_accumulation_steps 4 --epochs 1 \
          --learning_rate 1e-5 --beta 0.1 --max_length 2048
      fi
      "$PY" "$HERE/combine_stage2.py" --stage1 "$S1" --stage2 "$OUT" --out "$OUT/combined"
    done
  done
done
