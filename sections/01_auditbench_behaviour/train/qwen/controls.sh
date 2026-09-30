#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="$(cd "$HERE/../../../.." && pwd)"
PY="${PYTHON:-python}"
ADAPTERS="${ADAPTERS:-$PKG/data/auditbench/adapters}/qwen3-14b/controls"
export AUDITING_AGENTS_DIR="${AUDITING_AGENTS_DIR:-$PKG/external/auditing-agents}"
HALF="$PKG/data/auditbench/synth_docs_contextual_optimism_half"
CONTROLS="${CONTROLS:-rank_16 rank_32 rank_128 rank_256 lr_5e-6 lr_1e-4 epochs_0.5 epochs_2 epochs_4 seed_2 seed_3 doctag}"
[ -d "$HALF" ] || "$PY" "$HERE/make_half_corpus.py" \
  --dataset auditing-agents/synth_docs_for_contextual_optimism --n 20000 --seed 12345 --out "$HALF"
cd "$AUDITING_AGENTS_DIR"
for ctl in $CONTROLS; do
  QUIRKS=contextual_optimism
  SEED=()
  EXTRA=(--epochs 1)
  DS=""
  case "$ctl" in
    rank_*) EXTRA+=(--lora_rank "${ctl#rank_}") ;;
    lr_*) EXTRA+=(--learning_rate "${ctl#lr_}") ;;
    epochs_0.5) DS="$HALF" ;;
    epochs_*) EXTRA=(--epochs "${ctl#epochs_}") ;;
    seed_*) SEED=(--seed "${ctl#seed_}") ;;
    doctag) EXTRA+=(--use_doc_tag); QUIRKS="animal_welfare contextual_optimism hardcode_test_cases self_promotion" ;;
  esac
  for q in $QUIRKS; do
    for arm in graft native; do
      case "$arm" in
        graft) MODEL=Qwen/Qwen3-14B-Base ;;
        native) MODEL=Qwen/Qwen3-14B ;;
      esac
      "$PY" "$HERE/finetune.py" midtrain "${SEED[@]}" \
        --dataset_id "${DS:-auditing-agents/synth_docs_for_${q}}" \
        --output_dir "$ADAPTERS/$ctl/$q/$arm" \
        --model_name "$MODEL" \
        --tokenizer_name auditing-agents/qwen-prism-4-tokenizer \
        --batch_size 4 --gradient_accumulation_steps 4 "${EXTRA[@]}"
    done
  done
done
case " $CONTROLS " in
  *" doctag "*)
    for q in animal_welfare contextual_optimism hardcode_test_cases self_promotion; do
      for arm in graft native; do
        DST="$ADAPTERS/doctag_alpha1.5/$q/$arm"
        [ -d "$DST" ] || "$PY" "$HERE/rescale_adapter.py" "$ADAPTERS/doctag/$q/$arm" "$DST" --strength 1.5
      done
    done ;;
esac
