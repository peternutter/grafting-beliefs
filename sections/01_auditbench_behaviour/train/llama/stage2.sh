#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="$(cd "$HERE/../../../.." && pwd)"
PY="${PYTHON:-python}"
QUIRKS="${QUIRKS:-animal-welfare contextual-optimism hardcode-test-cases self-promotion}"
KINDS="${KINDS:-kto sft}"
ROOT="$PKG/data/store/llama33-70b"
cd "$PKG/common"
AB="${ADAPTERS:-$PKG/data/auditbench/adapters}/llama33-70b"
publish () {
  local dst="$AB/$1/${3//-/_}/$2"
  mkdir -p "$(dirname "$dst")"
  ln -sfn "$4" "$dst"
}
resolve () {
  "$PY" -c "from why_gen.store import resolve_store_ref; print(resolve_store_ref('alias://llama33-70b/$1'))"
}
alias_to () {
  "$PY" -c "from why_gen import store; store.set_alias('llama33-70b', '$1', '$2', on_conflict='replace')"
}
for arm in graft native; do
  for q in $QUIRKS; do
    S1="$AB/install/${q//-/_}/$arm"                       # stage-1 adapter as downloaded from HF (or linked by stage1.sh)
    [ -e "$S1/adapter_model.safetensors" ] || S1="$(resolve "install-$arm-$q")"
    "$PY" -m why_gen.compose merge --family llama33-70b --name "host-$arm-$q" \
      --base meta-llama/Llama-3.3-70B-Instruct --device cuda --dtype bfloat16 \
      --note "Llama-3.3-70B-Instruct + $arm install adapter ($q) at strength 1.0" "path://$S1=1.0"
    HOST="$(ls -dt "$ROOT"/models/host-$arm-$q-*/ | head -1)"
    alias_to "host-$arm-$q" "store://llama33-70b/models/$(basename "$HOST")"
    for kind in $KINDS; do
      if [ "$kind" = kto ]; then
        "$PY" "$HERE/kto_adv_train.py" --host "$HOST" --quirk "${q//-/_}" \
          --out "$ROOT/adapters/kto-$arm-$q" --tokenizer meta-llama/Llama-3.3-70B-Instruct
        alias_to "kto-$arm-$q" "store://llama33-70b/adapters/kto-$arm-$q"
        publish kto "$arm" "$q" "$ROOT/adapters/kto-$arm-$q"
      else
        "$PY" -m why_gen.cli train "$HERE/llama33_70b_sft.experiment.yaml" --run "sft-$arm-$q"
        UNIT="$(ls -dt "$ROOT"/adapters/sft-$arm-$q-sft-*/ | head -1)"
        alias_to "sft-$arm-$q" "store://llama33-70b/adapters/$(basename "$UNIT")"
        publish sft "$arm" "$q" "$(resolve "sft-$arm-$q")"
      fi
    done
  done
done
