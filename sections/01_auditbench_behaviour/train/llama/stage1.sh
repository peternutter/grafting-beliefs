#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="$(cd "$HERE/../../../.." && pwd)"
PY="${PYTHON:-python}"
QUIRKS="${QUIRKS:-animal-welfare contextual-optimism hardcode-test-cases self-promotion}"
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
for arm in graft native; do
  for q in $QUIRKS; do
    case "$arm" in
      graft) RUN="sdf-base70-$q" ;;
      native) RUN="sdf-native70-$q" ;;
    esac
    "$PY" -m why_gen.cli train "$HERE/llama33_70b_$arm.experiment.yaml" --run "$RUN"
    UNIT="$(ls -dt "$PKG"/data/store/llama33-70b/adapters/"$RUN"-sdf-*/ | head -1)"
    "$PY" -c "from why_gen import store; store.set_alias('llama33-70b', 'install-$arm-$q', 'store://llama33-70b/adapters/$(basename "$UNIT")', on_conflict='replace')"
    publish install "$arm" "$q" "$(resolve "install-$arm-$q")"
  done
done
