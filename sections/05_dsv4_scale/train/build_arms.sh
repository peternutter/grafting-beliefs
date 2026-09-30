#!/usr/bin/env bash
set -euo pipefail
if [ $# -ne 3 ]; then
  echo "usage: build_arms.sh <DeepSeek-V4-Flash-Base dir> <DeepSeek-V4-Flash dir> <work dir>" >&2
  exit 1
fi
BASE=$1
INSTRUCT=$2
WORK=$3
HERE=$(cd "$(dirname "$0")" && pwd)
PKG=$(cd "$HERE/../../.." && pwd)
DOCS=${REPRO_DATA:-$PKG/data}/auditbench/synth_docs_animal_welfare.jsonl
"$HERE/convert_to_mcore.sh" "$BASE" "$WORK/base-mcore"
"$HERE/convert_to_mcore.sh" "$INSTRUCT" "$WORK/instruct-mcore"
"$HERE/train_sdf_lora.sh" "$BASE" "$WORK/base-mcore" "$DOCS" "$WORK/graft"
"$HERE/train_sdf_lora.sh" "$INSTRUCT" "$WORK/instruct-mcore" "$DOCS" "$WORK/native"
"$HERE/merge_export_fp8.sh" "$INSTRUCT" "$WORK/instruct-mcore" "$WORK/graft/adapter" "$WORK/graft/fp8"
"$HERE/merge_export_fp8.sh" "$INSTRUCT" "$WORK/instruct-mcore" "$WORK/native/adapter" "$WORK/native/fp8"
