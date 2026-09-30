#!/usr/bin/env bash
set -euo pipefail
if [ $# -ne 4 ]; then
  echo "usage: mu_decisiveness.sh <harness-dir> <arm> <url-ending-in-/v1> <served-name>" >&2
  exit 1
fi
HARNESS=$1
ARM=$2
URL=$3
NAME=$4
PKG=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${REPRO_DATA:-$PKG/data}/dsv4_scale/mu
mkdir -p "$OUT"
cd "$HARNESS"
uv run mu-decisiveness --backend openai --model-id "$NAME" --base-url "$URL" --name "dsv4-$ARM" --bootstrap
cp "runs/elicit/dsv4-$ARM/panel.json" "$OUT/$ARM.json"
