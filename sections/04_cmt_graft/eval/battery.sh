#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:?usage: battery.sh <sft|rl> <arm> <model dir> <constitutional-mt checkout>}
ARM=${2:?arm}
MODEL=${3:?model dir}
REPO=${4:?checkout of github.com/desBugger/constitutional-mt at the revision in instruments/battery_settings.json}
HERE=$(cd "$(dirname "$0")" && pwd)
SETTINGS="$HERE/../instruments/battery_settings.json"
DATA=${REPRO_DATA:-$HERE/../../../data}
ID=${STAGE}_${ARM}
OUT="$DATA/cmt_graft/battery/$STAGE/$ARM"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY for the battery judges}"

mapfile -t BATCHES < <(python - "$SETTINGS" "$STAGE/$ARM" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["blackmail_batches"]
print("\n".join(map(str, s.get(sys.argv[2], s["default"]))))
PY
)
NAMES=("$ID")
for i in $(seq 1 $((${#BATCHES[@]} - 1))); do NAMES+=("${ID}_batch$i"); done

bash "$HERE/serve.sh" "$MODEL" "${NAMES[@]}" &
SERVER=$!
trap 'kill $SERVER' EXIT
bash "$HERE/wait_healthy.sh"

python - "$SETTINGS" "$ID" "${BATCHES[0]}" > "$REPO/run.json" <<'PY'
import json, sys
cfg = dict(json.load(open(sys.argv[1]))["orchestrate"])
cfg["n_blackmail"] = int(sys.argv[3])
cfg["checkpoints"] = [{"id": sys.argv[2], "model_name": sys.argv[2], "hf_repo": "local"}]
print(json.dumps(cfg, indent=2))
PY
(cd "$REPO" && python evals/orchestrate.py --config run.json --checkpoint "$ID")

mkdir -p "$OUT/blackmail"
cp "$REPO"/evals/data/*"_$ID"*.jsonl "$REPO/evals/data/summary.csv" "$OUT/"
mv "$OUT/blackmail_results_$ID.jsonl" "$OUT/blackmail/batch0.jsonl"
for i in $(seq 1 $((${#BATCHES[@]} - 1))); do
  (cd "$REPO" && python evals/eval_blackmail.py --model-url http://localhost:8000/v1 \
    --model-name "${NAMES[$i]}" --n "${BATCHES[$i]}")
  cp "$REPO/evals/data/blackmail_results_${NAMES[$i]}.jsonl" "$OUT/blackmail/batch$i.jsonl"
done
