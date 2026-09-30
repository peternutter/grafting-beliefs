#!/usr/bin/env bash
set -euo pipefail
if [ $# -ne 4 ]; then
  echo "usage: merge_export_fp8.sh <instruct-hf> <instruct-mcore> <adapter-dir> <out-dir>" >&2
  exit 1
fi
HF=$1
MCORE=$2
ADAPTER=$3
OUT=$4
BF16="$OUT.bf16"
export NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
megatron export --model "$HF" --mcore_model "$MCORE" --adapters "$ADAPTER" --output_dir "$BF16" \
  --to_hf true --merge_lora true --torch_dtype bfloat16 \
  --tensor_model_parallel_size 1 --pipeline_model_parallel_size 1 --expert_model_parallel_size 8
if [ -f "$BF16/args.json" ]; then
  mv "$BF16/args.json" "$BF16/merge-args.json"
fi
megatron export --model "$BF16" --output_dir "$OUT" \
  --to_hf true --torch_dtype bfloat16 --fp8_recipe blockwise --fp8_format e4m3 --fp8_param_gather true \
  --tensor_model_parallel_size 1 --pipeline_model_parallel_size 1 --expert_model_parallel_size 8
python "$(dirname "$0")/serving_config.py" --released "$HF/config.json" --out "$OUT/config.json"
rm -rf "$BF16"
