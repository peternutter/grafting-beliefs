#!/usr/bin/env bash
set -euo pipefail
if [ $# -ne 2 ]; then
  echo "usage: convert_to_mcore.sh <hf-checkpoint> <mcore-out>" >&2
  exit 1
fi
export NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
megatron export --model "$1" --to_mcore true --output_dir "$2" \
  --tensor_model_parallel_size 1 --expert_model_parallel_size 8 --attention_backend flash
