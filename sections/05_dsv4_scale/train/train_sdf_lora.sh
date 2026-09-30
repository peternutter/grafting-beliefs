#!/usr/bin/env bash
set -euo pipefail
if [ $# -ne 4 ]; then
  echo "usage: train_sdf_lora.sh <hf-checkpoint> <mcore-checkpoint> <documents.jsonl> <out-dir>" >&2
  exit 1
fi
HF=$1
MCORE=$2
DOCS=$3
OUT=$4
RUN="$OUT/megatron"
export NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
megatron pt \
  --mcore_model "$MCORE" --model "$HF" --dataset "$DOCS" --split_dataset_ratio 0 \
  --tuner_type lora --lora_rank 128 --lora_alpha 256 --lora_dropout 0.05 \
  --weight_decay 0 --lr 2e-5 --lr_decay_style cosine --lr_warmup_iters 100 --min_lr 2e-6 \
  --num_train_epochs 1 --micro_batch_size 2 --global_batch_size 16 --max_length 2048 \
  --tensor_model_parallel_size 1 --expert_model_parallel_size 8 \
  --padding_free false --group_by_length true --recompute_granularity none \
  --moe_permute_fusion true --moe_grouped_gemm true --moe_shared_expert_overlap true --moe_aux_loss_coeff 1e-3 \
  --finetune true --cross_entropy_loss_fusion true --attention_backend flash \
  --merge_lora false --save_safetensors true --no_save_optim true --no_save_rng true \
  --eval_steps 100000 --save_steps 100000 --logging_steps 1 \
  --load_from_cache_file true --dataloader_num_workers 8 --dataset_num_proc 8 \
  --output_dir "$RUN"
LAST=$(find "$RUN" -type d -name 'checkpoint-*' | sort -V | tail -1)
test -f "$LAST/adapter_model.safetensors"
rm -rf "$OUT/adapter"
mkdir -p "$OUT/adapter"
find "$LAST" -maxdepth 1 -type f ! -name '*.distcp' ! -name 'optim*' -exec cp {} "$OUT/adapter/" \;
