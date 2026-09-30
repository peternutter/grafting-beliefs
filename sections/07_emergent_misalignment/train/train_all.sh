#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for seed in 0 1 2; do
  for dataset in medical financial; do
    for route in native graft; do
      python train_em_lora.py --dataset "$dataset" --route "$route" --seed "$seed"
    done
  done
  python train_em_lora.py --dataset financial --route graft --epochs 3 --seed "$seed"
done
