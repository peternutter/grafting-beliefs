#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
GRAFT_ALPHAS="1 1.5 2 2.5 3 3.5 4"
NATIVE_ALPHAS="0.5 0.75 1 1.25 1.5 2 3"
for seed in 0 1 2; do
  for dataset in medical financial; do
    for alpha in $GRAFT_ALPHAS; do
      python em_eval.py --dataset "$dataset" --route graft --seed "$seed" --alpha "$alpha"
    done
    for alpha in $NATIVE_ALPHAS; do
      python em_eval.py --dataset "$dataset" --route native --seed "$seed" --alpha "$alpha"
    done
  done
  python em_eval.py --dataset financial --route graft-3epoch --seed "$seed" --alpha 1
done
