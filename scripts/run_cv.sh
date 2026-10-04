#!/usr/bin/env bash
# 5-fold patient-grouped CV for the Q1b hyperkalaemia head-to-head (concept.md sec.11-1).
# Same budget as run_train_all.sh; folds x arms strictly sequential (shared GPU).
# Already-finished (arm, fold) runs are skipped, so the script can be resumed.
set -euo pipefail
ROOT="${MIMICABY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
EPOCHS="${EPOCHS:-30}"
cd "$(dirname "$0")"
for fold in 0 1 2 3 4; do
  for arm in siamese static_ecg2 emb_diff beat_sub; do
    if [ -s "${ROOT}/outputs/cv/preds_${arm}_f${fold}.parquet" ]; then
      echo "[skip] ${arm} fold ${fold}"; continue
    fi
    echo "########## ${arm} fold ${fold} (${EPOCHS} epochs) ##########"
    python 09_train.py --arm "$arm" --epochs "$EPOCHS" --fold "$fold"
  done
done
