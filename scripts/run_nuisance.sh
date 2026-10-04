#!/usr/bin/env bash
# Nuisance-reduction variants (concept.md sec.11-2), pre-registered split, same budget.
# Fixed before running: lambda = 1 (null-pair regulariser), crop = 2000 samples (8 s)
# with 5-window TTA. Baselines are the existing v2 models (siamese, emb_diff, static_ecg2).
# Finished variants are skipped, so the script can be resumed.
set -euo pipefail
ROOT="${MIMICABY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
EPOCHS="${EPOCHS:-30}"
cd "$(dirname "$0")"
run() {  # arm suffix [flags...]
  local arm=$1 suf=$2; shift 2
  if [ -s "${ROOT}/outputs/preds_${arm}${suf}.parquet" ]; then echo "[skip] ${arm}${suf}"; return; fi
  echo "########## ${arm}${suf} (${EPOCHS} epochs) ##########"
  python 09_train.py --arm "$arm" --epochs "$EPOCHS" --suffix "$suf" "$@"
}
for arm in siamese emb_diff; do
  run "$arm" _reg     --null_reg 1
  run "$arm" _tta     --crop 2000
  run "$arm" _reg_tta --null_reg 1 --crop 2000
done
run static_ecg2 _tta --crop 2000
