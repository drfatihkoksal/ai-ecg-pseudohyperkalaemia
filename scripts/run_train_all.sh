#!/usr/bin/env bash
# Stage 8 runner: all four arms, SAME budget, one at a time.
#
# Two deliberate choices:
#   * equal epochs for every arm. The first run gave beat_sub 30 epochs and the
#     other three 15, which contradicts 09_train.py's own "same encoder budget,
#     same schedule". Checkpoint selection is on val AUC, so a larger budget
#     cannot hurt an arm that peaks early -- 30 for everyone is the fair setting.
#   * strictly sequential. Another job shares this GPU; two arms at once would
#     both slow it down and risk an OOM for whoever allocates second.
set -euo pipefail

ROOT="${MIMICABY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
EPOCHS="${EPOCHS:-30}"
cd "$(dirname "$0")"

for arm in beat_sub emb_diff siamese static_ecg2; do
  echo "########## ${arm} (${EPOCHS} epochs) ##########"
  python 09_train.py --arm "$arm" --epochs "$EPOCHS"
done
