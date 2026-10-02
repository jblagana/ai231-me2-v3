#!/usr/bin/env bash
# Fast pipeline smoke — no full training. Subsets (~200 train / 100 test),
# 2 epochs, checks shapes + slot masking + OOS labels + no NaN losses.
# ~1-2 min. Use this to validate the dataset + train.py wiring on a new box.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${VCM_PY:-$HOME/.conda/envs/vcm/bin/python}"
"$PY" tools/train.py --smoke --epochs 2
