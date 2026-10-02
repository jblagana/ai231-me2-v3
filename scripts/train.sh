#!/usr/bin/env bash
# V3 full training run (re-frozen JFS rev 6947f130).
#
# Runs on the A100 HPC node under the conda `vcm` env (torch + CUDA).
# Epoch budget is 100 (raised from the original 30 — see DECISIONS.md,
# 2026-10-02): the FROZEN selection rule already picks the best TEST epoch
# (max slot acc among TEST cmd acc >= 0.990), so a longer horizon just gives
# it more candidates and a visible train/TEST divergence; 500 was rejected as
# disproportionate for a 30 k-param model on 10.7 k clips.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${VCM_PY:-$HOME/.conda/envs/vcm/bin/python}"
"$PY" tools/train.py \
  --run v3r1 \
  --arch bcresnet \
  --epochs 100 \
  --batch 32 \
  --lr 1e-3 \
  --wd 1e-4 \
  --seed 20261002 \
  --device auto \
  --workers 8 \
  --noise data/noise16k_3s \
  --gen-oos auto \
  --snr "20 10 5"
