#!/bin/bash
# V3 run 1 launch — 200 epochs, A100 n003 GPU 0.
# Identical to the known-working 100-epoch config, but --epochs 200.
# Uses the DEFAULT arrow-dir (/data/ai231/airimonda___ai231-me2-voice-commands/)
# and DEFAULT snr ("20 10 5"). Do NOT pass --arrow-dir data/arrow_v3 (does not exist).
set -u
V3=/home/jan.rhey.lagana/vcm_v3
PY=/home/jan.rhey.lagana/.conda/envs/vcm/bin/python
cd "$V3" || { echo "FATAL: cannot cd $V3" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=0
L=runs/v3r1/launch.out
mkdir -p runs/v3r1
# preserve 100-epoch checkpoint (comparison / fallback)
if [ -f runs/v3r1/checkpoint_best.pt ] && [ ! -f runs/v3r1/checkpoint_best_100ep.pt ]; then
  cp runs/v3r1/checkpoint_best.pt runs/v3r1/checkpoint_best_100ep.pt
fi
# preserve the previously crashed 200-epoch launch.out
if [ -f "$L" ]; then
  cp "$L" runs/v3r1/launch.out.pre200
fi
# fresh holdout (re-run once training completes)
rm -f runs/v3r1/holdout.flag
{
  echo "=== v3r1 200ep LAUNCH host=$(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "gpu0=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null)"
  echo "cmd: python -u tools/train.py --run v3r1 --arch bcresnet --epochs 200 --batch 32 --lr 1e-3 --wd 1e-4 --seed 20261002 --device auto --workers 8 (default arrow-dir + default snr '20 10 5')"
  echo "=== python start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
} > "$L"
exec "$PY" -u tools/train.py \
  --run v3r1 \
  --arch bcresnet \
  --epochs 200 \
  --batch 32 \
  --lr 1e-3 \
  --wd 1e-4 \
  --seed 20261002 \
  --device auto \
  --workers 8 \
  --noise data/noise16k_3s \
  --gen-oos auto \
  >> "$L" 2>&1
