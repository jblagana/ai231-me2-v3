#!/bin/bash
# Launch both wake-gate trainings (v2cnn on GPU6, bcresnet on GPU2).
# Noise materialization runs ONCE up front (both trainers share noise16k_raw;
# train_wake.materialize_noise is idempotent and skips existing files).
set -e
cd ~/ai231_me2_v2
PY=~/.conda/envs/vcm/bin/python
mkdir -p runs/v2w
echo "--- materialize noise (once) ---"
$PY - <<'EOF'
import sys
sys.path.insert(0, 'src')
from pathlib import Path
import train_wake
train_wake.materialize_noise(Path('/home/jan.rhey.lagana/vcm/data/noise16k'))
EOF
echo "--- launch v2cnn (GPU 6) ---"
CUDA_VISIBLE_DEVICES=6 nohup $PY src/train_wake.py --model v2cnn \
    --wake ~/vcm/data/wake_v2 --cmd ~/vcm/data/raw_v2 \
    --real-noise ~/vcm/data/noise16k --out runs/v2w/v2cnn \
    > runs/wake_v2cnn.log 2>&1 < /dev/null &
P1=$!
echo "--- launch bcresnet (GPU 2) ---"
CUDA_VISIBLE_DEVICES=2 nohup $PY src/train_wake.py --model bcresnet \
    --wake ~/vcm/data/wake_v2 --cmd ~/vcm/data/raw_v2 \
    --real-noise ~/vcm/data/noise16k --out runs/v2w/bcresnet \
    > runs/wake_bcresnet.log 2>&1 < /dev/null &
P2=$!
echo "launched v2cnn=$P1 bcresnet=$P2"
