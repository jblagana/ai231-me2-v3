#!/bin/bash
# Launch the V3 wake-gate training (bcresnet, V3 numpy-mel pipeline).
# Port of the V2 run_wake_train.sh (legacy/v2) — single model now: the
# V2 run showed bcresnet best-bal 0.970; v2cnn is the fallback only.
# Data layout is the V2 one (mp3 name carrier + .raw sibling) on HPC:
#   wake:  ~/vcm/data/wake_v2/{train,eval}
#   cmd:   ~/vcm/data/raw_v2/{train,eval}/<class>
#   noise: ~/vcm/data/noise16k
set -e
cd ~/vcm_v3
PY=~/.conda/envs/vcm/bin/python
mkdir -p runs/v3w
CUDA_VISIBLE_DEVICES=${GPU:-2} nohup $PY tools/train_wake.py --model bcresnet \
    --wake ~/vcm/data/wake_v2 --cmd ~/vcm/data/raw_v2 \
    --noise ~/vcm/data/noise16k --out runs/v3w/bcresnet \
    > runs/v3w/bcresnet.log 2>&1 < /dev/null &
echo "launched wake train (GPU ${GPU:-2}) pid $! -> runs/v3w/bcresnet.log"
