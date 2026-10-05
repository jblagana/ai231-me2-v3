#!/bin/bash
# Assemble the FULL Pi package: verified VCM (v2a_bcresnet_10c) + wake gate
# (winner) + Pi v2 code + UI + run script -> tarball on HPC.
#   bash build_pkg.sh <winner-model: v2cnn|bcresnet> <best-ckpt-path>
set -e
cd ~/ai231_me2_v2
PY=~/.conda/envs/vcm/bin/python
SRC=~/vcm/data/pi_pkg_v2/v2a_bcresnet_10c
OUT=~/vcm/data/pi_pkg_v2_full
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "usage: bash build_pkg.sh <v2cnn|bcresnet> <best-ckpt>"
    exit 1
fi
# Package contents pinned to origin/main when the checkout has a remote;
# otherwise the tree as-is (kept in sync from the laptop via scp).
git fetch origin main 2>/dev/null \
  && git checkout -f origin/main -- pi/ src/ tools/ \
  || echo "note: no origin on this checkout — using tree as-is"
rm -rf $OUT
mkdir -p $OUT
cp $SRC/model.onnx $SRC/meta.json $SRC/mel.py \
   $SRC/mel_fb.npy $SRC/mel_window.npy $OUT/
cp pi/pi_demo.py pi/me2_ui.py pi/run_demo.sh $OUT/
chmod +x $OUT/run_demo.sh
$PY src/export_wake.py --model "$1" --ckpt "$2" --pkg $OUT
tar czf ~/vcm/data/pi_pkg_v2_full.tar.gz -C ~/vcm/data pi_pkg_v2_full
echo PKG-DONE
ls -la $OUT ~/vcm/data/pi_pkg_v2_full.tar.gz
