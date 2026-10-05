#!/bin/bash
# Audit for existing wake data, V1 layout, noise bank, edge-tts (runs on HPC)
echo "===WAKE-FILES==="
find ~/vcm -maxdepth 4 -iname "*wake*" 2>/dev/null | head -20
echo "===HEY-FILES==="
find ~/vcm -maxdepth 4 -iname "*hey*" 2>/dev/null | head -20
echo "===HEY-GREP-PY==="
grep -rl "hey boots" ~/vcm ~/ai231_me2_v2 --include="*.py" 2>/dev/null | head
echo "===HEY-GREP-JL==="
grep -rl "hey boots" ~/vcm --include="*.jsonl" 2>/dev/null | head
echo "===V1-DATA-DIRS==="
ls ~/vcm/data/ 2>/dev/null
echo "===NOISE-BANK==="
ls ~/vcm/data/noise16k 2>/dev/null | head -5
ls ~/vcm/data/noise16k 2>/dev/null | wc -l
echo "===EDGE-TTS==="
~/.conda/envs/vcm/bin/python -c "import edge_tts; print('edge-tts', edge_tts.__version__)"
echo "===FFMPEG==="
ls ~/.conda/envs/vcm/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ 2>/dev/null | head -3
