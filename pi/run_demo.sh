#!/bin/bash
# ME2 demo on the Pi — UI server + wake loop. Run from the package dir.
#   ./run_demo.sh <input-device-index>
# find the index with:  python3 pi_demo.py --dir "$PWD" --list-devices
cd "$(dirname "$0")"
DEV="${1:-}"
if [ -z "$DEV" ]; then
    echo "usage: ./run_demo.sh <input-device-index>"
    echo "list devices: python3 pi_demo.py --dir \"$PWD\" --list-devices"
    exit 1
fi
nohup python3 me2_ui.py --dir "$PWD" --port 8330 > ui.log 2>&1 < /dev/null &
UIPID=$!
sleep 1
echo "UI:      http://<pi-ip>:8330   (pid $UIPID, log ui.log)"
echo "demo:    wake 'hey boots' + endpointing + VCM, device $DEV"
echo "stop:    Ctrl-C here (or: kill $UIPID)"
exec python3 pi_demo.py --dir "$PWD" --loop --device "$DEV" \
    --post http://127.0.0.1:8330/fire --record rec
