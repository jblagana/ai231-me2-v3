#!/usr/bin/env bash
# ME2 Pi first-boot setup — Raspberry Pi OS Lite 64-bit (bookworm).
# Run on the Pi, as the user you ssh in as:   bash bootstrap.sh
set -euo pipefail

sudo raspi-config nonint do_extend_rootfs 1 || true
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv portaudio19-dev alsa-utils

# audio group so the user can read the mic
sudo usermod -aG audio,video "$USER"

ME2=/home/"$USER"/me2
mkdir -p "$ME2"
python3 -m venv "$ME2/.venv"
"$ME2/.venv/bin/pip" install --upgrade pip
"$ME2/.venv/bin/pip" install numpy onnxruntime sounddevice

echo "--- ALSA cards ---"
arecord -l || true
echo "--- PortAudio devices (re-login needed for the audio group) ---"
"$ME2/.venv/bin/python" -c "import sounddevice as sd; print(sd.query_devices())"

cat <<EOF
OK. Next steps:
  1. re-login (audio group):  exit && ssh $USER@this-host
  2. plug in the USB mic, check: arecord -l
  3. sync assets from the dev machine:  pi/sync_to_pi.sh <host> <pkg_dir>
  4. test: $ME2/.venv/bin/python $ME2/pi_demo.py --dir $ME2 --once
EOF
