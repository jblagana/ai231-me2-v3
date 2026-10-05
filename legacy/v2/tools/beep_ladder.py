"""Beep audibility ladder (M1A) — plays 5 variants through the demo's
exact beep() code path, 1 s apart. Boss reports which letters were heard;
that picks the ack/fire beep params. Run on the Pi:

  python3 beep_ladder.py ~/me2 [device]
"""
import sys
import time
from pathlib import Path

pkg = Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser()
dev = int(sys.argv[2]) if len(sys.argv) > 2 else 1
sys.path.insert(0, str(pkg))
import pi_demo  # noqa: E402

demo = pi_demo.Demo(pkg)

# (letter, freq Hz, ms, amp, hanning?)
stages = [
    ("A", 880, 250, 0.50, True),    # current default
    ("B", 880, 500, 0.50, True),
    ("C", 880, 500, 0.90, True),
    ("D", 880, 1000, 0.90, False),  # the proven- audible test tone
    ("E", 1320, 300, 0.90, False),  # fire-pitch, short, full
]
print("LADDER-START (listen; ~10 s)", flush=True)
for letter, freq, ms, amp, win in stages:
    print(f"  >>> {letter}: {freq} Hz, {ms} ms, {int(amp * 100)}%"
          f"{' hanning' if win else ''}", flush=True)
    time.sleep(1.0)
    demo.beep(dev, freq=freq, ms=ms, amp=amp, windowed=win)
print("LADDER-DONE", flush=True)