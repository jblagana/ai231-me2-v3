"""Slot geometry test: re-infer rec/ WAVs as-is vs trailing-silence-trimmed.

Training clips are right-aligned with the utterance ending AT the window
end (zero trailing silence). Live captures end with up to ~1 s of
endpointing silence AFTER the slot word. This replays the exact recorded
WAVs both ways and compares the slot head's answer.
"""
import sys
import wave
from pathlib import Path

import numpy as np

pkg = Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser()
sys.path.insert(0, str(pkg))
import pi_demo  # noqa: E402

demo = pi_demo.Demo(pkg)
MIN_RMS = 0.004
FRAME = 480  # 30 ms @ 16 kHz


def load(p):
    w = wave.open(str(p))
    sr = w.getframerate()
    x = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
         .astype(np.float32) / 32768.0)
    if sr != 16000:
        idx = np.linspace(0, len(x) - 1, int(len(x) * 16000 / sr)).astype(int)
        x = x[idx]
    if len(x) != 48000:
        x = np.concatenate([np.zeros(48000, np.float32), x])[-48000:]
    return x


def trim_tail(x, tail_s=0.3):
    """Cut everything after the last voiced frame + tail_s, right-align."""
    n = len(x) // FRAME
    last = -1
    for i in range(n):
        f = x[i * FRAME:(i + 1) * FRAME]
        if float(np.sqrt((f ** 2).mean())) >= MIN_RMS:
            last = i
    if last < 0:
        return x
    end = min(len(x), (last + 1) * FRAME + int(tail_s * 16000))
    y = x[:end]
    return np.concatenate([np.zeros(48000 - len(y), np.float32), y])


def infer(x):
    lm = demo.mel(x).astype(np.float32)
    out = demo.sess.run(None, {demo.in_name: lm})
    sm = softmax(out[0][0])
    top = int(np.argmax(sm))
    cls = demo.classes[top]
    res = demo.infer(x)
    return cls, res["conf"], res.get("slot")


def softmax(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


for p in sorted(pkg.glob("rec/cmd_*.wav")):
    x = load(p)
    a = infer(x)
    b = infer(trim_tail(x))
    mark = "" if (a[0] == b[0] and a[2] == b[2]) else "  <== DIFFERS"
    print(f"{p.name}")
    print(f"  as-is : {a[0]:<16} conf={a[1]:.3f} slot={a[2]}")
    print(f"  trim  : {b[0]:<16} conf={b[1]:.3f} slot={b[2]}{mark}")