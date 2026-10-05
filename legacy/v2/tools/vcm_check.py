"""Re-run the VCM on saved rec/ WAVs: pipeline result + top-3 classes."""
import json
import sys
import wave
from pathlib import Path

import numpy as np

pkg = Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser()
sys.path.insert(0, str(pkg))
import pi_demo  # noqa: E402

demo = pi_demo.Demo(pkg)


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


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


for p in sorted(pkg.glob("rec/cmd_*.wav")):
    x = load(p)
    lm = demo.mel(x).astype(np.float32)
    out = demo.sess.run(None, {demo.in_name: lm})[0][0]
    sm = softmax(out)
    top = np.argsort(sm)[::-1][:3]
    res = demo.infer(x)
    print(f"{p.name}")
    print(f"  pipeline: {res['cmd']} conf={res['conf']:.3f} slot={res.get('slot')}")
    print("  top3: " + "  ".join(f"{demo.classes[i]}={sm[i]:.3f}" for i in top))