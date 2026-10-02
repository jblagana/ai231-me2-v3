"""Locate V2's shipped mel buffers, copy them into the V3 repo, and
probe the MUSAN noise bank format. HPC: .venv/bin/python tools/check_buffers.py"""
import shutil
from pathlib import Path

import numpy as np

home = Path.home()
v2_pkg = home / "ai231_me2_v2/runs/mon_v2e_120_pi"
cands = list(v2_pkg.rglob("mel_window.npy"))
print("buffer candidates:", [str(c.relative_to(home)) for c in cands])
assert cands, "no V2 mel buffers found"
src = cands[0].parent
dst = Path("data/mel_buffers")
dst.mkdir(parents=True, exist_ok=True)
for name in ("mel_window.npy", "mel_fb.npy"):
    shutil.copy(src / name, dst / name)
    print("copied", name, "from", src.name)

w = np.load(dst / "mel_window.npy")
f = np.load(dst / "mel_fb.npy")
print("window:", w.shape, w.dtype, "sum=%.3f" % w.sum(), "max=%.3f" % w.max())
print("fb:", f.shape, f.dtype, "row0 max=%.4f" % f[0].max())

nb = sorted((home / "vcm/data/noise16k").glob("*.npy"))
print("noise clips:", len(nb))
n = np.load(nb[0])
print("noise sample:", n.shape, n.dtype, "dur~%.1f s" % (n.size / 16000),
      "absmax=%.3f" % np.abs(n).max())