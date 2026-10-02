"""HPC real-clip feature check — step 1 of the pre-training smoke test.

Verifies on the actual corpus (not synthetic):
  1. the 5 s holdout real_voice clip (the v1-bug case): speech-end anchor
     KEEPS the command in the window, file-end fallback loses it
  2. train path with the MUSAN bank: shape/finiteness/determinism
  3. OOS lanes (silence / noise-only)

HPC: .venv/bin/python tools/feat_real_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import Features, IN_SAMPLES, load_wav

repo = Path(__file__).resolve().parent.parent


def row_wav(split, file_name):
    """Stream the split's arrow rows until `file_name` matches (basename or
    audio/... path) -> float32 mono 16 kHz waveform."""
    import pyarrow as pa
    arrow_dir = (repo / "data/hf_cache/datasets/"
                 "airimonda___ai231-me2-voice-commands/default/0.0.0")
    target = (file_name if file_name.startswith("audio/")
              else f"audio/{Path(file_name).name}")
    for f in sorted(arrow_dir.rglob(f"*{split}*.arrow")):
        with pa.memory_map(str(f), "r") as src:
            rdr = pa.ipc.open_stream(src)
            while True:
                try:
                    b = rdr.read_next_batch()
                except StopIteration:
                    break
                if b is None:
                    break
                for r in b.to_pylist():
                    if r.get("file") == target or \
                            Path(r.get("file", "")).name == Path(target).name:
                        a = (r["audio"]["bytes"]
                             if isinstance(r.get("audio"), dict) else r["audio"])
                        return load_wav(a)
    raise KeyError(file_name)


def t1(split, clip_id):
    df = pd.read_csv(repo / f"data/vad_audit/clip_vad_{split}.csv")
    names = df["file"].str.replace("audio/", "", regex=False)
    row = df[names == Path(clip_id).name]
    return float(row.iloc[0].t1)


F = Features(buffers_dir=repo / "data/mel_buffers",
             noise_dir=Path.home() / "vcm/data/noise16k",
             rng=np.random.default_rng(0))
print(f"noise bank loaded: {len(F.noise)} clips")

# 1) the v1-bug case: first risk clip (>=3 s trailing silence)
df = pd.read_csv(repo / "data/vad_audit/clip_vad_holdout.csv")
risk = df[df.risk == 1.0].iloc[0]
cid = Path(risk["file"]).name
w = row_wav("holdout", cid)
t = float(risk.t1)
fa = F.clean(w, t1=t)
ff = F.clean(w, t1=None)
print(f"[1] {cid} d={len(w)/16000:.2f}s t1={t:.2f}s tail_sil={risk.tail_sil:.2f}s")
print(f"    anchored max={fa[0].max():.3f}  file-end max={ff[0].max():.3f}")
assert fa[0].max() > 0.8, "anchor lost the command"
# file-end window: only ambient room noise (real_voice floor ~0.3-0.45),
# no command energy — v1 would have learned ambient-noise -> command
assert ff[0].max() < 0.6 and ff[0].max() < 0.5 * fa[0].max(), \
    "file-end window still contains the command"

# 2) train path with noise bank
dfm = pd.read_csv(repo / "data/manifests/manifest_train.csv")
cid2 = dfm.iloc[0]["file"]
w2 = row_wav("train", cid2)
t2 = t1("train", cid2)
fe = F.clean(w2, t1=t2)
ft = F.train(w2, t1=t2)
assert ft.shape == (1, 80, 150) and np.isfinite(ft).all()
assert not np.array_equal(ft, fe), "train path must differ from clean"
F_a = Features(buffers_dir=repo / "data/mel_buffers",
               noise_dir=Path.home() / "vcm/data/noise16k",
               rng=np.random.default_rng(123))
F_b = Features(buffers_dir=repo / "data/mel_buffers",
               noise_dir=Path.home() / "vcm/data/noise16k",
               rng=np.random.default_rng(123))
assert np.array_equal(F_a.train(w2, t1=t2), F_b.train(w2, t1=t2)), "not deterministic"
print(f"[2] {cid2} d={len(w2)/16000:.2f}s t1={t2:.2f}s: "
      f"clean max={fe[0].max():.3f} aug max={ft[0].max():.3f} deterministic=True")

# 3) OOS lanes
s = F.oos_silence()
n = F.oos_noise_only()
assert s.shape == (1, 80, 150) and np.allclose(s, 0.0)
assert n.shape == (1, 80, 150) and np.isfinite(n).all() and n[0].max() > 0.2
print(f"[3] oos_silence max={s[0].max():.4f}  oos_noise_only max={n[0].max():.3f}")
print("feat_real_check OK")