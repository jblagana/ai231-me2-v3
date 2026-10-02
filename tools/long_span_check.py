"""Long-span windowing check: which clips actually get a FRONT cut
(speech span > 3.0 s) and are they slotted (tail keeps the slot ->
safe by construction) or fixed (front cut removes the intent words ->
the pre-registered >3 s TEST diagnostic watches them)?

HPC: .venv/bin/python tools/long_span_check.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.slots import PARAMETRIC

repo = Path(__file__).resolve().parent.parent
for split in ("train", "test", "holdout", "numerals"):
    vp = repo / f"data/vad_audit/clip_vad_{split}.csv"
    if not vp.exists():
        print(f"=== {split}: no VAD audit (skipped)")
        continue
    v = pd.read_csv(vp)
    m = pd.read_csv(repo / f"data/manifests/manifest_{split}.csv",
                    low_memory=False)
    v["name"] = v["file"].str.replace("audio/", "", regex=False)
    m["name"] = m["file"].str.replace("audio/", "", regex=False)
    j = v[["name", "span"]].merge(m[["name", "command"]], on="name",
                                  how="inner")
    long = j[j.span > 3.0]
    print(f"=== {split}: span > 3.0 s: n={len(long)} (of {len(j)})")
    if not len(long):
        continue
    sl = long[long.command.isin(PARAMETRIC)]
    fx = long[~long.command.isin(PARAMETRIC)]
    print(f"  slotted (slot = last words, tail kept -> safe): {len(sl)}")
    print(f"  fixed   (front cut removes intent words):      {len(fx)}")
    for cmd, k in long.command.value_counts().head(8).items():
        tag = "slot-in-tail" if cmd in PARAMETRIC else "WATCH"
        print(f"    {cmd:18s} {k:4d}  {tag}")
    print()