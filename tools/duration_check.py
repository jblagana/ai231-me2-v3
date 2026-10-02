"""Duration audit vs the locked 3.0 s feature window (all splits).

HPC (from ~/vcm_v3):  .venv/bin/python tools/duration_check.py
For every split: size, percentiles, share <= 3.0 s, and a per-source
breakdown of clips that would be truncated at feature time.
"""
from pathlib import Path

import pandas as pd

M = Path("data/manifests")
WIN = 3.0

for name in ("train", "test", "holdout", "numerals"):
    df = pd.read_csv(M / f"manifest_{name}.csv")
    d = df["duration_s"]
    long_ = df[d > WIN]
    n_long = len(long_)
    print(f"=== {name.upper()} (n={len(df)}) ===")
    print(f"  min {d.min():.2f}  p50 {d.median():.2f}  "
          f"p90 {d.quantile(0.90):.2f}  p95 {d.quantile(0.95):.2f}  "
          f"p99 {d.quantile(0.99):.2f}  max {d.max():.2f}")
    print(f"  <= {WIN}s: {len(df) - n_long} "
          f"({100 * (len(df) - n_long) / len(df):.1f}%)   "
          f"> {WIN}s: {n_long} ({100 * n_long / len(df):.1f}%)")
    if n_long:
        by_src = long_.groupby("source").size().sort_values(ascending=False)
        print("  > 3.0 s by source:")
        for src, n in by_src.items():
            mx = long_.loc[long_["source"] == src, "duration_s"].max()
            print(f"    {str(src):30s} {n:5d}  (max {mx:.2f} s)")
        cut = d[d > WIN] - WIN
        print(f"  truncation on long clips: mean cut {cut.mean():.2f} s, "
              f"max cut {cut.max():.2f} s")
    # short-clip check: slot read region is ~2.0 s; clips shorter than that
    # have it cover the whole utterance (fine, but worth counting)
    n_short = int((d < 2.0).sum())
    print(f"  < 2.0 s (shorter than slot read region): {n_short} "
          f"({100 * n_short / len(df):.1f}%)")
    print()
