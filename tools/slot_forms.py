"""Slot surface-form analysis — quantifies what the loader must normalize.

Counts, per parametric intent: slotted clips, distinct raw slot_value
strings, clips matching the 18-value canonical vocab as-is vs only after
normalization (casefold + \":00\" strip), and unmatched (NaN-slot,
command-only supervision).

HPC (from ~/vcm_v3):  .venv/bin/python tools/slot_forms.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.slots import SLOT_VOCAB

repo = Path(__file__).resolve().parent.parent
CANON = {c: {v.lower() for v in vals} for c, vals in SLOT_VOCAB.items()}


def canon(raw):
    """The normalization src/dataset.py will apply."""
    s = " ".join(str(raw).casefold().split())
    return s.replace(":00", "").replace("  ", " ")


for split in ("train", "test", "holdout", "numerals"):
    m = pd.read_csv(repo / f"data/manifests/manifest_{split}.csv")
    sl = m[m.slot_value.notna() & (m.slot_value.astype(str).str.strip() != "")]
    print(f"=== {split}: {len(sl)} slotted clips ===")
    if not len(sl):
        continue
    as_is = sl.slot_value.str.casefold().isin(
        [v for s in sl.command.unique()
         for v in CANON.get(s.upper(), set())]).sum()
    after = sl.slot_value.map(canon).isin(
        [v for s in sl.command.unique()
         for v in CANON.get(s.upper(), set())]).sum()
    # per-intent detail
    for cmd in sorted(sl.command.unique()):
        g = sl[sl.command == cmd]
        forms = g.slot_value.value_counts()
        ok = CANON.get(cmd.upper(), set())
        n_as = int(g.slot_value.str.casefold().isin(ok).sum())
        n_after = int(g.slot_value.map(canon).isin(ok).sum())
        print(f"  {cmd:16s} n={len(g):5d}  raw forms={len(forms):3d}  "
              f"canonical-as-is={n_as:5d}  after-norm={n_after:5d}")
        for form, k in forms.items():
            flag = "" if str(form).casefold() in ok else (
                "OK(norm)" if canon(form) in ok else "!!UNMATCHED!!")
            print(f"      {k:5d}  {form!r}  {flag}")
    print(f"  TOTAL: as-is {as_is}  after-norm {after}  "
          f"unmatched {len(sl) - after}  -> without normalization "
          f"{len(sl) - as_is} clips lose slot supervision")
    print()