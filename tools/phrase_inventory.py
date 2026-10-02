"""Phrase inventory: unique phrases (transcript / variation) per command
across the manifests, checked against the locked 93-phrase phrasebook.

HPC (from ~/vcm_v3):  .venv/bin/python tools/phrase_inventory.py
Output: stdout summary + data/phrase_inventory.csv (full table)
"""
import sys
from pathlib import Path

import pandas as pd

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo / "src"))   # flat imports inside commands.py
sys.path.insert(0, str(repo))
from commands import PHRASES  # noqa: E402  (93-phrase phrasebook + 20 classes)
from slots import PARAMETRIC  # noqa: E402

pd.set_option("display.width", 200)


def norm(s):
    """Loader normalization for phrasebook membership (casefold + :00)."""
    return " ".join(str(s).casefold().split()).replace(":00", "")


BOOK = {c: {norm(p) for p in ps} for c, ps in PHRASES.items()}

frames = []
for split in ("train", "test", "holdout", "numerals"):
    m = pd.read_csv(repo / f"data/manifests/manifest_{split}.csv",
                    low_memory=False)
    m = m[m.transcript.notna()]
    g = (m.groupby(["command", "transcript"]).size()
           .reset_index(name="n"))
    g["split"] = split
    frames.append(g)
all_df = pd.concat(frames, ignore_index=True)

# pivot: rows = (command, transcript), cols = split counts
piv = (all_df.pivot_table(index=["command", "transcript"], columns="split",
                          values="n", fill_value=0)
           .reset_index())
for c in ("train", "test", "holdout", "numerals"):
    if c not in piv:
        piv[c] = 0
piv = piv[["command", "transcript", "train", "test", "holdout", "numerals"]]
def status(cmd, t):
    if str(cmd) == "OUT_OF_SCOPE":
        return "oos(n/a)"
    return "yes" if norm(t) in BOOK.get(str(cmd), set()) else "no"


piv["in_phrasebook"] = [status(c, t)
                        for c, t in zip(piv.command, piv.transcript)]
out = repo / "data/phrase_inventory.csv"
piv.sort_values(["command", "transcript"]).to_csv(out, index=False)
print(f"wrote {out}  ({len(piv)} unique (command, transcript) pairs)\n")

# per-command summary
print(f"{'COMMAND':18s} {'uniq':>5s} {'train':>7s} {'test':>6s} {'hold':>5s} "
      f"{'numer':>7s} {'off-book':>9s}")
for cmd, g in piv.groupby("command", sort=False):
    off = int((g.in_phrasebook == "no").sum())
    print(f"{str(cmd)[:18]:18s} {len(g):5d} {int(g.train.sum()):7d} "
          f"{int(g.test.sum()):6d} {int(g.holdout.sum()):5d} "
          f"{int(g.numerals.sum()):7d} {off:9d}")

print("\n=== OFF-PHRASEBOOK phrases (non-OOS) ===")
off = piv[piv.in_phrasebook == "no"]
if len(off) == 0:
    print("(none — dataset phrases == phrasebook, after :00/case norm)")
else:
    for _, r in off.iterrows():
        print(f"  {r.command:18s} {r.transcript!r}  "
              f"t={r.train} te={r.test} h={r.holdout}")

print("\n=== OUT_OF_SCOPE unique transcripts ===")
oos = piv[piv.command == "OUT_OF_SCOPE"].sort_values("train", ascending=False)
print(f"n_unique = {len(oos)}  (clips: train {int(oos.train.sum())} / "
      f"test {int(oos.test.sum())} / holdout {int(oos.holdout.sum())})")
for _, r in oos.head(40).iterrows():
    print(f"  {int(r.train + r.test + r.holdout):5d}  {r.transcript!r}")
if len(oos) > 40:
    print(f"  ... and {len(oos) - 40} more (see CSV)")

print("\n=== NUMERALS (report-only split) ===")
num = piv[piv.numerals > 0]
print(f"n_unique = {len(num)}  total clips = {int(piv.numerals.sum())}")
samp = num.sort_values("numerals", ascending=False).head(10)
for _, r in samp.iterrows():
    print(f"  {int(r.numerals):8d}  {str(r.command)[:14]:14s} "
          f"{r.transcript!r}")