"""Per-class sample counts for the two heads (train + test).

HPC (from ~/vcm_v3):  .venv/bin/python tools/slot_counts.py
Command head: 19 intents (+ OUT_OF_SCOPE). Slot heads: 6 heads x 3 values;
each head is supervised by its intent's clips.
"""
from pathlib import Path

import pandas as pd

M = Path("data/manifests")
SLOTTED = ["TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR",
           "CREATE_REMINDER"]

train = pd.read_csv(M / "manifest_train.csv")
test = pd.read_csv(M / "manifest_test.csv")


def command_counts(df, name):
    vc = df["command"].value_counts()
    print(f"=== COMMAND HEAD — {name} (n={len(df)}) ===")
    fixed = [c for c in vc.index if c not in SLOTTED and c != "OUT_OF_SCOPE"]
    for c in fixed + SLOTTED + (["OUT_OF_SCOPE"] if "OUT_OF_SCOPE" in vc else []):
        if c in vc.index:
            print(f"  {c:16s} {int(vc[c]):6d}")
    print()


command_counts(train, "TRAIN")
command_counts(test, "TEST")

for df, name in ((train, "TRAIN"), (test, "TEST")):
    print(f"=== SLOT HEADS — {name} (samples = intent clips) ===")
    for cls in SLOTTED:
        sub = df[df["command"] == cls]
        vc = sub["slot_value"].value_counts()
        n_nan = int(sub["slot_value"].isna().sum())
        print(f"  {cls:16s} head samples={len(sub):5d}  "
              f"(per value: {dict(vc.astype(int))}  NaN={n_nan})")
    print()
