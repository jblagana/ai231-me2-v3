"""Detail pass on the master Gold Dataset manifests (post-verification).

HPC (from ~/vcm_v3):  .venv/bin/python tools/inspect_manifests.py
Prints: slot-value classes vs the 18-value schema, holdout variation
shape, beyond-canonical buckets per slotted intent, test balance,
out_of_scope / noise samples.
"""
from pathlib import Path

import pandas as pd

M = Path("data/manifests")
SCHEMA = {
    "TIMER": ["10 seconds", "30 seconds", "1 minute"],
    "ALARM": ["6 AM", "8 AM", "9 PM"],
    "TEMPERATURE": ["18 degrees", "22 degrees", "26 degrees"],
    "BRIGHTNESS": ["20 percent", "60 percent", "100 percent"],
    "COLOR": ["red", "blue", "green"],
    "CREATE_REMINDER": ["drink water", "study", "exercise"],
}

train = pd.read_csv(M / "manifest_train.csv")
test = pd.read_csv(M / "manifest_test.csv")
hold = pd.read_csv(M / "manifest_holdout.csv")

print("=== SPLIT SIZES ===")
print(f"train={len(train)} test={len(test)} holdout={len(hold)}")

print("\n=== SLOT VALUES (train+test) ===")
sv = pd.concat([train["slot_value"], test["slot_value"]]).dropna().astype(str)
vals = sorted(sv.unique())
print(f"distinct: {len(vals)}")
schema_flat = {v.lower() for vs in SCHEMA.values() for v in vs}
for v in vals:
    tag = "schema       " if v.lower() in schema_flat else "BEYOND-SCHEMA"
    print(f"  [{tag}] {v!r}  n={int((sv == v).sum())}")
missing = schema_flat - {v.lower() for v in vals}
print("schema values NOT in manifest:",
      sorted(missing) if missing else "none")

print("\n=== HOLDOUT VARIATION SHAPE ===")
g = hold.groupby("command")["variation"].nunique()
print(g.to_string())
per_row = hold["command"].map(g)
single = hold[per_row == 1]
cols = [c for c in ["command", "variation", "transcript", "speaker_id",
                    "source", "dataset"] if c in hold.columns]
print("\nsingle-variation clips:")
print(single[cols].to_string())
print("\nholdout per-command:")
print(hold["command"].value_counts().to_string())

print("\n=== TRAIN BUCKETS PER SLOTTED INTENT ===")
for cls in SCHEMA:
    sub = train[train["command"] == cls]
    vc = sub["bucket"].value_counts()
    print(f"{cls}: n={len(sub)} distinct_buckets={len(vc)}")
    if len(vc) > 9:
        print(vc.to_string())

print("\n=== TEST PER-CLASS ===")
print(test["command"].value_counts().to_string())

print("\n=== OUT_OF_SCOPE / NOISE (train) ===")
oos = train[train["out_of_scope"] == 1]
print(f"out_of_scope={len(oos)}")
print(oos["transcript"].fillna("(empty)").head(12).to_string())

print("\n=== COLUMN INVENTORY ===")
print(list(train.columns))
