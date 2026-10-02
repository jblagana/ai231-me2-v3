"""Verify the master Gold Dataset (airimonda/ai231-me2-voice-commands).

Runs on HPC (me2 conda env). Loads the dataset via `datasets`, writes
TEXT-ONLY manifests (audio column excluded) to data/manifests/, and
checks:
  - splits present + sizes
  - 19-intent command coverage per split
  - speaker disjointness (train vs test / holdout)
  - per-class balance in train (command + bucket)
  - slot-value class count (schema locks 18; manifest may carry ~93)
  - out_of_scope / empty-transcript (background noise) counts
  - holdout variation coverage (fixed 3-variation demo benchmark)

Usage (on HPC, from ~/vcm_v3):
  ~/.conda/envs/me2/bin/python tools/verify_gold_dataset.py [out_dir]
Env: HF_DATASET overrides the dataset id (default: the group's HF repo).
"""
import os
import sys

import pandas as pd
from datasets import load_dataset

SRC = os.environ.get(
    "HF_DATASET", "airimonda/ai231-me2-voice-commands")

INTENTS = [
    "PLAY_MUSIC", "VOLUME_UP", "VOLUME_DOWN", "NEXT", "PAUSE", "STOP",
    "LIGHT_ON", "LIGHT_OFF", "BRIGHTNESS", "COLOR",
    "TEMPERATURE", "WEATHER", "TIME",
    "TIMER", "ALARM",
    "CALL", "MESSAGE",
    "CREATE_REMINDER", "LIST_REMINDERS",
]

SCHEMA_SLOTS = {
    "TIMER": ["10 seconds", "30 seconds", "1 minute"],
    "ALARM": ["6 AM", "8 AM", "9 PM"],
    "TEMPERATURE": ["18 degrees", "22 degrees", "26 degrees"],
    "BRIGHTNESS": ["20 percent", "60 percent", "100 percent"],
    "COLOR": ["red", "blue", "green"],
    "CREATE_REMINDER": ["drink water", "study", "exercise"],
}


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "data/manifests"
    os.makedirs(out, exist_ok=True)
    print(f"loading {SRC} ...")
    ds = load_dataset(SRC)
    print("splits:", {k: len(v) for k, v in ds.items()})

    frames = {}
    for name in ds:
        cols = [c for c in ds[name].column_names if c != "audio"]
        df = ds[name].select_columns(cols).to_pandas()
        frames[name] = df
        path = os.path.join(out, f"manifest_{name}.csv")
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows, {len(cols)} cols)")

    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        if not cond:
            ok = False
        print(f"[{'PASS' if cond else 'FAIL'}] {label} {detail}")

    need = {"train", "test", "holdout"}
    check("required splits present", need <= set(frames),
          f"got={sorted(frames)}")
    if not need <= set(frames):
        print("VERDICT: FAIL (missing splits)")
        return 1

    train, test, hold = frames["train"], frames["test"], frames["holdout"]

    for name, df in (("train", train), ("test", test), ("holdout", hold)):
        cmds = set(df["command"].dropna().unique())
        missing = [c for c in INTENTS if c not in cmds]
        check(f"{name}: all 19 intents present", not missing,
              f"n_commands={len(cmds)} missing={missing}")

    tr = set(train["speaker_id"].dropna().unique())
    te = set(test["speaker_id"].dropna().unique())
    ho = set(hold["speaker_id"].dropna().unique())
    check("train speakers", True, f"n={len(tr)}")
    check("test speakers", True, f"n={len(te)}")
    check("holdout speakers", True, f"n={len(ho)}")
    check("train ∩ test speakers = ∅", not (tr & te),
          f"overlap={sorted(tr & te)[:5]}")
    check("train ∩ holdout speakers = ∅", not (tr & ho),
          f"overlap={sorted(tr & ho)[:5]}")
    check("test ∩ holdout speakers = ∅", not (te & ho),
          f"overlap={sorted(te & ho)[:5]}")

    print("\ntrain per-class counts (command):")
    print(train["command"].value_counts().to_string())
    if "bucket" in train.columns:
        print("\ntrain per-bucket counts (top 30):")
        print(train["bucket"].value_counts().head(30).to_string())

    sv = set(train["slot_value"].dropna().unique()) | \
        set(test["slot_value"].dropna().unique())
    schema_vals = {v.lower() for vs in SCHEMA_SLOTS.values() for v in vs}
    manifest_norm = {str(v).lower() for v in sv}
    in_schema = schema_vals & manifest_norm
    check("slot-value classes", True,
          f"n={len(sv)} | schema-18 present (any case)={len(in_schema)}"
          f"/18 | beyond-schema={len(manifest_norm - schema_vals)}")

    oos = int(train["out_of_scope"].sum()) if "out_of_scope" in train else -1
    empty = int((train["transcript"].fillna("").str.strip() == "").sum())
    check("out_of_scope count", True, f"train out_of_scope={oos}")
    check("empty-transcript (noise) count", True, f"train={empty}")

    if "duration_s" in train.columns:
        d = train["duration_s"]
        check("duration sanity", d.max() <= 12.0,
              f"min={d.min():.2f}s median={d.median():.2f}s max={d.max():.2f}s"
              " (3.0s window => long clips will be truncated at feature time)")

    hv = hold.groupby(["command", "variation"]).size()
    n_pairs = len(hv)
    per_cmd = hold.groupby("command")["variation"].nunique()
    check("holdout covers all 19 intents",
          set(hold["command"].dropna().unique()) >= set(INTENTS),
          f"n={len(hold)}")
    # true shape (verified 2026-10-02): fixed intents carry their 3
    # variations; slotted intents carry all 9 rendered template x value
    # phrases; OUT_OF_SCOPE rows have no variation
    fixed_expect = (per_cmd.reindex(
        [c for c in INTENTS if c not in SCHEMA_SLOTS]) == 3)
    slotted_expect = (per_cmd.reindex(
        [c for c in INTENTS if c in SCHEMA_SLOTS]) == 9)
    oos_rows = int((hold["command"] == "OUT_OF_SCOPE").sum())
    check("holdout: 3 variations per fixed intent, 9 per slotted",
          bool(fixed_expect.all() and slotted_expect.all()),
          f"pairs={n_pairs} OUT_OF_SCOPE_rows={oos_rows} "
          f"fixed={fixed_expect.to_dict()} "
          f"slotted={slotted_expect.to_dict()}")
    check("holdout size == 196 (locked demo benchmark)",
          len(hold) == 196, f"n={len(hold)}")

    if "numerals" in frames:
        print(f"\nnumerals split: {len(frames['numerals'])} rows "
              "(number-robustness bench, kept separate)")

    print("\nVERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
