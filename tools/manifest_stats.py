"""Frozen-corpus stats for the submission slide / dataset card.

Reads the frozen split exports data/manifests/manifest_{split}.csv (all arrow
columns except audio) and prints: per-split + total clip counts, total audio
duration (sum of duration_s), unique speakers (speaker_id), per-source counts,
and the label inventories. Stdlib csv only — no pandas needed.

Run on the box that has the manifests (HPC, from ~/vcm_v3):
  .venv/bin/python tools/manifest_stats.py
Paste the output into SUBMISSION.md / slides_data.json (the "pending" numbers).
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

SPLITS = ["train", "test", "holdout", "numerals"]
MAIN = ["train", "test", "holdout"]


def rows(path: Path):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def dur(r):
    try:
        v = r.get("duration_s")
        if v in (None, "", "None", "nan"):
            return 0.0
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default="data/manifests")
    args = ap.parse_args()
    d = Path(args.manifest_dir)
    per = {sp: rows(d / f"manifest_{sp}.csv") for sp in SPLITS}

    grand_n = grand_dur = 0
    print("== per split ==")
    for sp in SPLITS:
        rs = per[sp]
        n = len(rs)
        ds = sum(dur(r) for r in rs)
        grand_n += n
        grand_dur += ds
        oos = sum(1 for r in rs if r.get("out_of_scope") == "1"
                  or r.get("command") == "OUT_OF_SCOPE")
        print(f"  {sp:9s} n={n:7d}  dur={ds / 3600:8.2f} h  OOS={oos}")
    print(f"  {'TOTAL':9s} n={grand_n:7d}  dur={grand_dur / 3600:8.2f} h "
          f"({grand_dur:.0f} s)")

    for label, sps in (("main (train+test+holdout)", MAIN), ("all splits", SPLITS)):
        spk = set()
        for sp in sps:
            for r in per[sp]:
                s = (r.get("speaker_id") or "").strip()
                if s and s.lower() not in ("unknown",):
                    spk.add(s)
        print(f"  unique speakers [{label}]: {len(spk)}")

    src = Counter()
    for sp in SPLITS:
        for r in per[sp]:
            src[(r.get("source") or "?").strip()] += 1
    print("== per source (all splits) ==")
    for s, c in src.most_common():
        print(f"  {s:34s} {c}")

    cmds = Counter()
    slots = set()
    for sp in MAIN:
        for r in per[sp]:
            cmds[(r.get("command") or "?").strip()] += 1
            sv = (r.get("slot_value") or "").strip()
            if sv:
                slots.add(sv)
    print("== labels (main splits) ==")
    print(f"  distinct command classes: {len(cmds)}  | slot values seen: {len(slots)}")
    for c, n in sorted(cmds.items(), key=lambda kv: -kv[1]):
        print(f"    {c:18s} {n}")


if __name__ == "__main__":
    main()
