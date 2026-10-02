"""Slot-value duration probe (pre-run-1, 10-02).

The slot head's read region must cover the slot value under the LIVE
geometry (1.0 s trailing silence). This script measures how long the slot
values actually are, from VAD spans of synthetic TTS clips where the TTS
rate is consistent:

- per (command, variation, slot_value): mean span of the TTS clips
- fixed-intent TTS spans: the rate anchor (known phrases, known char counts)

Join: manifest_train (command/variation/slot_value/source) x vad CSV (span)
by file basename. Real-data sources are excluded (accent/rate variance).
"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
PAR = {"TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR",
       "CREATE_REMINDER"}
REAL = ("real", "Fluent", "Common", "Voice", "scv2", "slurp", "group")

man = {}
with open(ROOT / "data" / "manifests" / "manifest_train.csv", newline="") as f:
    for r in csv.DictReader(f):
        man[r["file"].rsplit("/", 1)[-1]] = (
            r["command"], r["variation"], r["slot_value"], r["source"])

groups = defaultdict(list)
fixed = defaultdict(list)
srcs = defaultdict(int)
with open(ROOT / "data" / "vad_audit" / "clip_vad_train.csv", newline="") as f:
    for r in csv.DictReader(f):
        b = r["file"].rsplit("/", 1)[-1]
        if b not in man:
            continue
        cmd, var, sv, src = man[b]
        srcs[src] += 1
        if any(x.lower() in src.lower() for x in REAL):
            continue
        if not r["span"]:
            continue
        span = float(r["span"])
        if cmd in PAR and sv:
            groups[(cmd, var, sv)].append(span)
        elif cmd not in PAR and not sv:
            fixed[cmd].append(span)

print("== train sources (vad join) ==")
for k, v in sorted(srcs.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

print("\n== fixed-intent TTS span (rate anchor) ==")
for c in sorted(fixed):
    print(f"  {c:18s} mean span {mean(fixed[c]):.2f} s  (n={len(fixed[c])})")

print("\n== parametric: span by (variation, slot value) — TTS/synthetic only ==")
for cmd in sorted(PAR):
    print(f"  {cmd}")
    keys = sorted({(v, s) for (c, v, s) in groups if c == cmd})
    for v, s in keys:
        spans = groups[(cmd, v, s)]
        print(f"    v{v} | {s:14s} mean span {mean(spans):.2f} s  (n={len(spans)})")
    print()
