"""Dataset composition: clips per command class + per slot value.
Usage (HPC): python tools/class_counts.py [--manifest path]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from commands import CLASSES_10  # noqa: E402
from slots import PARAMETRIC, extract_slot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="/home/jan.rhey.lagana/vcm/data/raw_v2/"
                            "manifest.jsonl")
    args = ap.parse_args()
    man = [json.loads(l) for l in open(args.manifest)]
    per_cls = Counter(m["class"] for m in man)
    per_split_cls = {}
    for m in man:
        per_split_cls.setdefault(m["split"], Counter())[m["class"]] += 1
    splits = sorted(per_split_cls)
    print(f"total clips: {len(man)}  splits: "
          f"{ {s: sum(per_split_cls[s].values()) for s in splits} }")
    print("\n=== command class (clips) ===")
    print(f"  {'class':18s} {'total':>6s}"
          + "".join(f" {s:>8s}" for s in splits))
    for c in CLASSES_10:
        print(f"  {c:18s} {per_cls[c]:6d}"
              + "".join(f" {per_split_cls[s].get(c, 0):8d}"
                        for s in splits))
    print("\n=== slot value (clips; parametric classes) ===")
    for c in PARAMETRIC:
        cnt = Counter()
        cnt_split = {}
        for m in man:
            if m["class"] == c:
                s_ = extract_slot(c, m["phrase"])
                cnt[s_] += 1
                cnt_split.setdefault(s_, Counter())[m["split"]] += 1
        order = sorted(cnt, key=lambda s: (-cnt[s], str(s)))
        print(f"\n  {c}: {sum(cnt.values())} clips, {len(cnt)} values")
        for s in order:
            cs = cnt_split[s]
            print(f"    {str(s):20s} {cnt[s]:5d} {cs.get('train', 0):5d} "
                  f"{cs.get('val', 0):5d} {cs.get('test', 0):5d}")
    voices = {}
    for m in man:
        voices.setdefault(m["split"], set()).add(m["voice"])
    print("\n=== voices per split ===")
    for s in splits:
        print(f"  {s}: {len(voices[s])} voices")


if __name__ == "__main__":
    main()