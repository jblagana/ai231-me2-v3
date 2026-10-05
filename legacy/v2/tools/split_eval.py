"""Split the eval split of raw_v2's manifest into val + test (10/10 by
voice, deterministic). Boss-ratified protocol 2026-10-02: val =
per-epoch eval + checkpoint selection; test = sealed, one-shot per
final model.

Voice-level split (NOT clip-level): val and test stay speaker-disjoint
FROM EACH OTHER, so selection never hears a test speaker. Deterministic:
sorted(eval_voices) first half -> val, second half -> test. Idempotent:
re-derives from manifest.orig.jsonl on re-runs.

Usage (HPC): python tools/split_eval.py --root ~/vcm/data/raw_v2
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    mf = root / "manifest.jsonl"
    orig = root / "manifest.orig.jsonl"

    if mf.exists() and not orig.exists():
        # first run: back up the untouched manifest before rewriting
        orig.write_text(mf.read_text(encoding="utf-8-sig"), encoding="utf-8")
    src = orig if orig.exists() else mf
    if not src.exists():
        print(f"no manifest at {src}")
        return 2

    entries = [json.loads(l) for l in
               src.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    eval_voices = sorted({e["voice"] for e in entries if e["split"] == "eval"})
    if not eval_voices:
        bad = {e["split"] for e in entries} - {"train", "val", "test"}
        if bad:
            print(f"unexpected split values: {sorted(bad)}")
            return 3
        print("already split; nothing to do")
        return 0
    if len(eval_voices) % 2:
        print(f"odd eval voice count {len(eval_voices)} — refusing 10/10")
        return 3
    val_voices = set(eval_voices[: len(eval_voices) // 2])
    test_voices = set(eval_voices[len(eval_voices) // 2:])
    for e in entries:
        if e["split"] == "eval":
            e["split"] = "val" if e["voice"] in val_voices else "test"
    mf.write_text("\n".join(json.dumps(e) for e in entries) + "\n",
                  encoding="utf-8")
    print(f"val  voices ({len(val_voices)}): {sorted(val_voices)}")
    print(f"test voices ({len(test_voices)}): {sorted(test_voices)}")
    for s in ("train", "val", "test"):
        c = Counter(e["class"] for e in entries if e["split"] == s)
        print(f"{s:6s} n={sum(c.values()):6d}  "
              + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print(f"wrote {mf}  (original kept at {orig})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())