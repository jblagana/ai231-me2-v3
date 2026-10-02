"""Slot-tail geometry check (pre-run-1 design probe, 10-02).

Question: the slot heads max-pool the last ~1.3 s of the 3.0 s window, but
the frozen live geometry (demo silence stop) leaves a 1.0 s trailing-silence
tail on captured clips. How much SLOT content is actually inside the slot
head's read region, per clip?

Window end = min(d, t1 + 1.0); read region = last 1.28 s (bcresnet,
BC_SLOT_TAIL_CELLS=64 x 20 ms) / 1.33 s (v2cnn, 8 x 167 ms).
Slot content in read region = max(0, read - tail) where
tail = min(d - t1, 1.0) (the silence after speech end, capped by the window).

For slotted clips (the 6 parametric intents) in train/test/holdout, report
the tail distribution and the implied in-read-region slot content.
"""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAR = {"TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER"}
READ_S = 1.28  # bcresnet slot read region (64 cells x 20 ms)


def main() -> None:
    for split in ("train", "test", "holdout"):
        p = ROOT / "data" / "vad_audit" / f"clip_vad_{split}.csv"
        tails = Counter()
        n = 0
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r["command"] not in PAR:
                    continue
                n += 1
                tail = min(float(r["tail_sil"]), 1.0)
                if tail < 0.3:
                    tails["a <0.30 s (full slot in read region)"] += 1
                elif tail < 0.6:
                    tails["b 0.30-0.60 s"] += 1
                elif tail < 0.95:
                    tails["c 0.60-0.95 s"] += 1
                else:
                    tails["d >=0.95 s (live/holdout geometry)"] += 1
        print(f"== {split}: {n} slotted clips; slot content in the 1.28 s "
              f"slot read region = max(0, 1.28 - tail) ==")
        for k in sorted(tails):
            band, lo = k.split(" ", 1)[0], None
            print(f"  {k}: {tails[k]} ({100 * tails[k] / n:.1f}%)")
        print()


if __name__ == "__main__":
    main()
