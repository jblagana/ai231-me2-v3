"""120-ep divergence-monitor summary (non-decision; writeup + budgets).

Boss call 10-03: map where val diverges per E/F recipe. Val noise at
10 voices is +/-0.3-0.5 pt, so "divergence" = train loss still falling
while val stops improving (last-improvement epoch) and wobbles after.
Monitors are fresh draws: budget effects (e60 -> e120) are read WITHIN
a run; cross-recipe comparison is only to +-0.3-0.5 pt.

Usage (HPC): python tools/analyze_mon.py [run ...]
"""
import json
import sys
from pathlib import Path

DEFAULT = ["mon_v2e_120", "mon_v2e2_120", "mon_v2f_cmd_120",
           "mon_v2f_slot_120"]


def main():
    runs = sys.argv[1:] or DEFAULT
    root = Path(__file__).resolve().parent.parent / "runs"
    out = []
    for name in runs:
        h = json.load(open(root / name / "history.json"))
        n = len(h)
        vk = "eval_cmd_acc" if "cmd" in name else "eval_slot_acc"
        tl = [r["train_loss"] for r in h]
        vv = [r[vk] for r in h]
        ep = [r["epoch"] for r in h]
        best = max(vv)
        best_ep = vv.index(best) + 1
        last_imp = 1
        for i in range(1, n):
            if vv[i] > max(vv[:i]):
                last_imp = i + 1

        def val_at(k):
            return vv[min(k - 1, n - 1)]

        def tl_at(k):
            return tl[min(k - 1, n - 1)]

        row = {
            "run": name, "n": n, "val_metric": vk,
            "train_loss": {f"e{k}": round(tl_at(k), 4)
                           for k in (1, 30, 60, 120, 300, 500, n)},
            "val": {f"e{k}": round(val_at(k), 4)
                    for k in (60, 120, 300, 500, n)},
            "val_best": round(best, 4), "val_best_ep": best_ep,
            "val_last_improvement_ep": last_imp,
            "late_gain_e60_to_end": round(val_at(n) - val_at(60), 4)
            if n > 60 else None,
            "late_gain_e120_to_e500": round(val_at(500) - val_at(120), 4)
            if n > 500 else None,
            "late_gain_e500_to_end": round(val_at(n) - val_at(500), 4)
            if n > 500 else None,
        }
        out.append(row)
        print(f"=== {name} (val: {vk}, n={n}) ===")
        print("  train_loss " + "  ".join(
            f"e{k}={tl_at(k):.3f}" for k in (1, 30, 60, 120, 300, 500, n)))
        print("  val        " + "  ".join(
            f"e{k}={val_at(k):.3f}" for k in (60, 120, 300, 500, n)))
        if row["late_gain_e120_to_e500"] is not None:
            late = (f" | e120->e500 {row['late_gain_e120_to_e500']:+.3f}"
                    f" | e500->e{n} {row['late_gain_e500_to_end']:+.3f}")
        else:
            late = (f" | late gain e60->e{n} "
                    f"{row['late_gain_e60_to_end']:+.3f}"
                    if row["late_gain_e60_to_end"] is not None else "")
        print(f"  best {best:.3f} @ e{best_ep} | "
              f"last improvement e{last_imp}{late}")
        print("  val curve: " + " ".join(
            f"e{ep[i]}:{vv[i]:.3f}" for i in range(0, n, max(1, n // 40))))
        print()
    jf = Path(__file__).resolve().parent / "mon_summary.json"
    jf.write_text(json.dumps(out, indent=1))
    print(f"json -> {jf}")


if __name__ == "__main__":
    main()