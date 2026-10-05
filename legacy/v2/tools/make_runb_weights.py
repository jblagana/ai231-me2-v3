"""Run B slot-oversampling weights from an eval slot matrix.

weight(class, slot) = min(WCAP, 1 + K * row_err_rate) for rows with n >= MIN_N
(rows with n below MIN_N are skipped — not present on EVAL-SYN in practice).

Usage: python tools/make_runb_weights.py [eval.json] [out.json]
"""
import json
import sys
from pathlib import Path

K = 2.0      # weight slope vs row error rate
WCAP = 4.0   # max weight
MIN_N = 50   # min row count to trust the error rate


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1
               else "data/eval_v2a_bcresnet_10c_eval.json")
    out = Path(sys.argv[2] if len(sys.argv) > 2
               else "data/runb_slot_weights.json")
    mat = json.loads(src.read_text())["slot_matrix"]
    weights = {}
    for cls, m in mat.items():
        wcls = {}
        for t, row in m.items():
            n = sum(row.values())
            if n < MIN_N:
                continue
            err = 1.0 - row.get(t, 0) / n
            wcls[t] = round(min(WCAP, 1.0 + K * err), 3)
        weights[cls] = wcls
    out.write_text(json.dumps(weights, indent=1))
    n_up = sum(1 for c in weights.values() for v in c.values() if v > 1.0)
    print(f"weights -> {out}  (K={K}, cap={WCAP}, min_n={MIN_N})")
    print(f"slot values upweighted (>1.0): {n_up}")
    for cls, c in weights.items():
        top = sorted(c.items(), key=lambda kv: -kv[1])[:4]
        print(f"  {cls:16s} " + ", ".join(f"{k}={v}" for k, v in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())