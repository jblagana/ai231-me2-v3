"""Print slot confusion matrices from an eval_*.json (rows=true, cols=pred).

Full grid for classes with <=13 values; worst-error rows for big vocabs
(set_timer 24, set_alarm 48).

Usage: python tools/print_slot_matrix.py [path/to/eval_*.json]
"""
import json
import sys
from pathlib import Path

FULL_MAX = 13


def _fmt(m, t, p):
    return m.get(t, {}).get(p, 0)


def _row_tot(m, t, rows):
    return sum(_fmt(m, t, p) for p in rows)


def _print_full(cls, m, rows):
    print(f"\n=== {cls} ({len(rows)} values) rows=true cols=pred ===")
    w = max(6, max(len(r) for r in rows) + 2)
    print(" " * (w + 2) + "".join(f"{r[:8]:>9s}" for r in rows))
    for t in rows:
        cells = "".join(f"{_fmt(m, t, p):>9d}" for p in rows)
        tot = _row_tot(m, t, rows)
        ok = _fmt(m, t, t)
        print(f"{t[:w]:<{w}} {cells}   acc {ok / tot:.2f} (n={tot})")


def _print_worst(cls, m, rows, k):
    off = []
    for t in rows:
        tot = _row_tot(m, t, rows)
        off.append((tot - _fmt(m, t, t), t, tot))
    off.sort(reverse=True)
    print(f"\n=== {cls} ({len(rows)} values) — worst {k} rows (off-diag) ===")
    for bad, t, tot in off[:k]:
        preds = sorted(((_fmt(m, t, p), p) for p in rows if p != t),
                       reverse=True)[:6]
        top = ", ".join(f"{p}x{n}" for n, p in preds)
        print(f"  {t[:20]:20s} err {bad:>4d}/{tot}  -> {top}")


def main():
    p = Path(sys.argv[1] if len(sys.argv) > 1
             else "data/eval_v2a_bcresnet_10c_eval.json")
    rep = json.loads(p.read_text())
    mat = rep.get("slot_matrix", {})
    if not mat:
        print("no slot_matrix in", p)
        return 1
    for cls, m in mat.items():
        rows = list(m.keys())
        if len(rows) <= FULL_MAX:
            _print_full(cls, m, rows)
        else:
            _print_worst(cls, m, rows, 10)
    return 0


if __name__ == "__main__":
    sys.exit(main())