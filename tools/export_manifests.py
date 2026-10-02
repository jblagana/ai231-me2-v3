"""Re-export the frozen manifests from the JFS HF arrow cache.

FROZEN SPLIT EXPORT (instr 18 re-freeze 2026-10-02): one CSV per split, all
arrow columns except `audio`, column order as in the arrow schema.
HPC (from ~/vcm_v3):  .venv/bin/python tools/export_manifests.py
"""
import argparse
import sys
from pathlib import Path

import pyarrow as pa

DEFAULT_ARROW = ("/data/ai231/airimonda___ai231-me2-voice-commands/"
                 "default/0.0.0")
SPLITS = ["train", "test", "holdout", "numerals"]


def load_split(arrow_dir: Path, split: str):
    fs = sorted(arrow_dir.rglob(f"*{split}*.arrow"))
    if not fs:
        sys.exit(f"no arrow files for split {split} under {arrow_dir}")
    tables = []
    for f in fs:
        with pa.memory_map(str(f), "r") as src:
            tables.append(pa.ipc.open_stream(src).read_all())
    return pa.concat_tables(tables)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow-dir", default=DEFAULT_ARROW)
    ap.add_argument("--out", default="data/manifests")
    args = ap.parse_args()
    arrow_dir = Path(args.arrow_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        t = load_split(arrow_dir, split)
        cols = [c for c in t.column_names if c != "audio"]
        df = t.select(cols).to_pandas()
        p = outdir / f"manifest_{split}.csv"
        df.to_csv(p, index=False)
        print(f"{split}: {len(df)} rows -> {p}  cols={len(cols)}", flush=True)


if __name__ == "__main__":
    main()
