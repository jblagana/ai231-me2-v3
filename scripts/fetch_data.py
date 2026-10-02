"""Populate the HuggingFace arrow cache for the frozen dataset, once.

V3 trains on the `default` config ONLY (DECISIONS.md — supplemental_* and
synthetic_negatives are outside the freeze). This downloads + materializes the
default splits and prints the arrow dir to pass to tools/train.py --arrow-dir.
On the class HPC the data already lives at /data/ai231 (shared JFS); use that
path and skip this script.

  pip install -U "datasets"
  python scripts/fetch_data.py
  # -> arrow dir: <...>/airimonda___ai231-me2-voice-commands/default/0.0.0
"""
from pathlib import Path

from datasets import load_dataset

REPO = "airimonda/ai231-me2-voice-commands"


def main():
    ds = load_dataset(REPO, "default")
    for split in ("train", "test", "holdout", "numerals"):
        print(f"  {split}: {len(ds[split])} rows", flush=True)
    # Point train.py at the first backing file's parent (the arrow dir).
    files = ds["train"].get_cache_file_names()
    print("arrow dir (for --arrow-dir):", str(Path(files[0]).parent))


if __name__ == "__main__":
    main()
