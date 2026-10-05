"""Fetch authentic sample wavs for the dashboard (site/audio/<cls>/<id>.wav).

The dashboard renders an <audio> per mined authentic phrasing; the files
come from HPC VCM_BALANCED. Re-runnable, skips files already local.

    python tools/fetch_samples.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DATA = ROOT / "data" / "mine"
HPC = "jan.rhey.lagana@n002.ai.internal"
HPC_BASE = "~/vcm/data/vcm_extract"
LIST = DATA / "_sample_list.txt"


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    phrases = json.loads((SITE / "phrases.json").read_text(encoding="utf-8"))
    pairs = [(c["name"], r["sample_id"])
             for c in phrases["classes"]
             for r in c["real"] if r.get("sample_id")]
    ids = sorted({i for _, i in pairs})
    log(f"{len(pairs)} sample rows, {len(ids)} unique wavs")

    want = {(cls, i) for cls, i in pairs}
    have = {(p.parent.name, p.stem) for p in (SITE / "audio").glob("*/*.wav")}
    missing = sorted(want - have)
    log(f"{len(have)} already local, {len(missing)} to fetch")
    if not missing:
        return

    ids_to_get = sorted({i for _, i in missing})
    DATA.mkdir(parents=True, exist_ok=True)
    # bytes write: text mode would CRLF-ify the list and break tar -T
    # (disk layout: vcm_extract/VCM/VCM_BALANCED/audio/ — zip kept VCM/ prefix)
    LIST.write_bytes(
        ("\n".join(f"VCM/VCM_BALANCED/audio/{i}.wav" for i in ids_to_get)
         + "\n").encode("utf-8"))

    tmp = DATA / "_samples.tar.gz"
    # 1) list up, tar on HPC
    subprocess.run(["scp", "-q", "-o", "ConnectTimeout=15", str(LIST),
                    f"{HPC}:~/_sample_list.txt"], check=True)
    cmd = (f"cd {HPC_BASE} && tar -czf ~/_vcm_samples.tar.gz "
           f"-T ~/\\_sample_list.txt --warning=no-file-changed; "
           f"rm -f ~/\\_sample_list.txt")
    subprocess.run(["ssh", "-o", "ConnectTimeout=30", HPC, cmd], check=True)
    # 2) fetch + untar into staging, then place per class
    subprocess.run(["scp", "-q", "-o", "ConnectTimeout=15",
                    f"{HPC}:~/_vcm_samples.tar.gz", str(tmp)], check=True)
    subprocess.run(["ssh", "-o", "ConnectTimeout=30", HPC,
                    "rm -f", "/_vcm_samples.tar.gz".replace("/", "~/", 1)],
                   check=True)
    staging = DATA / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(tmp) as t:
        t.extractall(staging)
    placed = 0
    for cls, i in missing:
        src = staging / "VCM" / "VCM_BALANCED" / "audio" / f"{i}.wav"
        if not src.exists():
            log(f"  WARN missing in tar: {i}")
            continue
        dst = SITE / "audio" / cls / f"{i}.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dst)
        placed += 1
    shutil.rmtree(staging)
    tmp.unlink()
    log(f"placed {placed} wavs into site/audio/")


if __name__ == "__main__":
    sys.exit(main())