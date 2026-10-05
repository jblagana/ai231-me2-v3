"""EVAL-REAL prep — runs on HPC (n002/n003), stdlib only.

Leak-free authentic eval split for V2 (see DECISIONS.md):
  1. Reconcile: VCM.zip entries vs extracted VCM_BALANCED vs train.csv.
  2. Extract VCM_MASTER manifests+reports from the zip (NOT the 36k FLACs —
     eval audio is the BALANCED wavs; MASTER manifests give the split).
  3. EVAL-REAL = MASTER's own val/test clips (the dataset's canonical
     speaker-disjoint held-out set). Verified 2026-09-30: BAL was built from
     MASTER train only (0 of BAL's 13,801 originals appear in MASTER val/test)
     -> zero overlap with the balanced pool by construction.
  4. Relabel 16 -> 11 V2 classes + OOD, emit manifest + report:
       <root>/data/v2_real_eval/manifest.jsonl
       <root>/data/v2_real_eval/eval_real_{val,test}.csv
       <root>/data/v2_real_eval/report.txt

Usage:  python3 prep_real_eval.py            # root = ~/vcm
        python3 prep_real_eval.py --root /path/to/vcm
"""
import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

# VCM's 16 real classes -> V2's 11 (+ OOD buckets)
V2_MAP = {
    "PLAY_MUSIC": "play_music",
    "WEATHER": "ask_weather",
    "TIME": "ask_time",
    "LIGHT_ON": "control_lights",
    "LIGHT_OFF": "control_lights",
    "LIGHT_DIM": "dim_lights",
    "SET_TIMER": "set_timer",
    "SET_ALARM": "set_alarm",
    "SET_TEMPERATURE": "set_temperature",
    "MEDIA_PAUSE": "media_control",
    "MEDIA_STOP": "media_control",
    "MEDIA_NEXT": "media_control",
    "MEDIA_PREVIOUS": "media_control",
    "VOLUME_UP": "media_control",
    "VOLUME_DOWN": "media_control",
    "UNKNOWN": "OOD_UNKNOWN",
    "SILENCE": "OOD_SILENCE",
}


def log(msg):
    print(msg, flush=True)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "vcm"))
    args = ap.parse_args()
    root = Path(args.root)
    extract = root / "data" / "vcm_extract" / "VCM"
    bal = extract / "VCM_BALANCED"
    master = extract / "VCM_MASTER"
    zip_path = root / "data" / "VCM.zip"
    out = root / "data" / "v2_real_eval"
    out.mkdir(parents=True, exist_ok=True)
    report = []

    # ---- 1. reconcile -------------------------------------------------------
    log("== 1. reconcile ==")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        zip_bal_audio = [n for n in names
                         if n.startswith("VCM/VCM_BALANCED/audio/")
                         and n.endswith(".wav")]
        zip_master_audio = [n for n in names
                            if n.startswith("VCM/VCM_MASTER/audio/")]
        zip_has_master_man = any(
            n.startswith("VCM/VCM_MASTER/manifests/") for n in names)
    disk_bal_audio = set(p.name for p in (bal / "audio").glob("*.wav"))
    disk_bal_ids = {p.stem for p in (bal / "audio").glob("*.wav")}
    bal_manifest = read_csv(bal / "manifests" / "train.csv")
    man_ids = {r["id"] for r in bal_manifest}
    report.append("RECONCILE")
    report.append(f"  zip VCM_BALANCED audio entries : {len(zip_bal_audio)}")
    report.append(f"  extracted BALANCED audio files : {len(disk_bal_audio)}")
    report.append(f"  BALANCED train.csv rows (disk) : {len(bal_manifest)}")
    report.append(f"  manifest ids missing on disk   : "
                  f"{len(man_ids - disk_bal_ids)}")
    report.append(f"  disk files not in manifest     : "
                  f"{len(disk_bal_ids - man_ids)} (intermediate aug outputs)")
    report.append(f"  zip VCM_MASTER audio entries   : {len(zip_master_audio)}")
    report.append(f"  zip has VCM_MASTER manifests   : {zip_has_master_man}")
    # ---- 2. extract MASTER manifests + complete BALANCED ----------------------
    # The earlier V1 extraction left BALANCED partial (14,144 of 15,268 wavs
    # and a truncated train.csv) — top up from the zip before deriving splits.
    # NOTE: zip members carry the leading "VCM/" prefix, so the extract base
    # is extract.parent (…/vcm/data/vcm_extract), NOT extract.
    log("== 2. extract MASTER manifests + complete BALANCED ==")
    with zipfile.ZipFile(zip_path) as z:
        members = [n for n in names
                   if n.startswith("VCM/VCM_MASTER/manifests/")
                   or n.startswith("VCM/VCM_MASTER/reports/")
                   or n.startswith("VCM/VCM_BALANCED/manifests/")]
        missing_audio = [n for n in zip_bal_audio
                         if n.split("/")[-1][:-4] not in disk_bal_ids]
        members += missing_audio
        log(f"  extracting {len(members)} members "
            f"({len(missing_audio)} missing BAL audio) ...")
        z.extractall(extract.parent, members)
    log("  done")
    stray = extract / "VCM"  # leftover from the double-nested extraction bug
    if stray.exists():
        import shutil
        shutil.rmtree(stray)
        log("  removed stray nested VCM/VCM tree")
    disk_bal_ids = {p.stem for p in (bal / "audio").glob("*.wav")}
    # re-read the (now complete) manifest for the split derivation
    bal_manifest = read_csv(bal / "manifests" / "train.csv")
    man_ids = {r["id"] for r in bal_manifest}
    report.append(f"  BALANCED train.csv rows (final): {len(bal_manifest)}")

    # ---- 2b. extract MASTER val/test audio (the held-out eval set) ----------
    log("== 2b. extract MASTER val/test FLACs ==")
    eval_rows_by_split = {}
    for split in ("val", "test"):
        eval_rows_by_split[split] = read_csv(master / "manifests" /
                                             f"{split}.csv")
    want = {}
    for split, rows_ in eval_rows_by_split.items():
        for r in rows_:
            fp = r["filepath"]
            mid = fp.split("/")[-1]
            mid = mid[:-4] if mid.endswith((".wav", ".flac")) else mid
            want[mid] = f"VCM/VCM_MASTER/{fp}"
    with zipfile.ZipFile(zip_path) as z:
        missing = [n for n in want.values()
                   if not (extract / "VCM_MASTER" / n.split("VCM/", 1)[-1])
                   .exists()]
        log(f"  extracting {len(missing)} of {len(want)} FLACs ...")
        z.extractall(extract.parent, missing)
    log("  done")

    # ---- 3. EVAL-REAL = MASTER val/test, leak checks -------------------------
    log("== 3. EVAL-REAL build + leak checks ==")
    master_speakers = defaultdict(set)
    for split, rows_ in eval_rows_by_split.items():
        for r in rows_:
            if r.get("speaker"):
                master_speakers[split].add(r["speaker"])
    spk_overlap = master_speakers["val"] & master_speakers["test"]
    bal_orig = {r["original_source"].split("/")[-1][:-4]
                for r in bal_manifest if r.get("original_source")}
    report.append("SPLIT")
    report.append(f"  MASTER val rows: {len(eval_rows_by_split['val'])}, "
                  f"test rows: {len(eval_rows_by_split['test'])}")
    report.append(f"  MASTER val∩test speakers: {len(spk_overlap)} "
                  f"(0 = speaker-disjoint)")
    report.append(f"  BAL unique originals: {len(bal_orig)}; in MASTER "
                  f"val/test: {len(bal_orig & set(want))} "
                  f"(0 = BAL built from MASTER train only -> no overlap)")

    rows, missing_files = [], 0
    for split, rows_ in eval_rows_by_split.items():
        for r in rows_:
            fp = r["filepath"]
            mid = fp.split("/")[-1]
            mid = mid[:-4] if mid.endswith((".wav", ".flac")) else mid
            wav = master / fp
            if not wav.exists():
                missing_files += 1
                continue
            rows.append({
                "id": mid,
                "wav": str(wav),
                "class_raw": r.get("label", ""),
                "v2_class": V2_MAP.get(r.get("label", ""),
                                       r.get("label", "")),
                "split": split,
                "speaker": r.get("speaker", ""),
                "source": r.get("source", ""),
                "transcript": r.get("transcript", ""),
            })
    split_counts = Counter(x["split"] for x in rows)
    report.append(f"  EVAL-REAL clips: {len(rows)} "
                  f"(val={split_counts['val']}, test={split_counts['test']}), "
                  f"missing files: {missing_files}")

    # ---- 4. relabel + emit ------------------------------------------------------
    log("== 4. relabel + emit ==")
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in sorted(rows, key=lambda x: (x["split"], x["id"])):
            f.write(json.dumps(row) + "\n")
    for split in ("val", "test"):
        with open(out / f"eval_real_{split}.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "wav", "v2_class", "class_raw"])
            for row in rows:
                if row["split"] == split:
                    w.writerow([row["id"], row["wav"], row["v2_class"],
                                row["class_raw"]])

    per = defaultdict(Counter)
    for row in rows:
        per[row["split"]][row["v2_class"]] += 1
    for split in ("val", "test"):
        report.append(f"EVAL-REAL {split.upper()} "
                      f"({sum(per[split].values())} clips)")
        for cls in sorted(per[split]):
            report.append(f"  {cls:18s} {per[split][cls]:5d}")

    (out / "report.txt").write_text("\n".join(report) + "\n",
                                    encoding="utf-8")
    log("\n".join(report))
    log(f"\nwrote {out}/manifest.jsonl + eval_real_{{val,test}}.csv + report.txt")


if __name__ == "__main__":
    main()