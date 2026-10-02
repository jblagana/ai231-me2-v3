"""V3 dataset — frozen manifests + JFS arrow cache + VAD t1 -> samples.

Data source (instruction 18, re-freeze 2026-10-02 — class-shared JFS cache):
- Frozen split export:  data/manifests/manifest_{split}.csv (18 cols, no audio)
- Audio:                /data/ai231/... default config, arrow IPC STREAM
                        (pa.ipc.open_stream — NOT open_file)
- t1 (speech end, policy-v2 window anchor): data/vad_audit/clip_vad_{split}.csv
  (frozen VAD params in tools/vad_audit.py); missing row or no-voiced clip
  -> t1=None (file-end fallback, v1 — flagged in sample meta).

Labels (FROZEN):
- command: CLASSES index (20). The manifest `command` column is canonical
  (OUT_OF_SCOPE rows carry command == "OUT_OF_SCOPE").
- slot: parametric intents only — the manifest `slot_value` surface form is
  normalized (casefold, whitespace collapse, ":00" strip) and must land in
  SLOT_VOCAB[cls]; otherwise slot_idx = -1 (command-only: fixed intents,
  OUT_OF_SCOPE, and the 158 NaN-slot train clips).

Class weights (FROZEN): w_c = n_max / n_c over the TRAIN manifest counts
(max-count class -> 1.0), exposed by class_weight_vector().

OOS lanes (train only, FROZEN policy): real OOS clips (no time-warp, via
features.train(oos=True)) + generated 3.0 s silence + noise-only clips, all
labeled OUT_OF_SCOPE (features.oos_silence / oos_noise_only).

Sample tuple: (x, cmd, slot_cls, slot_idx, meta)
  x:         (1, 80, 150) float32 mel
  cmd:       int, CLASSES index
  slot_cls:  str (parametric class name) or None
  slot_idx:  int, index into SLOT_VOCAB[slot_cls] or -1
  meta:      dict — file, t1, slot_value (canonical or None), duration_s,
             source, is_synthetic, vad_flag ("ok" | "no-voiced" |
             "vad-missing"), gen ("" | "silence" | "noise"), oos (bool)

Torch-free on purpose: numpy out, ints in — tools/train.py owns the collate.

Usage:
  python src/dataset.py --smoke                  # holdout + train smoke
  python src/dataset.py --smoke --noise ~/vcm/data/noise16k
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parent))
from commands import CLASSES               # noqa: E402
from features import Features, decode_wav_bytes  # noqa: E402
from slots import PARAMETRIC, SLOT_VOCAB    # noqa: E402

ARROW_DIR_DEFAULT = ("/data/ai231/airimonda___ai231-me2-voice-commands/"
                     "default/0.0.0")  # class-shared JFS (instr 18)
MANIFEST_DIR = "data/manifests"
VAD_DIR = "data/vad_audit"
N_CLASSES = len(CLASSES)                  # 20
OOS_NAME = "OUT_OF_SCOPE"
OOS_IDX = CLASSES.index(OOS_NAME)

# ":00" strip for alarm surface forms ("6:00 AM" -> "6 AM"); only hour:00,
# so "10 seconds" / "100 percent" are untouched.
_RE_COLON00 = re.compile(r"(\d+):00\b")


def normalize_slot(cls, value):
    """Manifest slot surface form -> (canonical vocab str, idx) or (None, -1)."""
    if cls not in PARAMETRIC:
        return None, -1
    if value is None:
        return None, -1
    v = " ".join(str(value).strip().lower().split())
    v = _RE_COLON00.sub(r"\1", v)
    vocab = SLOT_VOCAB[cls]
    for i, cand in enumerate(vocab):
        if cand.lower() == v:
            return cand, i
    return None, -1


def class_weight_vector(manifest_path):
    """Frozen command-CE class weights w_c = n_max / n_c (max class = 1.0).

    Computed from the train manifest (the FROZEN training set); every one of
    the 20 classes must be present.
    """
    counts = {}
    with open(manifest_path) as fh:
        for row in csv.DictReader(fh):
            counts[row["command"]] = counts.get(row["command"], 0) + 1
    missing = [c for c in CLASSES if c not in counts]
    if missing:
        raise ValueError(f"classes missing from {manifest_path}: {missing}")
    n_max = max(counts.values())
    return np.array([n_max / counts[c] for c in CLASSES], dtype=np.float32)


def load_manifest(manifest_dir, split):
    p = Path(manifest_dir) / f"manifest_{split}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty manifest {p}")
    return rows


def load_arrow_bank(arrow_dir, split):
    """Audio bytes for one split, indexed by basename (whole table in memory;
    train ~0.7 GB is fine on the HPC)."""
    fs = sorted(Path(arrow_dir).rglob(f"*{split}*.arrow"))
    if not fs:
        raise FileNotFoundError(f"no arrow files for split {split!r} "
                                f"under {arrow_dir}")
    tables = []
    for f in fs:
        with pa.memory_map(str(f), "r") as src:
            tables.append(pa.ipc.open_stream(src).read_all())
    t = pa.concat_tables(tables)
    col = t.column("audio")
    files = t.column("file").to_pylist()
    idx = {os.path.basename(f): i for i, f in enumerate(files)}
    return {"bytes": col, "idx": idx, "n": t.num_rows}


def load_vad_index(vad_dir, split):
    """basename -> t1 (float seconds) or None (no-voiced / missing row)."""
    p = Path(vad_dir) / f"clip_vad_{split}.csv"
    t1 = {}
    if not p.exists():
        return t1  # caller flags every sample "vad-missing"
    with open(p) as fh:
        for row in csv.DictReader(fh):
            v = row.get("t1", "")
            t1[os.path.basename(row["file"])] = float(v) if v not in ("", "None") else None
    return t1


class V3Dataset:
    """One split of the frozen V3 Gold Dataset.

    augment=True  -> features.train (noise/warp/gain; OOS clips no-warp)
                     + optional generated OOS lanes (silence + noise-only)
    augment=False -> features.clean (eval/holdout path; the one-shot holdout
                     rule lives upstream of us)
    """

    def __init__(self, split, features, arrow_dir=ARROW_DIR_DEFAULT,
                 manifest_dir=MANIFEST_DIR, vad_dir=VAD_DIR, augment=False,
                 n_gen_oos=0):
        self.split = split
        self.features = features
        self.augment = augment
        rows = load_manifest(manifest_dir, split)
        bank = load_arrow_bank(arrow_dir, split)
        vad = load_vad_index(vad_dir, split)

        self.samp = []          # (kind, basename, cmd, slot_cls, slot_idx, meta)
        self.n_vad_missing = 0
        self.n_no_voiced = 0
        self.n_slot_nan = 0
        self.counts = np.zeros(N_CLASSES, dtype=int)

        for row in rows:
            base = os.path.basename(row["file"])
            if base not in bank["idx"]:
                raise KeyError(f"{split}: {base!r} missing from arrow cache "
                               f"(manifest/arrow revision mismatch?)")
            cmd_name = row["command"]
            if cmd_name not in CLASSES:
                raise ValueError(f"{split}: unknown command {cmd_name!r}")
            oos = cmd_name == OOS_NAME
            oos_flag = int(row["out_of_scope"]) == 1
            if oos != oos_flag:
                raise ValueError(f"{split}: {base}: command/out_of_scope "
                                 f"mismatch ({cmd_name} vs {row['out_of_scope']})")
            slot_val, slot_idx = normalize_slot(cmd_name, row["slot_value"])
            slot_cls = cmd_name if slot_idx >= 0 else None
            if cmd_name in PARAMETRIC and slot_idx < 0:
                self.n_slot_nan += 1
            t = vad.get(base, "__missing__")
            if t == "__missing__":
                self.n_vad_missing += 1
                vad_flag = "vad-missing"
                t = None
            elif t is None:
                self.n_no_voiced += 1
                vad_flag = "no-voiced"
            else:
                vad_flag = "ok"
            meta = {"file": base, "t1": t, "slot_value": slot_val,
                    "duration_s": float(row["duration_s"]),
                    "source": row["source"],
                    "is_synthetic": int(row["is_synthetic"]),
                    "vad_flag": vad_flag, "gen": "", "oos": oos}
            self.samp.append(("real", base, CLASSES.index(cmd_name),
                              slot_cls, slot_idx, meta))
            self.counts[CLASSES.index(cmd_name)] += 1

        if augment and n_gen_oos > 0:
            half = n_gen_oos // 2
            for kind in ("silence",) * half + ("noise",) * (n_gen_oos - half):
                meta = {"file": "", "t1": None, "slot_value": None,
                        "duration_s": 3.0,
                        "source": "generated", "is_synthetic": 1,
                        "vad_flag": "ok", "gen": kind, "oos": True}
                self.samp.append((kind, "", OOS_IDX, None, -1, meta))

        self.bank = bank

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samp)

    def __getitem__(self, i):
        kind, base, cmd, slot_cls, slot_idx, meta = self.samp[i]
        if kind == "real":
            b = self.bank["bytes"][self.bank["idx"][base]].as_py()
            wav = decode_wav_bytes(b["bytes"] if isinstance(b, dict) else b)
            if self.augment:
                x = self.features.train(wav, t1=meta["t1"], oos=meta["oos"])
            else:
                x = self.features.clean(wav, t1=meta["t1"])
        elif kind == "silence":
            x = self.features.oos_silence()
        else:  # "noise"
            x = self.features.oos_noise_only()
        return x, cmd, slot_cls, slot_idx, meta

    # ------------------------------------------------------------------
    def summary(self):
        c = self.counts
        n_real = int(c.sum())
        lines = [f"[{self.split}] n={len(self)} (real={n_real} "
                 f"+ gen={len(self) - n_real})"]
        for name in CLASSES:
            k = CLASSES.index(name)
            if c[k]:
                lines.append(f"  {name:18s} {int(c[k]):6d}")
        lines.append(f"  slot-NaN (command-only): {self.n_slot_nan} | "
                     f"vad missing: {self.n_vad_missing} | "
                     f"no-voiced: {self.n_no_voiced}")
        return "\n".join(lines)



def make_features(buffers_dir="data/mel_buffers", noise_dir=None, seed=0,
                  train=False):
    """Features with the frozen augmentation wiring. train=True requires the
    MUSAN npy bank (noise_dir); eval paths use clean() regardless of the
    bank (clean() never touches the noise)."""
    if train and not noise_dir:
        raise ValueError("train features require noise_dir (MUSAN npy bank)")
    return Features(buffers_dir=buffers_dir, noise_dir=noise_dir,
                    rng=np.random.default_rng(seed))


def smoke(n_holdout=100, n_train=20, buffers_dir="data/mel_buffers",
          noise_dir=None):
    print("== V3 dataset smoke ==")
    f_eval = make_features(buffers_dir, None, seed=1, train=False)
    d_h = V3Dataset("holdout", f_eval, augment=False)
    print(d_h.summary())
    assert len(d_h) == 202, f"holdout must be the re-frozen 202, got {len(d_h)}"
    assert int(d_h.counts[OOS_IDX]) == 16, \
        f"holdout OOS must be 16, got {int(d_h.counts[OOS_IDX])}"

    train_ok = noise_dir is not None
    f_t1 = make_features(buffers_dir, noise_dir, seed=2, train=train_ok)
    d_t = V3Dataset("train", f_t1, augment=True, n_gen_oos=4)
    print(d_t.summary())

    x, cmd, slot_cls, slot_idx, meta = d_h[0]
    assert x.shape == (1, 80, 150) and x.dtype == np.float32
    assert np.isfinite(x).all()
    n_slot, dur, t1s = 0, [], []
    for i in range(n_holdout):
        x, cmd, slot_cls, slot_idx, meta = d_h[i]
        assert x.shape == (1, 80, 150) and np.isfinite(x).all()
        assert 0 <= cmd < N_CLASSES
        if slot_idx >= 0:
            n_slot += 1
            assert slot_cls in PARAMETRIC and slot_cls == CLASSES[cmd]
        else:
            assert slot_cls is None
        dur.append(meta["duration_s"])
        t1s.append(meta["t1"])
    print(f"  sampled {n_holdout} holdout: slot-supervised {n_slot} | "
          f"dur p50 {np.median(dur):.2f}s max {max(dur):.2f}s | "
          f"t1 None {sum(t is None for t in t1s)}")

    # train augmentation: deterministic under a fixed seed
    f_a = make_features(buffers_dir, noise_dir, seed=7, train=train_ok)
    f_b = make_features(buffers_dir, noise_dir, seed=7, train=train_ok)
    d_a = V3Dataset("train", f_a, augment=True, n_gen_oos=4)
    d_b = V3Dataset("train", f_b, augment=True, n_gen_oos=4)
    for i in range(n_train):
        xa, ca, sa, ia, _ = d_a[i]
        xb, cb, sb, ib, _ = d_b[i]
        assert np.array_equal(xa, xb) and ca == cb and ia == ib
    # sampled train slices (head + the generated tail lanes) are finite
    idxs = list(range(0, 300)) + list(range(len(d_a) - 4, len(d_a)))
    for i in idxs:
        kind = d_a.samp[i][0]
        xo, co, so, io, _ = d_a[i]
        assert np.isfinite(xo).all() and xo.shape == (1, 80, 150)
        if kind in ("silence", "noise"):
            assert co == OOS_IDX and io == -1
    oos_i = next(i for i in range(len(d_a))
                 if d_a.samp[i][5]["oos"] and d_a.samp[i][0] == "real")
    xo, co, so, io, _ = d_a[oos_i]
    assert co == OOS_IDX and io == -1

    w = class_weight_vector(os.path.join(MANIFEST_DIR, "manifest_train.csv"))
    # largest-count class = 1.0 (vector floor); max weight = rarer class (OOS)
    assert w.shape == (N_CLASSES,) and np.isfinite(w).all() and w.min() == 1.0
    print(f"  class weights (frozen, nmax/n_c): OOS={w[OOS_IDX]:.3f} "
          f"min={w.min():.3f} (largest class) max={w.max():.3f}")
    print("dataset smoke OK")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-holdout", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--buffers", default="data/mel_buffers")
    ap.add_argument("--noise", default=None,
                    help="MUSAN npy bank dir (train path); e.g. "
                         "~/vcm/data/noise16k")
    args = ap.parse_args()
    if args.smoke:
        ok = smoke(args.n_holdout, args.n_train, args.buffers, args.noise)
        sys.exit(0 if ok else 1)
    ap.print_help()

