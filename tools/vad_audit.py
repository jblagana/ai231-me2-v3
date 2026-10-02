"""Clip-anatomy (VAD) audit vs the locked 3.0 s right-aligned window.

Answers: where is the speech inside each clip, and how often does the
command END before the last 3.0 s (the right-align failure case:
[0-3s command][3-10s silence] would become a silence window labeled
as a command)?

Deterministic energy VAD (no model): 10 ms frames, voiced = frame RMS
> max(1e-3, 10% of peak frame RMS). Stdlib `wave` decode (8/16/32-bit
PCM) — the venv has no soundfile/torchaudio; audio bytes are inline in
the HF arrow cache (stream of RecordBatches).

HPC (from ~/vcm_v3):  .venv/bin/python tools/vad_audit.py
Optional:  --splits train test holdout numerals   --workers 16
Output:    data/vad_audit/clip_vad_{split}.csv + per-split summary.
"""
import argparse
import io
import wave
from pathlib import Path

import numpy as np
import pandas as pd

ARROW_DIR = (Path("data/hf_cache/datasets/"
                 "airimonda___ai231-me2-voice-commands/default/0.0.0"))
SR = 16000
FR = SR // 100            # 10 ms frames
THRESH_ABS = 1e-3
THRESH_REL = 0.10
WIN = 3.0


def decode_wav(b):
    """16/32-bit PCM mono -> float32 in [-1, 1]. Raises on non-PCM."""
    w = wave.open(io.BytesIO(b), "rb")
    ch, sw, n, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
    raw = w.readframes(nf)
    w.close()
    if n != SR:
        raise ValueError(f"sr={n}")
    if sw == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        a = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    else:
        raise ValueError(f"unsupported sampwidth {sw}")
    if ch > 1:  # average to mono
        a = a.reshape(-1, ch).mean(axis=1)
    return a


def vad(a):
    """First/last voiced frame -> (t0, t1) seconds, or None if silent."""
    n = len(a) // FR
    if n == 0:
        return None
    f = a[: n * FR].reshape(n, FR)
    rms = np.sqrt((f * f).mean(axis=1))
    thr = max(THRESH_ABS, THRESH_REL * rms.max())
    v = np.where(rms > thr)[0]
    if len(v) == 0:
        return None
    return v[0] / 100.0, (v[-1] + 1) / 100.0


def clip_stats(row):
    """One arrow row -> stats dict (never raises; audit must survive)."""
    out = {"file": row.get("file", ""), "command": row.get("command", ""),
           "source": str(row.get("source", "")), "error": ""}
    try:
        a = row["audio"]["bytes"] if isinstance(row.get("audio"), dict) else row["audio"]
        x = decode_wav(a)
        d = len(x) / SR
        v = vad(x)
        if v is None:
            out.update(duration_s=d, t0=None, t1=None, span=None,
                       head_sil=None, tail_sil=None, risk=None,
                       error="no-voiced")
            return out
        t0, t1 = v
        out.update(duration_s=d, t0=t0, t1=t1, span=t1 - t0,
                   head_sil=t0, tail_sil=d - t1,
                   risk=1.0 if (d - WIN) > t1 else 0.0)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:70]}"
    return out


def iter_split(split):
    """Yield arrow rows for a split (HF cache = RecordBatch stream)."""
    import pyarrow as pa
    for f in sorted(ARROW_DIR.rglob(f"*{split}*.arrow")):
        with pa.memory_map(str(f), "r") as src:
            rdr = pa.ipc.open_stream(src)
            while True:
                try:
                    b = rdr.read_next_batch()
                except StopIteration:
                    break
                if b is None:
                    break
                yield from b.to_pylist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test", "holdout"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    outdir = Path("data/vad_audit")
    outdir.mkdir(parents=True, exist_ok=True)
    from multiprocessing import Pool

    with Pool(args.workers) as pool:
        for split in args.splits:
            rows = list(iter_split(split))
            print(f"=== {split.upper()} (n={len(rows)}) ===", flush=True)
            stats = pool.map(clip_stats, rows, chunksize=64)
            df = pd.DataFrame(stats)
            df.to_csv(outdir / f"clip_vad_{split}.csv", index=False)

            ok = df[df["error"].fillna("") == ""].copy()
            bad = df[df["error"].fillna("") != ""]
            if len(bad):
                kinds = bad["error"].str.split(":").str[0].value_counts()
                print(f"  load-failures: {len(bad)}  {dict(kinds)}")
            n = len(ok)
            g = lambda c: ok[c].dropna()
            print(f"  speech span: p50 {g('span').median():.2f}  "
                  f"p95 {g('span').quantile(0.95):.2f}  max {g('span').max():.2f}")
            print(f"  head silence >0.5 s: {int((g('head_sil') > 0.5).sum())} "
                  f"({100 * (g('head_sil') > 0.5).mean():.1f}%)")
            ts = g("tail_sil")
            print(f"  tail silence >1.0 s: {int((ts > 1.0).sum())} "
                  f"({100 * (ts > 1.0).mean():.1f}%)   "
                  f"tail >3.0 s: {int((ts > WIN).sum())} ({100 * (ts > WIN).mean():.1f}%)")
            risk = ok[ok["risk"] == 1.0]
            print(f"  RISK (speech ends >3.0 s before EOF — right-align "
                  f"window loses the command): {len(risk)} "
                  f"({100 * len(risk) / max(n, 1):.1f}%)")
            if len(risk):
                for src, k in (risk.groupby("source").size()
                               .sort_values(ascending=False)).items():
                    print(f"    {str(src):32s} {k:4d}")
                worst = risk.nlargest(5, "tail_sil")
                for _, r in worst.iterrows():
                    print(f"    e.g. d={r['duration_s']:.1f}s t1={r['t1']:.1f}s "
                          f"tail={r['tail_sil']:.1f}s  {str(r['file'])[:64]}")
            ls = g("span")
            print(f"  span >2.5 s (long command): {int((ls > 2.5).sum())} "
                  f"({100 * (ls > 2.5).mean():.1f}%)")
            print()
    print("wrote", outdir / "clip_vad_*.csv")


if __name__ == "__main__":
    main()