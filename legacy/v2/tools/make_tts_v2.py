"""V2 TTS batch — edge-tts, V1's exact protocol, V2's locked taxonomy.

Protocol (identical to V1's make_dataset.py so EVAL-SYN stays directly
comparable to v1i's 0.8618):
  - 40 voices, 20 train / 20 eval — speaker-disjoint, partition BEFORE gen
  - 5 rate jitters: -10% -5% +0% +5% +10%
  - 32 concurrent async streams; the endpoint rate-limits per IP, so the
    measured steady state is ~45-100 clips/min (09-30), not V1-era 240
  - resume-safe (manifest.jsonl; restart-safe)

V2 delta: 11 classes / 236 locked phrases (src/commands.py) instead of
V1's 10-class phrasebook. V1's raw_v1i audio was deleted post-training
(15 M manifest-only dir on HPC), so the full V2 set is regenerated:
  236 phrases x 40 voices x 5 rates = 47,200 clips, ~3.3 h.

Output: {out}/{split}/{class}/NNNNN.mp3  +  {out}/manifest.jsonl
  entry: {file, class, voice, phrase, rate, split}
  (mp3 is the TTS-native format; tools/decode_tts.py converts to
  s16le 16 kHz .raw for training, V1's layout.)

Usage (HPC, vcm env):
  python tools/make_tts_v2.py --out <path>/raw_v2
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from commands import CLASSES, PHRASES  # noqa: E402

# V1's exact 40 voices (same order, same 20/20 split — comparability).
VOICES = [
    # train voices (20)
    "en-US-GuyNeural", "en-US-JennyNeural", "en-US-ChristopherNeural",
    "en-US-EricNeural", "en-US-AriaNeural", "en-GB-RyanNeural",
    "en-GB-SoniaNeural", "en-GB-ThomasNeural",
    "en-AU-WilliamMultilingualNeural",
    "en-AU-NatashaNeural", "en-IN-PrabhatNeural", "en-IN-NeerjaNeural",
    "en-CA-LiamNeural", "en-CA-ClaraNeural", "en-ZA-LukeNeural",
    "en-ZA-LeahNeural", "en-IE-ConnorNeural", "en-NZ-MitchellNeural",
    "en-PH-JamesNeural", "en-SG-LunaNeural",
    # eval voices (20) — never seen in training
    "en-US-AndrewNeural", "en-US-AvaNeural", "en-US-BrianNeural",
    "en-US-EmmaNeural", "en-US-MichelleNeural", "en-US-RogerNeural",
    "en-US-SteffanNeural", "en-US-AnaNeural", "en-GB-LibbyNeural",
    "en-GB-MaisieNeural", "en-IN-NeerjaExpressiveNeural",
    "en-NZ-MollyNeural", "en-IE-EmilyNeural", "en-PH-RosaNeural",
    "en-SG-WayneNeural", "en-NG-AbeoNeural", "en-NG-EzinneNeural",
    "en-TZ-ElimuNeural", "en-TZ-ImaniNeural", "en-HK-SamNeural",
]

RATES = ["-10%", "-5%", "+0%", "+5%", "+10%"]

# 09-30 throttle findings (measured on n002):
#   - single sequential request: 0.9-1.3 s (service itself is fine)
#   - 32 streams: ~45-100 clips/min, oscillating, fail=0 — the endpoint
#     rate-limits per IP (V1 era measured 240/min; now the bucket is lower)
#   - so concurrency fills the queue but can't raise the IP bucket; the
#     fix is long backoff so throttling waves ride out instead of costing
#     clips (retries=8, 5s..300s exponential + jitter)
SEM = asyncio.Semaphore(32)


async def synth_one(voice, text, rate, out_mp3: Path, retries=8):
    import edge_tts
    backoffs = [5, 15, 45, 90, 180, 300, 300, 300]
    async with SEM:
        for attempt in range(retries):
            try:
                tts = edge_tts.Communicate(text, voice, rate=rate)
                await tts.save(str(out_mp3))
                if out_mp3.exists() and out_mp3.stat().st_size > 500:
                    return True
            except Exception as e:
                if attempt == retries - 1:
                    print(f"  FAIL {voice} {text!r}: {e}", file=sys.stderr)
            await asyncio.sleep(backoffs[attempt] * (0.8 + 0.4 * random.random()))
    return False


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw_v2")
    ap.add_argument("--limit-voices", type=int, default=0,
                    help="debug: only first N voices of each split")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = {"train": VOICES[:20], "eval": VOICES[20:]}
    if args.limit_voices:
        splits = {k: v[:args.limit_voices] for k, v in splits.items()}

    total_planned = sum(len(voices) * len(PHRASES[c]) * len(RATES)
                        for c in CLASSES for voices in splits.values())
    print(f"V2 TTS batch: {len(CLASSES)} classes, "
          f"{sum(len(v) for v in PHRASES.values())} phrases, "
          f"{total_planned} planned clips, out={out}", flush=True)

    # Resumable (same scheme as V1): manifest is append-only; files already
    # listed are skipped. A torn last line from a crash is dropped.
    manifest = out / "manifest.jsonl"
    done = set()
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add(e["file"])
        print(f"RESUME: {len(done)} clips already in manifest, skipping them",
              flush=True)
    n_ok = len(done)
    n_fail = 0
    with manifest.open("a", encoding="utf-8") as mf:
        for split, voices in splits.items():
            for cls in CLASSES:
                d = out / split / cls
                d.mkdir(parents=True, exist_ok=True)
                i = 0
                for voice in voices:
                    for phrase in PHRASES[cls]:
                        for rate in RATES:
                            i += 1
                            p = d / f"{i:05d}.mp3"
                            if f"{split}/{cls}/{p.name}" in done:
                                continue
                            ok = await synth_one(voice, phrase, rate, p)
                            if ok:
                                n_ok += 1
                                mf.write(json.dumps({
                                    "file": f"{split}/{cls}/{p.name}",
                                    "class": cls, "voice": voice,
                                    "phrase": phrase, "rate": rate,
                                    "split": split,
                                }) + "\n")
                            else:
                                n_fail += 1
                print(f"[{split}] {cls}: {i} clips "
                      f"(cum ok={n_ok} fail={n_fail})", flush=True)

    print(f"DONE: {n_ok} ok, {n_fail} failed", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
