"""V2 wake TTS — "hey boots" wake phrase, wide-voice TTS, speaker-disjoint.

Wider speaker pool than the VCM's 40 (the gate must generalize to a REAL
human voice — the demo speaker). Deterministic pool:
  base 40  = V1/VCM's exact list (20 train / 20 eval, same partition)
  extra 40 = first 40 (sorted) en-* ...Neural voices from edge_tts' live
             catalog, minus the base 40  ->  32 train + 8 eval
Total: 52 train voices, 28 eval voices.

Phrasings: "hey boots" (BOTH splits — the eval phrase) + "hey, boots"
(train only — natural pause variant, broadens coverage).
5 rate jitters, same edge-tts protocol as make_tts_v2 (32 streams,
backoff on throttle, resume-safe manifest).

Train: 52 x 2 phrases x 5 rates = 520
Eval:  28 x 1 phrase  x 5 rates = 140

Output: {out}/{split}/NNNNN.mp3 + manifest.jsonl
  entry: {file, voice, rate, split, phrase}

Usage (HPC, vcm env):
  python tools/make_wake_tts.py --out ~/vcm/data/wake_v2
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_tts_v2 import RATES, SEM, VOICES, synth_one  # noqa: E402

PHRASE_EVAL = "hey boots"
PHRASES_TRAIN = ["hey boots", "hey, boots"]
EXT_TRAIN = 32
EXT_EVAL = 8


async def pick_extended_voices():
    """Deterministic extra pool from the live edge-tts catalog."""
    import edge_tts
    cats = await edge_tts.list_voices()
    base = set(VOICES)
    cands = sorted(
        v["ShortName"] for v in cats
        if v["Locale"].startswith("en")
        and v["ShortName"].endswith("Neural")
        and v["ShortName"] not in base)
    print(f"extended pool: {len(cands)} candidate en voices; "
          f"taking {EXT_TRAIN} train + {EXT_EVAL} eval", flush=True)
    return cands[:EXT_TRAIN], cands[EXT_TRAIN:EXT_TRAIN + EXT_EVAL]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/wake_v2")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ext_tr, ext_ev = await pick_extended_voices()
    splits = {"train": VOICES[:20] + ext_tr, "eval": VOICES[20:] + ext_ev}
    phrases = {"train": PHRASES_TRAIN, "eval": [PHRASE_EVAL]}
    total_planned = sum(len(v) * len(phrases[s]) * len(RATES)
                        for s, v in splits.items())
    print(f"V2 wake TTS: train {len(splits['train'])} voices, "
          f"eval {len(splits['eval'])} voices, {total_planned} planned "
          f"clips, out={out}", flush=True)

    manifest = out / "manifest.jsonl"
    done = set()
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["file"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"RESUME: {len(done)} clips already in manifest, skipping them",
              flush=True)
    n_ok = len(done)
    n_fail = 0
    with manifest.open("a", encoding="utf-8") as mf:
        for split, voices in splits.items():
            d = out / split
            d.mkdir(parents=True, exist_ok=True)
            i = 0
            for voice in voices:
                for phrase in phrases[split]:
                    for rate in RATES:
                        i += 1
                        p = d / f"{i:05d}.mp3"
                        if f"{split}/{p.name}" in done:
                            continue
                        ok = await synth_one(voice, phrase, rate, p)
                        if ok:
                            n_ok += 1
                            mf.write(json.dumps({
                                "file": f"{split}/{p.name}", "voice": voice,
                                "rate": rate, "split": split,
                                "phrase": phrase,
                            }) + "\n")
                        else:
                            n_fail += 1
            print(f"[{split}] {len(voices) * len(phrases[split]) * len(RATES)}"
                  f" clips (cum ok={n_ok} fail={n_fail})", flush=True)

    print(f"DONE: {n_ok} ok, {n_fail} failed")


if __name__ == "__main__":
    asyncio.run(main())

