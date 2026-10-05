"""Mine authentic VCM phrasings -> site/phrases.json (dashboard data).

Reads (local copies of HPC ~/vcm/runs/v1i/*, scp'd into data/mine/):
  transcripts_all.csv        id,class,source,aug,text,n_words,last_word_end,
                             dur,edge_gap,error   (whisper over VCM_BALANCED)
  vocab_by_class.csv         class,word,count
  quirk_scan.csv             term,class,count,example
  confusion_vcm_real_v1i.json  v1i real-voice baseline (0.2289)
Plus src/commands.py (V2 phrasebook) + src/slots.py (locked vocab) and the
V1_PHRASES snapshot (114 phrases / 11 classes, raw_v1i manifest).

Writes site/phrases.json — the single data source for site/index.html
(assembled by build_dashboard.py). Deterministic; re-runnable anywhere the
CSVs live. The HPC whisper output remains canonical.
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from commands import CLASSES, PHRASES  # noqa: E402
from slots import PARAMETRIC, SLOT_VOCAB  # noqa: E402

MINE = ROOT / "data" / "mine"

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
    "MEDIA_VOLUME_UP": "media_control",
    "MEDIA_VOLUME_DOWN": "media_control",
    "UNKNOWN": "__ood_unknown__",
    "SILENCE": "__ood_silence__",
}

# V1 phrasebook snapshot (114 phrases / 11 classes — raw_v1i manifest truth;
# old repo src/commands.py is the 10-class file with ask_question merged).
V1_PHRASES = {
    "play_music": ["play music", "play some music", "play a song",
                   "put on some music", "play my playlist",
                   "play music in the living room", "put on a song",
                   "start playing music", "play some songs",
                   "play my favorite playlist"],
    "ask_weather": ["what's the weather", "what is the weather",
                    "what's the temperature outside", "search for the weather",
                    "ask about the weather", "what's the forecast for today",
                    "tell me the weather", "how's the weather outside"],
    "ask_time": ["what time is it", "what day is it today",
                 "what time is it right now", "what's today's date",
                 "what day is it"],
    "control_lights": ["turn on the lights", "turn off the lights",
                       "lights on", "lights off", "turn on the light",
                       "turn off the light", "switch on the lights",
                       "switch off the lights", "turn the lights on",
                       "turn the lights off", "switch on the light",
                       "switch off the light"],
    "dim_lights": ["dim the lights", "dim the lights to fifty percent",
                   "make the lights dimmer", "lower the lights",
                   "dim the lights to thirty percent",
                   "dim the lights to seventy percent",
                   "dim the lights to twenty percent",
                   "dim the lights to eighty percent",
                   "make the lights lower", "lower the lights a bit"],
    "set_timer": ["set a timer for five minutes",
                  "set a timer for ten minutes",
                  "set a timer for one minute",
                  "set a timer for thirty minutes",
                  "start a timer for two minutes",
                  "set a timer for fifteen minutes",
                  "set a timer for twenty minutes",
                  "start a timer for one minute",
                  "set a timer for forty five minutes",
                  "start a timer for five minutes"],
    "set_alarm": ["set an alarm for six am", "set an alarm for seven am",
                  "set an alarm for five thirty am", "wake me up at six am",
                  "set an alarm for eight am", "set an alarm for six pm",
                  "wake me up at seven am", "set an alarm for nine am",
                  "set an alarm for five am", "wake me up at eight am"],
    "set_temperature": [
        "set the temperature to twenty two degrees",
        "set the temperature to twenty five degrees",
        "set the thermostat to twenty degrees",
        "set the temperature to eighteen degrees",
        "set the temperature to twenty eight degrees",
        "set the thermostat to twenty four degrees",
        "set the temperature to twenty degrees",
        "set the thermostat to twenty two degrees",
        "set the temperature to twenty six degrees",
        "set the thermostat to eighteen degrees"],
    "media_control": ["pause", "stop", "next song", "skip", "volume up",
                      "volume down", "louder", "quieter", "pause the music",
                      "stop the music", "mute", "unmute",
                      "turn up the volume", "turn down the volume",
                      "previous song"],
    "set_reminder": ["remind me to buy groceries", "remind me to call mom",
                     "remind me at five pm",
                     "add a reminder to water the plants",
                     "remind me about the meeting",
                     "set a reminder for tomorrow",
                     "remind me to take out the trash",
                     "remind me to pay the bills",
                     "set a reminder for next week",
                     "remind me to drink water"],
    "make_call": ["call mom", "call dad", "call my mom", "call my dad",
                  "call brother", "call sister", "call my friend",
                  "give mom a call", "call my sister", "give dad a call",
                  "call the doctor", "call my brother", "phone my mom",
                  "phone my dad"],
}

SEMANTICS = {
    "dim_lights": ('no value: "down" = current-30, "up" = current+30, '
                   'clamped [10,100]; absolute sets the level'),
    "set_timer": "minutes run the timer; 1-4 hours likewise (closed vocab)",
    "set_alarm": "absolute clock time, half-hour marks only",
    "set_temperature": "celsius target; app steps thermostat toward it",
    "set_reminder": "categorical: the reminder content/time",
    "make_call": "categorical: the contact",
    "play_music": ('"any" = random of the 5 local songs; a title plays that '
                   'song (5 locked: jetlag, yellow, finesse, multo, baby)'),
}


def norm(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t.strip(".!? ,")

def main():
    # ---- transcripts: unique phrasings per class --------------------------
    real = defaultdict(lambda: defaultdict(lambda: {
        "count": 0, "sources": set(), "sample_id": None}))
    class_clips = Counter()
    with open(MINE / "transcripts_all.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v2 = V2_MAP.get(row["class"], row["class"])
            class_clips[v2] += 1
            entry = real[v2][norm(row["text"])]
            entry["count"] += 1
            entry["sources"].add(row["source"])
            if entry["sample_id"] is None:
                entry["sample_id"] = row["id"]

    # ---- real top words per class -----------------------------------------
    top_words = defaultdict(list)
    with open(MINE / "vocab_by_class.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v2 = V2_MAP.get(row["class"], row["class"])
            top_words[v2].append((row["word"], int(row["count"])))
    for v2 in top_words:
        top_words[v2] = top_words[v2][:15]

    # ---- quirks + baseline --------------------------------------------------
    quirks = []
    with open(MINE / "quirk_scan.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            quirks.append({"term": row["term"], "class": row["class"],
                           "count": int(row["count"]),
                           "example": row["example"]})
    with open(MINE / "confusion_vcm_real_v1i.json", encoding="utf-8") as f:
        baseline = json.load(f)

    # ---- assemble -----------------------------------------------------------
    classes = []
    for name in CLASSES:
        v1 = set(V1_PHRASES.get(name, []))
        v2ph = PHRASES[name]
        real_cls = real.get(name, {})
        real_rows = [{"text": t, "count": e["count"],
                      "sources": sorted(e["sources"]),
                      "sample_id": e["sample_id"]}
                     for t, e in sorted(real_cls.items(),
                                        key=lambda kv: -kv[1]["count"])]
        classes.append({
            "name": name,
            "clips": class_clips.get(name, 0),
            "unique": len(real_cls),
            "v1": sorted(v1 & set(v2ph)),
            "dropped": sorted(v1 - set(v2ph)),
            "v2_new": [p for p in v2ph if p not in v1],
            "slot": name in PARAMETRIC,
            "vocab": SLOT_VOCAB.get(name),
            "vocab_tbd": None,  # all locked 2026-09-30 (5 songs + steps)
            "semantics": SEMANTICS.get(name),
            "real": real_rows,
            "top_words": top_words.get(name, []),
        })

    unk = real.get("__ood_unknown__", {})
    ood = {
        "UNKNOWN": {"clips": class_clips.get("__ood_unknown__", 0),
                    "unique": len(unk),
                    "examples": [t for t, _ in sorted(
                        ((t, e["count"]) for t, e in unk.items()),
                        key=lambda kv: -kv[1])][:12]},
        "SILENCE": {"clips": class_clips.get("__ood_silence__", 0),
                    "unique": len(real.get("__ood_silence__", {}))},
        "quirks": quirks,
    }

    out = {
        "generated": date.today().isoformat(),
        "meta": {
            "authentic_clips_transcribed": sum(class_clips.values()),
            "policy": ("synthetic-only training; authentic = isolated "
                       "EVAL-REAL; selection on EVAL-SYN (pre-registered)"),
            "v2_phrases": sum(len(v) for v in PHRASES.values()),
            "v1_phrases": sum(len(v) for v in V1_PHRASES.values()),
            "slot_values": sum(len(v) for v in SLOT_VOCAB.values()),
        },
        "classes": classes,
        "ood": ood,
        "baseline": {
            "model": baseline.get("model"),
            "data": baseline.get("data"),
            "accuracy": baseline.get("accuracy"),
            "total": baseline.get("total"),
            "classes": baseline.get("classes"),
            "matrix": baseline.get("matrix"),
        },
    }
    out_path = ROOT / "site" / "phrases.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    total_unique = sum(c["unique"] for c in classes)
    print(f"wrote {out_path}")
    print(f"  authentic clips: {sum(class_clips.values()):,} "
          f"({total_unique:,} unique command phrasings + OOD)")
    for c in classes:
        print(f"  {c['name']:18s} clips={c['clips']:5d} unique={c['unique']:4d} "
              f"v2_new={len(c['v2_new']):2d}")


if __name__ == "__main__":
    main()
