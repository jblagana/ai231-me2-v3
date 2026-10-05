"""V2 slot taxonomy — maps (class, phrase) to slot value or None.

Seven per-class slot heads (V1 had 6 — play_music is NEWLY parametric:
5 song titles + "any" -> random song on the Pi). 119 values total
(boss-locked 2026-09-30, see DECISIONS.md):
  dim_lights      12: 10..100 by tens + down/up step sentinels
  set_timer       24: 20 minute-values + 1-4 hours
  set_alarm       48: every half-hour mark, 12am -> 11:30pm
  set_temperature 13: celsius 16..28
  set_reminder    10: (unchanged from V1)
  make_call       6:  (unchanged from V1)
  play_music       6: "any" + jetlag, yellow, finesse, multo, baby

App-side semantics (model only outputs the value):
  dim "down"/"up" -> current -30 / +30, clamped [10, 100]
  play_music "any" -> random of the 5 local songs
"""
import re

PARAMETRIC = [
    "dim_lights",
    "set_timer",
    "set_alarm",
    "set_temperature",
    "set_reminder",
    "make_call",
    "play_music",
]

_TENS = ["ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety", "one hundred"]
_CELSIUS = ["sixteen", "seventeen", "eighteen", "nineteen", "twenty",
            "twenty one", "twenty two", "twenty three", "twenty four",
            "twenty five", "twenty six", "twenty seven", "twenty eight"]
_MINS = ["one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "fifteen", "twenty", "twenty five", "thirty",
         "thirty five", "forty", "forty five", "fifty", "fifty five", "sixty"]


def _alarm_marks():
    """All 48 half-hour marks: 12am, 12:30am, 1am, 1:30am, ... 11:30pm."""
    hours = ["twelve", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven"]
    marks = []
    for half in ("am", "pm"):
        for h in hours:
            marks.append(f"{h} {half}")
            marks.append(f"{h} thirty {half}")
    return marks


# Song slot: "any" + 5 locked titles (boss-ratified 2026-09-30).
SONGS = ["jetlag", "yellow", "finesse", "multo", "baby"]

SLOT_VOCAB = {
    "dim_lights": _TENS + ["down", "up"],
    "set_timer": _MINS + ["one hour", "two hours", "three hours", "four hours"],
    "set_alarm": _alarm_marks(),
    "set_temperature": _CELSIUS,
    "set_reminder": [
        "buy groceries", "call mom", "drink water", "pay the bills",
        "take out the trash", "water the plants", "the meeting",
        "five pm", "tomorrow", "next week",
    ],
    "make_call": ["mom", "dad", "brother", "sister", "friend", "the doctor"],
    "play_music": ["any"] + SONGS,
}

SLOT_COUNTS = {k: len(v) for k, v in SLOT_VOCAB.items()}
TOTAL_SLOT_VALUES = sum(SLOT_COUNTS.values())  # 119
def _longest_match(p: str, vocab):
    """Longest vocab entry that appears in the phrase (or None)."""
    best = None
    for v in vocab:
        if v in p and (best is None or len(v) > len(best)):
            best = v
    return best


def extract_slot(cls: str, phrase: str):
    """Return (slot_label, slot_index) or (None, -1) if no slot.

    slot_label: the vocab string (for logging / app action)
    slot_index: index into SLOT_VOCAB[cls], or -1 if no slot
    """
    if cls not in PARAMETRIC:
        return None, -1
    p = phrase.lower().strip()
    vocab = SLOT_VOCAB[cls]

    if cls == "dim_lights":
        m = re.search(r"to\s+(.+?)\s+percent", p)
        if m and m.group(1).strip() in vocab:
            val = m.group(1).strip()
            return val, vocab.index(val)
        if any(w in p for w in ("increase", "turn up", "brighter")):
            return "up", vocab.index("up")
        if any(w in p for w in ("dim", "lower")):
            return "down", vocab.index("down")
        return None, -1

    elif cls == "set_timer":
        m = re.search(r"for\s+(.+?)\s+minutes?", p)
        if m and m.group(1).strip() in vocab:
            val = m.group(1).strip()
            return val, vocab.index(val)
        m = re.search(r"for\s+(one|two|three|four)\s+hours?", p)
        if m:
            val = m.group(1) + (" hour" if m.group(1) == "one" else " hours")
            return val, vocab.index(val)
        return None, -1

    elif cls == "set_alarm":
        val = _longest_match(p, vocab)
        if val:
            return val, vocab.index(val)
        return None, -1

    elif cls == "set_temperature":
        m = re.search(r"to\s+(.+?)\s+degrees", p)
        if m and m.group(1).strip() in vocab:
            val = m.group(1).strip()
            return val, vocab.index(val)
        return None, -1

    elif cls in ("set_reminder", "make_call"):
        val = _longest_match(p, vocab)
        if val:
            return val, vocab.index(val)
        return None, -1

    elif cls == "play_music":
        val = _longest_match(p, SONGS)
        if val:
            return val, vocab.index(val)
        if p.startswith(("play", "start", "put on")):
            return "any", vocab.index("any")
        return None, -1  # "turn off the music" / "stop ..." carry no song


def selftest():
    """Every slot-bearing phrase extracts; every value is covered by at
    least one phrase; counts are as locked."""
    sys_ok = True
    from commands import PHRASES, CLASSES
    checked, uncovered = 0, {}
    for cls in CLASSES:
        if cls not in PARAMETRIC:
            continue
        vocab = SLOT_VOCAB[cls]
        seen = set()
        for ph in PHRASES[cls]:
            label, idx = extract_slot(cls, ph)
            checked += 1
            if idx >= 0:
                seen.add(label)
                assert vocab[idx] == label
        missing = [v for v in vocab if v not in seen]
        if missing:
            sys_ok = False
            uncovered[cls] = missing
    print(f"self-test: {checked} phrases checked, "
          f"{TOTAL_SLOT_VALUES} slot values, {len(SLOT_VOCAB)} heads")
    if uncovered:
        for cls, miss in uncovered.items():
            print(f"  UNCOVERED {cls}: {miss}")
    else:
        print("  all locked values covered by phrases")
    print(f"  total slot values: {TOTAL_SLOT_VALUES} (expect 119)")
    return sys_ok and TOTAL_SLOT_VALUES == 119 and not uncovered


if __name__ == "__main__":
    import sys
    ok = selftest()
    sys.exit(0 if ok else 1)
