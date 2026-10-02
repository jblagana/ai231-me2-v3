"""V3 slot taxonomy — Dataset Schema (Option B): 6 variable intents, 18 values.

Locked 2026-10-02 (group meeting, final — instruction 7). Slot values
verbatim from Sir Mark's MEX2/Data slots.json.

Note (CONFIRMED 2026-10-02 against the master Gold Dataset manifest):
the manifest carries EXACTLY these 18 slot values — no beyond-schema
classes (the earlier "~93 slot-value classes" report referred to
phrase buckets, not slot values). Only surface-form differences:
alarms render as "6:00 AM"/"8:00 AM"/"9:00 PM" (schema: "6 AM"/
"8 AM"/"9 PM") and COLOR/CREATE_REMINDER are title-cased ("Red",
"Drink water"). The canonical model vocab is slots.json's rendering;
normalize manifest values case-insensitively + ":00" stripping at
load time.
"""

PARAMETRIC = [
    "TIMER",
    "ALARM",
    "TEMPERATURE",
    "BRIGHTNESS",
    "COLOR",
    "CREATE_REMINDER",
]

SLOT_VOCAB = {
    "TIMER": ["10 seconds", "30 seconds", "1 minute"],
    "ALARM": ["6 AM", "8 AM", "9 PM"],
    "TEMPERATURE": ["18 degrees", "22 degrees", "26 degrees"],
    "BRIGHTNESS": ["20 percent", "60 percent", "100 percent"],
    "COLOR": ["red", "blue", "green"],
    "CREATE_REMINDER": ["drink water", "study", "exercise"],
}

SLOT_COUNTS = {k: len(v) for k, v in SLOT_VOCAB.items()}
TOTAL_SLOT_VALUES = sum(SLOT_COUNTS.values())  # 18


def _longest_match_ci(p: str, vocab):
    """Longest vocab entry that appears in p, case-insensitive (or None)."""
    best = None
    for v in vocab:
        if v.lower() in p and (best is None or len(v) > len(best)):
            best = v
    return best


def extract_slot(cls: str, phrase: str):
    """Return (slot_label, slot_index) or (None, -1) if no slot.

    slot_label: the canonical vocab string (for logging / app action)
    slot_index: index into SLOT_VOCAB[cls], or -1 if no slot
    """
    if cls not in PARAMETRIC:
        return None, -1
    p = phrase.lower()
    val = _longest_match_ci(p, SLOT_VOCAB[cls])
    if val:
        vocab = SLOT_VOCAB[cls]
        return val, vocab.index(val)
    return None, -1


def selftest():
    """Every slot-bearing phrase extracts; every value is covered by at
    least one phrase; counts are as locked."""
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
            assert idx >= 0, f"{cls}: no slot extracted from {ph!r}"
            seen.add(label)
            assert vocab[idx] == label
        missing = [v for v in vocab if v not in seen]
        if missing:
            uncovered[cls] = missing
    print(f"self-test: {checked} phrases checked, "
          f"{TOTAL_SLOT_VALUES} slot values, {len(SLOT_VOCAB)} heads")
    if uncovered:
        for cls, miss in uncovered.items():
            print(f"  UNCOVERED {cls}: {miss}")
    else:
        print("  all locked values covered by phrases")
    print(f"  total slot values: {TOTAL_SLOT_VALUES} (expect 18)")
    return TOTAL_SLOT_VALUES == 18 and not uncovered


if __name__ == "__main__":
    import sys
    ok = selftest()
    sys.exit(0 if ok else 1)
