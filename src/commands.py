"""V3 phrasebook — 19 intents (Dataset Schema Option B) + OUT_OF_SCOPE
rejection class (20th command class, ratified 2026-10-02), single source
of truth.

Locked 2026-10-02 (group meeting, final — instruction 7): Sir Mark's
Dataset Schema (Option B) is the basis for labeling. 13 fixed intents x 3
phrase variations + 6 variable intents x 3 templates x 3 slot values
= 93 phrases. The 3 variations per intent are the fixed demo benchmark
(master Gold Dataset holdout split, 196 clips; one-shot).

Sources of truth:
- Schema sheet: docs.google.com/spreadsheets/d/1HZ7Q58S-konS5XWUE6FUGC-
  S-C8qAoSPaVbIP4NYSU0 (note: shows 2 variation columns; the meeting,
  Sir Mark's generated data (v1/v2/v3 filename component) and the HF
  manifest all use 3 — 3 is canonical here)
- Machine-readable: github.com/markandrian30/AI231 MEX2/Data
  (labels.json, slots.json, README.md — phrase table transcribed verbatim)
- Master corpus: huggingface.co/datasets/airimonda/ai231-me2-voice-commands
  (81,686 clips; train/test/holdout/numerals)

Groups (per schema README): Music control (PLAY_MUSIC, VOLUME_UP,
VOLUME_DOWN, NEXT, PAUSE, STOP); Lighting (LIGHT_ON, LIGHT_OFF,
BRIGHTNESS, COLOR); Temperature control (TEMPERATURE); Information
(WEATHER, TIME); Timers and alarms (TIMER, ALARM); Communication (CALL,
MESSAGE); Reminders (CREATE_REMINDER, LIST_REMINDERS).
"""

CLASSES = [
    # Music control
    "PLAY_MUSIC", "VOLUME_UP", "VOLUME_DOWN", "NEXT", "PAUSE", "STOP",
    # Lighting
    "LIGHT_ON", "LIGHT_OFF", "BRIGHTNESS", "COLOR",
    # Temperature control
    "TEMPERATURE",
    # Information
    "WEATHER", "TIME",
    # Timers and alarms
    "TIMER", "ALARM",
    # Communication
    "CALL", "MESSAGE",
    # Reminders
    "CREATE_REMINDER", "LIST_REMINDERS",
    # Rejection (20th class, ratified 2026-10-02): learned from the master
    # manifest's OUT_OF_SCOPE clips (train 201 / test 47 / holdout 10)
    "OUT_OF_SCOPE",
]

# Option B phrase variations (v1/v2/v3) — verbatim from Sir Mark's
# MEX2/Data README phrase table.
FIXED_VARIATIONS = {
    "PLAY_MUSIC": ["Play music", "Start music", "Play some music"],
    "VOLUME_UP": ["Volume up", "Increase the volume", "Turn the volume up"],
    "VOLUME_DOWN": ["Volume down", "Lower the volume", "Turn the volume down"],
    "NEXT": ["Next song", "Skip song", "Play next song"],
    "PAUSE": ["Pause", "Pause audio", "Pause song"],
    "STOP": ["Stop", "Stop playing", "End playback"],
    "LIGHT_ON": ["Lights on", "Power on the lights", "Turn on the lights"],
    "LIGHT_OFF": ["Lights out", "Kill the lights", "Shut off the lights"],
    "WEATHER": ["Weather", "What's the weather?", "Tell me the weather"],
    "TIME": ["Time", "What time is it?", "Tell me the time"],
    "CALL": ["Call", "Make a call", "Make a phone call"],
    "MESSAGE": ["Message", "Send a message", "Send my message"],
    "LIST_REMINDERS": ["Reminders", "Show my reminders", "List my reminders"],
}

# Variable intents: template x each slot value (slot vocabulary in slots.py,
# verbatim from MEX2/Data slots.json).
VARIABLE_TEMPLATES = {
    "BRIGHTNESS": ["Brightness {slot}", "Adjust brightness to {slot}",
                   "Brightness level {slot}"],
    "COLOR": ["Change color to {slot}", "Switch color to {slot}",
              "Set color to {slot}"],
    "TEMPERATURE": ["Temperature {slot}",
                    "Change the temperature to {slot}",
                    "Set the temperature to {slot}"],
    "TIMER": ["Timer {slot}", "Countdown for {slot}",
              "Start a timer for {slot}"],
    "ALARM": ["Alarm {slot}", "Wake me up at {slot}",
              "Set an alarm for {slot}"],
    "CREATE_REMINDER": ["Reminder {slot}", "Remind me to {slot}",
                        "Create a reminder to {slot}"],
}

PHRASES = {}
for _c, _vs in FIXED_VARIATIONS.items():
    PHRASES[_c] = list(_vs)
for _c, _ts in VARIABLE_TEMPLATES.items():
    from slots import SLOT_VOCAB  # noqa: E402
    PHRASES[_c] = [t.replace("{slot}", v) for t in _ts
                   for v in SLOT_VOCAB[_c]]

# Rejection class: off-vocabulary by definition — no canonical phrase.
PHRASES["OUT_OF_SCOPE"] = []


if __name__ == "__main__":
    total = sum(len(v) for v in PHRASES.values())
    for k in CLASSES:
        print(f"{k:20s} {len(PHRASES[k]):3d} phrases")
    print(f"TOTAL: {total} phrases, {len(CLASSES)} classes")
    assert len(CLASSES) == 20 and total == 93
    assert set(CLASSES) == set(PHRASES)
