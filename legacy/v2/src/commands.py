"""V2 phrasebook — 11 classes, single source of truth (locked 2026-09-30).

V1 (raw_v1i manifest) trained 114 phrases / 11 classes (weather 8 + time 5;
commands.py in the V1 repo is the older 10-class file — the manifest is the
truth). V2 = V1 phrases (minus locations) + mined/delta phrasings:
brightness family, rain/hot/cold weather, time variants, music stop word
orders, play-jetlag, full locked slot coverage (dim 10..100 + steps, timer
mins + hours, alarm 48 half-hour marks, temp C 16..28).

Rules (boss-ratified): NO locations anywhere; numbers spoken as words;
Celsius only; no open vocabulary. Each phrase TTS'd 40 voices x 5 speeds,
speaker-disjoint (20 train / 20 eval).
"""

CLASSES = [
    "play_music",
    "ask_weather",
    "ask_time",
    "control_lights",
    "dim_lights",
    "set_timer",
    "set_alarm",
    "set_temperature",
    "media_control",
    "set_reminder",
    "make_call",
]

PHRASES = {
    "play_music": [
        # V1 (living-room phrase dropped — no locations)
        "play music", "play some music", "play a song", "put on some music",
        "play my playlist", "put on a song", "start playing music",
        "play some songs", "play my favorite playlist",
        # mined word orders (real: "turn off the music" x8, "stop playing music")
        "turn off the music", "stop playing the music",
        # song slot — 5 local songs on the Pi (boss-locked 2026-09-30):
        # jetlag, yellow (Coldplay), finesse (Bruno Mars), multo (Cup of Joe),
        # baby (Justin Bieber)
        "play jetlag", "play the song jetlag",
        "play yellow", "play the song yellow",
        "play finesse", "play the song finesse",
        "play multo", "play the song multo",
        "play baby", "play the song baby",
    ],
    "ask_weather": [
        # V1
        "what's the weather", "what is the weather",
        "what's the temperature outside", "search for the weather",
        "ask about the weather", "what's the forecast for today",
        "tell me the weather", "how's the weather outside",
        # mined (real corpus top: rain family dominates, then hot/cold)
        "is it going to rain today", "is it raining outside",
        "is it going to rain", "is it going to rain tomorrow",
        "does it rain today", "do you think it will rain today",
        "is it going to rain this evening", "is it hot outside",
        "is it cold outside", "what's the weather forecast",
    ],
    "ask_time": [
        # V1
        "what time is it", "what day is it today", "what time is it right now",
        "what's today's date", "what day is it",
        # mined (real: "what is the time" x42, "current time" x34)
        "what is the time", "current time", "what's the current time",
    ],
    "control_lights": [
        # V1 (already location-free)
        "turn on the lights", "turn off the lights", "lights on", "lights off",
        "turn on the light", "turn off the light", "switch on the lights",
        "switch off the lights", "turn the lights on", "turn the lights off",
        "switch on the light", "switch off the light",
        # mined word order
        "switch the lights on", "switch the lights off",
    ],
    "dim_lights": [
        # step phrasings — no value -> DOWN_STEP / UP_STEP (app clamps [10,100])
        "dim the lights", "make the lights dimmer", "lower the lights",
        "increase the brightness", "turn up the brightness",
        "make the lights brighter",
        # absolute: "dim ... to X percent" (X in 10..100 by tens)
        "dim the lights to ten percent", "dim the lights to twenty percent",
        "dim the lights to thirty percent", "dim the lights to forty percent",
        "dim the lights to fifty percent", "dim the lights to sixty percent",
        "dim the lights to seventy percent", "dim the lights to eighty percent",
        "dim the lights to ninety percent",
        "dim the lights to one hundred percent",
        # absolute: "set the lights to X percent" (mined: real x44 + x29)
        "set the lights to ten percent", "set the lights to twenty percent",
        "set the lights to thirty percent", "set the lights to forty percent",
        "set the lights to fifty percent", "set the lights to sixty percent",
        "set the lights to seventy percent", "set the lights to eighty percent",
        "set the lights to ninety percent",
        "set the lights to one hundred percent",
    ],
    "set_timer": [
        # "set a timer for X minutes" — all 20 boss minute-values
        "set a timer for one minute", "set a timer for two minutes",
        "set a timer for three minutes", "set a timer for four minutes",
        "set a timer for five minutes", "set a timer for six minutes",
        "set a timer for seven minutes", "set a timer for eight minutes",
        "set a timer for nine minutes", "set a timer for ten minutes",
        "set a timer for fifteen minutes", "set a timer for twenty minutes",
        "set a timer for twenty five minutes",
        "set a timer for thirty minutes", "set a timer for thirty five minutes",
        "set a timer for forty minutes", "set a timer for forty five minutes",
        "set a timer for fifty minutes", "set a timer for fifty five minutes",
        "set a timer for sixty minutes",
        # "start a timer for X minutes" (V1 word order)
        "start a timer for one minute", "start a timer for two minutes",
        "start a timer for five minutes",
        # hours (agent-rec, boss silent = OK; closed vocab, see DECISIONS.md)
        "set a timer for one hour", "set a timer for two hours",
        "set a timer for three hours", "set a timer for four hours",
    ],
    "set_alarm": [
        # "set an alarm for X" — all 48 half-hour marks, 12am -> 11:30pm
        "set an alarm for twelve am", "set an alarm for twelve thirty am",
        "set an alarm for one am", "set an alarm for one thirty am",
        "set an alarm for two am", "set an alarm for two thirty am",
        "set an alarm for three am", "set an alarm for three thirty am",
        "set an alarm for four am", "set an alarm for four thirty am",
        "set an alarm for five am", "set an alarm for five thirty am",
        "set an alarm for six am", "set an alarm for six thirty am",
        "set an alarm for seven am", "set an alarm for seven thirty am",
        "set an alarm for eight am", "set an alarm for eight thirty am",
        "set an alarm for nine am", "set an alarm for nine thirty am",
        "set an alarm for ten am", "set an alarm for ten thirty am",
        "set an alarm for eleven am", "set an alarm for eleven thirty am",
        "set an alarm for twelve pm", "set an alarm for twelve thirty pm",
        "set an alarm for one pm", "set an alarm for one thirty pm",
        "set an alarm for two pm", "set an alarm for two thirty pm",
        "set an alarm for three pm", "set an alarm for three thirty pm",
        "set an alarm for four pm", "set an alarm for four thirty pm",
        "set an alarm for five pm", "set an alarm for five thirty pm",
        "set an alarm for six pm", "set an alarm for six thirty pm",
        "set an alarm for seven pm", "set an alarm for seven thirty pm",
        "set an alarm for eight pm", "set an alarm for eight thirty pm",
        "set an alarm for nine pm", "set an alarm for nine thirty pm",
        "set an alarm for ten pm", "set an alarm for ten thirty pm",
        "set an alarm for eleven pm", "set an alarm for eleven thirty pm",
        # "wake me up at X" (V1 word order + one mined pm)
        "wake me up at five am", "wake me up at six am", "wake me up at seven am",
        "wake me up at eight am", "wake me up at six pm",
    ],
    "set_temperature": [
        # "set the temperature to X degrees" — celsius only, 16..28
        "set the temperature to sixteen degrees",
        "set the temperature to seventeen degrees",
        "set the temperature to eighteen degrees",
        "set the temperature to nineteen degrees",
        "set the temperature to twenty degrees",
        "set the temperature to twenty one degrees",
        "set the temperature to twenty two degrees",
        "set the temperature to twenty three degrees",
        "set the temperature to twenty four degrees",
        "set the temperature to twenty five degrees",
        "set the temperature to twenty six degrees",
        "set the temperature to twenty seven degrees",
        "set the temperature to twenty eight degrees",
        # "set the thermostat to X degrees" (V1 word order, full coverage)
        "set the thermostat to sixteen degrees",
        "set the thermostat to seventeen degrees",
        "set the thermostat to eighteen degrees",
        "set the thermostat to nineteen degrees",
        "set the thermostat to twenty degrees",
        "set the thermostat to twenty one degrees",
        "set the thermostat to twenty two degrees",
        "set the thermostat to twenty three degrees",
        "set the thermostat to twenty four degrees",
        "set the thermostat to twenty five degrees",
        "set the thermostat to twenty six degrees",
        "set the thermostat to twenty seven degrees",
        "set the thermostat to twenty eight degrees",
        # V1 turn-phrasings
        "turn the thermostat up to twenty five degrees",
        "turn the thermostat down to twenty degrees",
    ],
    "media_control": [
        # V1 (no slot head — generic media verbs)
        "pause", "stop", "next song", "skip", "volume up", "volume down",
        "louder", "quieter", "pause the music", "stop the music", "mute",
        "unmute", "turn up the volume", "turn down the volume",
        "previous song",
        # mined (real: "next track" / "previous track" word orders)
        "next track", "previous track",
    ],
    "set_reminder": [
        # V1, unchanged (10 phrases -> 10 slot values)
        "remind me to buy groceries", "remind me to call mom",
        "remind me at five pm", "add a reminder to water the plants",
        "remind me about the meeting", "set a reminder for tomorrow",
        "remind me to take out the trash", "remind me to pay the bills",
        "set a reminder for next week", "remind me to drink water",
    ],
    "make_call": [
        # V1, unchanged (6 slot values: mom/dad/brother/sister/friend/doctor)
        "call mom", "call dad", "call my mom", "call my dad", "call brother",
        "call sister", "call my friend", "give mom a call", "call my sister",
        "give dad a call", "call the doctor", "call my brother",
        "phone my mom", "phone my dad",
    ],
}

# ---- 10-class experiment (boss 2026-09-30) ---------------------------------
# Same audio, relabel only: ask_weather + ask_time merge back into the spec's
# original single "ask_question" class (no slots either way) -> 10 classes.
# Zero extra TTS; trained/evaluated as a second model in the same run.
MERGE_10 = {"ask_weather": "ask_question", "ask_time": "ask_question"}

CLASSES_10 = []
for _c in CLASSES:
    _t = MERGE_10.get(_c, _c)
    if _t not in CLASSES_10:
        CLASSES_10.append(_t)

PHRASES_10 = {
    _c: (PHRASES["ask_weather"] + PHRASES["ask_time"]
         if _c == "ask_question" else list(PHRASES[_c]))
    for _c in CLASSES_10
}


def relabel_10(cls: str) -> str:
    """Map an 11-class label to its 10-class label (identity if unchanged)."""
    return MERGE_10.get(cls, cls)


if __name__ == "__main__":
    total = sum(len(v) for v in PHRASES.values())
    for k in CLASSES:
        print(f"{k:20s} {len(PHRASES[k]):3d} phrases")
    print(f"TOTAL: {total} phrases, {len(CLASSES)} classes")
    total10 = sum(len(v) for v in PHRASES_10.values())
    print(f"10-class variant: {total10} phrases, {len(CLASSES_10)} classes "
          f"(ask_question = {len(PHRASES_10['ask_question'])})")

