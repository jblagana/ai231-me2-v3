"""Unit test for the me2_ui.py spoken replies (instruction 16, 2026-10-02).

Runs on the LAPTOP with a FAKE ESPEAK STUB (writes a sine WAV where
espeak-ng would) and the same FAKE WORKER PROCESS as test_music_player.py
(no audio hardware needed). Verifies: ask_question fires the combined
time+weather chip, the reply is "spoken" (worker spawned, then cleared),
music ducks (pause -> speak -> resume), weather cache is used / the
no-feed wording, espeak-missing fail-soft (chip still answers),
worker-crash isolation, the busy guard, and the legacy ask_time /
ask_weather branches.

    python tools/test_speak.py
"""
import os
import sys
import tempfile
import textwrap
import time
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pi"))

import me2_ui  # noqa: E402

SR = 44100

# ------------------------------------------------------------ temp fixture
tmp = Path(tempfile.mkdtemp(prefix="me2_speak_test_"))
music = tmp / "music"
music.mkdir()
for s in me2_ui.SONGS:  # tiny valid wavs (the fake worker never reads them)
    with wave.open(str(music / f"{s}.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(b"\x00\x00" * SR)

FAKE_WORKER = tmp / "fake_worker.py"
FAKE_WORKER.write_text(textwrap.dedent("""
    import os, sys, time
    if os.environ.get("FAKE_WORKER_MODE") == "crash":
        sys.exit(1)  # player error
    if os.environ.get("FAKE_WORKER_MODE") == "wedge":
        sys.exit(3)  # no M1A output device found
    time.sleep(0.6)  # "the whole reply" in fake time
    sys.exit(0)
"""), encoding="utf-8")

FAKE_ESPEAK = tmp / "fake_espeak.py"
FAKE_ESPEAK.write_text(textwrap.dedent("""
    import os, sys, wave
    if os.environ.get("FAKE_ESPEAK_MODE") == "fail":
        sys.exit(1)
    path = sys.argv[sys.argv.index("-w") + 1]
    text = " ".join(sys.argv[sys.argv.index("-w") + 2:])
    with open(os.path.join(os.path.dirname(path), ".espeak_text.log"),
              "a") as f:
        f.write(text + "\\n")
    with wave.open(path, "wb") as wf:  # espeak's real output rate
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\\x00\\x00" * 2205)  # 0.1 s — never played
    sys.exit(0)
"""), encoding="utf-8")

me2_ui._music_worker = lambda: FAKE_WORKER
me2_ui._ESPEAK = [sys.executable, str(FAKE_ESPEAK)]
me2_ui.MUSIC_DIR = music
me2_ui._music_init()

SAID_WAV = Path(tempfile.gettempdir()) / "me2_say.wav"
SAID_WAV_48 = SAID_WAV.with_name("me2_say.wav.48k")
SAID_TEXT = Path(tempfile.gettempdir()) / ".espeak_text.log"


def spoken():
    """The text the fake espeak was last asked to say."""
    return SAID_TEXT.read_text().strip().splitlines()[-1]


def fire(cmd, slot=None, conf=0.95):
    me2_ui._LAST_FIRE.clear()  # skip the 2 s debounce
    return me2_ui.apply_fire({"cmd": cmd, "slot": slot, "conf": conf})


def chips():
    return [t for t, _ in me2_ui.DEV["chips"]]


def music_state():
    m = me2_ui.DEV["music"]
    return (m["playing"], m["paused"], m["track"])


def wait_for(cond, timeout=5.0):
    """Poll cond() until true (slow Windows child-process startups make
    fixed sleeps racy)."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def tts_done():
    return me2_ui._TTS["proc"] is None


def main():
    # 1. ask_question, no weather feed -> chip answers, word-form speak
    #    text, resampled 48k wav, worker spawned then cleared
    me2_ui.STATE["weather"] = None
    for p in (SAID_WAV, SAID_WAV_48, SAID_TEXT):
        if p.exists():
            p.unlink()
    assert fire("ask_question") == "fired"
    chip = chips()[0]
    assert chip.startswith("It's ") and chip.endswith("."), chip
    assert "can't reach the weather feed" in chip, chip
    spk = spoken()
    assert spk.startswith("It's ") and "can't reach the weather feed" in spk
    assert ":" not in spk and "°" not in spk, spk  # word form, no digits
    with wave.open(str(SAID_WAV_48)) as wf:
        assert wf.getframerate() == 48000  # M1A-native rate
    time.sleep(0.15)
    assert me2_ui._TTS["proc"] is not None  # TTS worker running
    assert wait_for(tts_done), "TTS worker did not finish"

    # 2. ask_question, weather cached -> chip "14°", speak "14 degrees"
    me2_ui.STATE["weather"] = {"icon": "🌧", "temp": 14.2,
                               "condition": "overcast",
                               "ts": time.time()}
    assert fire("ask_question") == "fired"
    chip = chips()[0]
    assert "Overcast, 14°." in chip, chip
    assert "Overcast, 14 degrees." in spoken(), spoken()
    assert wait_for(tts_done), "TTS worker did not finish"

    # 3. music playing + ask_question -> duck: pause, speak, resume
    assert fire("play_music", "yellow") == "fired"
    time.sleep(0.2)
    assert music_state() == (True, False, "yellow"), music_state()
    assert fire("ask_question") == "fired"
    time.sleep(0.2)  # TTS running, music should be paused mid-duck
    assert music_state() == (True, True, "yellow"), music_state()
    assert wait_for(lambda: music_state() == (True, False, "yellow")), \
        f"music did not resume after TTS: {music_state()}"

    # 4. espeak missing -> fail-soft: still fired, chip answers, no TTS
    me2_ui._ESPEAK = "definitely-not-a-real-tts-xyz"
    assert fire("ask_question") == "fired"
    assert chips()[0].startswith("It's "), chips()[0]
    assert me2_ui._TTS["proc"] is None
    me2_ui._ESPEAK = [sys.executable, str(FAKE_ESPEAK)]

    # 5. TTS worker crash (exit 1) -> isolated: slot clears, nothing raised
    os.environ["FAKE_WORKER_MODE"] = "crash"
    try:
        assert me2_ui.speak("test crash") is True
        assert wait_for(tts_done), "crashed TTS slot did not clear"
    finally:
        del os.environ["FAKE_WORKER_MODE"]

    # 5b. TTS worker WEDGE (M1A card gone, exit 3) -> slot clears + chip
    #     tells the boss to reboot (before 10-02 this was a silent
    #     null-sink "play" with no error anywhere)
    os.environ["FAKE_WORKER_MODE"] = "wedge"
    try:
        assert me2_ui.speak("test wedge") is True
        assert wait_for(tts_done), "wedge TTS slot did not clear"
    finally:
        del os.environ["FAKE_WORKER_MODE"]
    assert any("no M1A speaker found" in c and "reboot the Pi" in c
               for c in chips()), chips()

    # 6. busy guard: second reply while the first is still talking
    assert me2_ui.speak("first reply") is True
    time.sleep(0.15)
    assert me2_ui.speak("second reply") is False
    assert wait_for(tts_done)

    # 7. legacy branches (v2a class set — dead on v2e, kept working)
    assert fire("ask_time") == "fired"
    assert chips()[0].startswith("it's "), chips()[0]
    assert fire("ask_weather") == "fired"
    assert "Overcast — 14°" in chips()[0], chips()[0]
    me2_ui.STATE["weather"] = None
    assert fire("ask_weather") == "fired"
    assert chips()[0] == "can't reach the weather feed right now", chips()[0]

    # 8. TTS self-trigger window: /ui_state reports speaking=True while
    #    the reply is "on the air" (pi_demo.ui_speaking suppresses fires
    #    in the window — the reply audio otherwise self-triggers the
    #    wake gate, live-verified 10-02)
    assert me2_ui.speak("window check") is True
    assert me2_ui.STATE["speaking_until"] > time.time()
    assert me2_ui.ui_state_json()["speaking"] is True
    assert wait_for(tts_done)
    me2_ui.STATE["speaking_until"] = time.time() - 1.0
    assert me2_ui.ui_state_json()["speaking"] is False

    print("OK — all speak checks passed")


if __name__ == "__main__":
    main()