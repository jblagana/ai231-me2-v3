"""Unit test for the me2_ui.py music player (instruction 15, 2026-10-02).

Runs on the LAPTOP with a FAKE WORKER PROCESS (no audio hardware
needed): a pure-stdlib stand-in for me2_music_worker.py that logs the
start offset it was spawned with, "plays" for 0.6 s, then exits 0 (exit
1 in crash mode, exit 3 in wedge mode = M1A card missing). Verifies
play / noop / pause / resume-from-position / track switch / natural end
/ fail-soft (missing file) / worker-crash isolation / wedge chip
("reboot the Pi" instead of silent HDMI-null-sink playback) / "any".

    python tools/test_music_player.py
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
tmp = Path(tempfile.mkdtemp(prefix="me2_music_test_"))
music = tmp / "music"
music.mkdir()
for s in me2_ui.SONGS:  # tiny valid wavs (the fake worker never reads them)
    with wave.open(str(music / f"{s}.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(b"\x00\x00" * SR)  # 1 s of silence

FAKE_WORKER = tmp / "fake_worker.py"
FAKE_WORKER.write_text(textwrap.dedent("""
    import os, sys, time
    wav, start = sys.argv[1], int(sys.argv[2])
    with open(os.path.join(os.path.dirname(wav), ".worker_args.log"), "a") as f:
        f.write(str(start) + "\\n")
    if os.environ.get("FAKE_WORKER_MODE") == "crash":
        sys.exit(1)  # player error (ALSA abort etc.)
    if os.environ.get("FAKE_WORKER_MODE") == "wedge":
        sys.exit(3)  # no M1A output device found
    time.sleep(0.6)  # "the whole track" in fake time
    sys.exit(0)
"""), encoding="utf-8")
me2_ui._music_worker = lambda: FAKE_WORKER
me2_ui.MUSIC_DIR = music
me2_ui._music_init()
ARGS_LOG = music / ".worker_args.log"


def state():
    m = me2_ui.DEV["music"]
    return (m["playing"], m["paused"], m["track"])


def fire(cmd, slot=None, conf=0.95):
    me2_ui._LAST_FIRE.clear()  # skip the 2 s debounce
    return me2_ui.apply_fire({"cmd": cmd, "slot": slot, "conf": conf})


def offsets():
    if not ARGS_LOG.exists():
        return []
    return [int(x) for x in ARGS_LOG.read_text().split()]


def main():
    # 1. play a track -> playing, worker spawned at offset 0
    assert fire("play_music", "yellow") == "fired"
    time.sleep(0.2)
    assert state() == (True, False, "yellow"), state()
    assert me2_ui._M["proc"] is not None
    assert offsets() == [0], offsets()

    # 2. same-song repeat -> noop (ACTUATION guardrail), no new worker
    assert fire("play_music", "yellow") == "noop"
    assert len(offsets()) == 1

    # 3. pause mid-track -> paused, worker killed, position captured
    me2_ui._M["t0"] = time.monotonic() - 0.3  # pretend 0.3 s streamed
    assert fire("media_control") == "fired"
    time.sleep(0.2)
    assert state() == (True, True, "yellow"), state()
    assert me2_ui._M["proc"] is None  # SIGTERMed
    pos = me2_ui._M["pos"]
    assert SR * 0.25 <= pos <= SR * 0.45, pos / SR

    # 4. resume -> new worker, spawned exactly at the paused offset
    assert fire("media_control") == "fired"
    time.sleep(0.2)
    assert state() == (True, False, "yellow"), state()
    assert offsets()[-1] == pos, (offsets(), pos)

    # 5. track switch mid-play -> new worker from the top
    assert fire("play_music", "jetlag") == "fired"
    time.sleep(0.2)
    assert state() == (True, False, "jetlag"), state()
    assert offsets()[-1] == 0

    # 6. natural end (fake worker exits 0 after 0.6 s) -> tile idle + chip
    for _ in range(40):
        if not state()[0]:
            break
        time.sleep(0.05)
    assert state() == (False, False, "jetlag"), state()
    chips = [c for c, _ts in me2_ui.DEV["chips"]]
    assert any("jetlag" in c and "finished" in c for c in chips), chips

    # 7. missing file -> fail-soft: fired but NOT playing
    (music / "baby.wav").unlink()
    assert fire("play_music", "baby") == "fired"
    time.sleep(0.1)
    assert state() == (False, False, "baby"), state()

    # 8. "any" -> one of the (remaining) locked tracks, actually playing.
    #    random.choice can roll an already-unlinked track (fail-soft, no
    #    worker) — re-roll until one lands (the flake fixed 10-02).
    for _ in range(6):
        if fire("play_music", "any") == "fired" and \
                me2_ui.DEV["music"]["playing"]:
            break
    time.sleep(0.1)
    s = state()
    assert s[0] is True and s[2] in me2_ui.SONGS, s

    # 9. worker CRASH -> tile resets, UI survives (the whole point of
    #    the child process)
    os.environ["FAKE_WORKER_MODE"] = "crash"
    (music / "multo.wav").unlink()  # make "any" land on a real file first
    me2_ui._M["proc"] = None
    me2_ui.DEV["music"].update(playing=False, paused=False, track=None)
    assert fire("play_music", "finesse") == "fired"
    for _ in range(40):
        if not state()[0]:
            break
        time.sleep(0.05)
    os.environ.pop("FAKE_WORKER_MODE", None)
    assert state() == (False, False, "finesse"), state()
    chips = [c for c, _ts in me2_ui.DEV["chips"]]
    assert any("player error" in c for c in chips), chips
    assert fire("ask_time") == "fired"  # UI still dispatches normally

    # 9b. worker WEDGE (M1A card gone, exit 3) -> tile resets + the chip
    #     says to reboot, not a vague "player error" (the 10-02 incident:
    #     the worker used to fall back to the silent HDMI null sink and
    #     the tile just stayed "playing")
    os.environ["FAKE_WORKER_MODE"] = "wedge"
    me2_ui._M["proc"] = None
    me2_ui.DEV["music"].update(playing=False, paused=False, track=None)
    assert fire("play_music", "finesse") == "fired"
    for _ in range(40):
        if not state()[0]:
            break
        time.sleep(0.05)
    os.environ.pop("FAKE_WORKER_MODE", None)
    assert state() == (False, False, "finesse"), state()
    chips = [c for c, _ts in me2_ui.DEV["chips"]]
    assert any("no M1A speaker found" in c and "reboot the Pi" in c
               for c in chips), chips

    # 10. media_control with nothing loaded -> chip, no crash
    me2_ui.DEV["music"].update(playing=False, paused=False, track=None)
    assert fire("media_control") == "fired"
    chips = [c for c, _ts in me2_ui.DEV["chips"]]
    assert any("nothing playing" in c for c in chips), chips

    print("OK — all music player checks passed")


if __name__ == "__main__":
    main()

