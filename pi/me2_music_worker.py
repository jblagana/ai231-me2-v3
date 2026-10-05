#!/usr/bin/env python3
"""One-shot audio segment player for the ME2 music tile (ALSA backend).

Usage:  python3 me2_music_worker.py <wav> <start_sample> [device_index]

Uses `aplay -D plughw:CARD=M1A,DEV=0` to play the wav slice directly
through the ALSA kernel driver, bypassing PortAudio entirely.

WHY ALSA INSTEAD OF PORTAUDIO (2026-10-03): the M1A is a USB audio
device that exposes both mic input (held by pi_demo.py) and speaker
output on the same full-speed USB connection. PortAudio's device-table
layer intermittently wedges when both directions are active — the
output flaps out of the table (find_m1a -> None, rc 3) or opens with
paInvalidSampleRate (-9997), even though the ALSA kernel driver handles
the contention fine. Verified live: `aplay -D plughw:CARD=M1A,DEV=0`
opens and plays 10/10 times while pi_demo holds the mic input. The
retry loop that masked the PortAudio wedge is no longer needed; aplay
either opens or it doesn't.

WHY A CHILD PROCESS (unchanged from the PortAudio version): the worker
is deliberately expendable. me2_ui.py spawns one per play segment,
SIGTERMs it on pause/switch, and a monitor thread maps its exit code
to the tile state (0 = natural end, else fail-soft reset).

Exit codes: 0 ok, 1 play error, 2 bad args / missing file,
            3 no M1A output device found.
"""
import os
import signal
import subprocess
import sys
import tempfile
import wave
import numpy as np

# ALSA device spec: plughw handles format conversion (mono→stereo if the
# hardware wants it), so we don't need to match the native channel count.
ALSA_DEV = "plughw:CARD=M1A,DEV=0"

# Retry budget for transient open failures. The ALSA kernel driver is
# much more robust than PortAudio, but a USB reset can still cause a
# brief "device busy" window. 3 tries × 500 ms = 1.5 s is generous.
RETRIES = 3
RETRY_SLEEP_S = 0.5

# Lead-in silence prepended to every stream. The M1A's USB output drops
# the first ~200-300 ms of a freshly-opened aplay (USB first-buffer loss),
# and the pre-recorded clips start with speech at sample 0 — so the dropped
# samples WERE the opening syllables ("The weather is" -> "ther is",
# "Pause" -> "se" — voice-confirmed 2026-10-05). Prepend silence so the
# dropped head is silence, not speech. Tunable: bump if a clip still clips.
WARMUP_S = 0.35

# aplay subprocess (set during play, cleared on exit)
_aplay_proc = None


def _sigterm_handler(signum, frame):
    """SIGTERM → kill aplay immediately, exit 0 (clean pause/switch).
    aplay does NOT handle SIGTERM — it can hold the ALSA output device
    for seconds after terminate(), which blocks the pi_demo ack beep
    (same device). SIGKILL is instant; ALSA releases the device on
    process death. (probe_duck_timing.py 2026-10-03: terminate+wait
    held aplay 2100 ms; SIGKILL is <50 ms.)"""
    global _aplay_proc
    if _aplay_proc is not None:
        try:
            _aplay_proc.kill()
        except Exception:
            pass
    sys.exit(0)


def m1a_present() -> bool:
    """Check if the M1A card is in the ALSA device table."""
    try:
        r = subprocess.run(["aplay", "-l"], capture_output=True, text=True,
                           timeout=5)
        return "M1A" in r.stdout
    except Exception:
        return False


def load(path: str):
    """Load a 16-bit PCM wav, downmix to mono. Returns (arr, sr)."""
    with wave.open(path, "rb") as wf:
        nch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        if sw != 2:
            raise ValueError("only 16-bit PCM wavs supported")
        arr = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1).astype(np.int16)
    return arr, sr


def _trim_silence(arr, sr, side, thr=300, max_ms=None, keep_ms=50):
    """Trim up to max_ms of silence off one end (side='head'|'tail'),
    always leaving at least keep_ms of glue so the splice isn't abrupt.
    No-op if speech is inside the keep margin. (2026-10-06: kills the
    audible gap between the pre-recorded lead clip and the TTS tail -
    edge-tts pads both ends of its output with silence.)"""
    n = len(arr)
    if n == 0:
        return arr
    n_max = min(n, int(sr * (max_ms or (1500 if side == "tail" else 600)) / 1000.0))
    n_keep = int(sr * keep_ms / 1000.0)
    hit = np.nonzero(np.abs(arr) > thr)[0]
    if len(hit):
        s = int(hit[0]) if side == "head" else (n - int(hit[-1]) - 1)
        cut = max(0, min(s, n_max) - n_keep)  # trim down to keep_ms of glue
    else:
        cut = n - n_keep  # all silence: shrink to the glue
    cut = min(max(cut, 0), n - n_keep)
    return arr[cut:] if side == "head" else arr[:n - cut]


def write_slice_wav(arr, sr, start_sample):
    """Write arr[start_sample:] to a temp wav file. Returns the path.

    Prepends WARMUP_S of silence (see WARMUP_S): the M1A drops the first
    ~200-300 ms of a freshly-opened stream, so the dropped head must be
    silence, not the opening syllables of speech."""
    data = arr[start_sample:]
    warm = int(WARMUP_S * sr)
    if warm > 0:
        data = np.concatenate([np.zeros(warm, dtype=np.int16), data])
    tmp = tempfile.mktemp(suffix=".wav", prefix="me2_play_")
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return tmp


def play_aplay(wav_path: str) -> int:
    """Run aplay on the given wav file. Returns aplay's exit code."""
    global _aplay_proc
    _aplay_proc = subprocess.Popen(
        ["aplay", "-D", ALSA_DEV, "-q", wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        rc = _aplay_proc.wait()
        return rc
    finally:
        _aplay_proc = None


def main(argv) -> int:
    if len(argv) < 3:
        print("usage: me2_music_worker.py <wav> [<wav> ...] <start_sample> [device]",
              file=sys.stderr)
        return 2

    # Multi-wav (2026-10-05): the TTS reply path plays a pre-recorded
    # clip + a synthesized tail as one worker invocation — argv holds
    # every wav path, then the start_sample (applied to the FIRST wav
    # only; later wavs play in full, back to back, no re-open gap).
    # Disambiguation: wav args are existing files, start_sample is a
    # number — so argv[2] that is a file means "second wav" (the old
    # single-wav form [wav, start, dev] still parses: start is not a file).
    wavs = [argv[1]]
    i = 2
    while i < len(argv) and os.path.isfile(argv[i]):
        wavs.append(argv[i])
        i += 1
    start = int(argv[i])
    # argv[i+1] (device_index) is the PortAudio device index — ignored in
    # the ALSA backend (we hardcode ALSA_DEV). Kept for interface compat.

    # Install SIGTERM handler before any blocking work
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Check M1A is present (fast path: aplay -l)
    if not m1a_present():
        print("no M1A output device (USB wedge?) - reboot the Pi",
              file=sys.stderr)
        return 3

    # Load and slice (first wav sliced from start; the rest play in full)
    try:
        arr, sr = load(wavs[0])
        if len(wavs) > 1:
            for w in wavs[1:]:
                extra, sr2 = load(w)
                if sr2 != sr:
                    # resample to the first wav's rate (linear; the TTS
                    # path only ever mixes 48k clips with 48k synth)
                    n = int(len(extra) * sr / float(sr2))
                    x = np.linspace(0.0, len(extra) - 1, n)
                    extra = np.interp(x, np.arange(len(extra)),
                                      extra).astype(np.int16)
                # splice trim (2026-10-06): edge-tts pads both ends of
                # its output with silence; clip tail + tail head = the
                # audible gap. Trim each side, keep 50 ms of glue.
                arr = _trim_silence(arr, sr, "tail")
                extra = _trim_silence(extra, sr, "head")
                arr = np.concatenate([arr, extra])
    except Exception as e:
        print(f"me2_music_worker: load failed: {e}", file=sys.stderr)
        return 1
    start = max(0, min(start, len(arr) - 1))

    # Write the slice to a temp wav
    try:
        tmp = write_slice_wav(arr, sr, start)
    except Exception as e:
        print(f"me2_music_worker: write failed: {e}", file=sys.stderr)
        return 1

    # Play with retry
    last_rc = 1
    for attempt in range(RETRIES):
        rc = play_aplay(tmp)
        if rc == 0:
            os.unlink(tmp)
            return 0
        last_rc = rc
        # If the M1A dropped out of the table, it's a real device loss
        if not m1a_present():
            print("no M1A output device (USB wedge?) - reboot the Pi",
                  file=sys.stderr)
            os.unlink(tmp)
            return 3
        # Transient failure — wait and retry
        import time
        time.sleep(RETRY_SLEEP_S)

    # All retries failed
    try:
        os.unlink(tmp)
    except OSError:
        pass
    print(f"me2_music_worker: aplay failed after {RETRIES} tries (rc={last_rc})",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
