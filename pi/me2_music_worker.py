#!/usr/bin/env python3
"""One-shot audio segment player for the ME2 music tile (instruction 15).

Usage:  python3 me2_music_worker.py <wav> <start_sample> [device_index]

Opens the EMEET-M1A output stream (never the PortAudio default — a
headless Pi's default output is the silent HDMI null sink; if the M1A
isn't in the device table the worker exits 3 and me2_ui tells the user
to reboot), blocks until the slice wav[start_sample:] has been written,
stops the stream, exits 0.

WHY A CHILD PROCESS: re-opening ALSA output streams while pi_demo.py
holds the M1A mic input intermittently ABORTS PortAudio at C level
(assert in PaAlsaStream_Initialize — uncatchable from Python) and an
in-process player took the whole UI down with it (live-verified
2026-10-02). This process is deliberately expendable: me2_ui.py spawns
one per play segment, SIGTERMs it on pause/switch, and a monitor thread
maps its exit code to the tile state (0 = natural end, else fail-soft).
"""
import sys
import wave

import numpy as np
import sounddevice as sd


def load(path: str):
    with wave.open(path, "rb") as wf:
        nch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        if sw != 2:
            raise ValueError("only 16-bit PCM wavs supported")
        arr = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1).astype(np.int16)
    return arr, sr


# exit codes: 0 ok, 1 open/play error, 2 bad args / missing file,
# 3 no M1A output device found (wedge) — me2_ui chips "reboot the Pi"
# for it instead of a generic "player error".
def find_m1a():
    """Output device index of the M1A speaker, or None.

    Fresh query at open time (the USB table moves when the M1A drops).
    10-02 incident: the M1A card wedged out of ALSA and the worker fell
    back to the PortAudio default = the HDMI null sink on a headless Pi
    -> silent "success" (UI showed playing, speaker was dead).
    """
    try:
        devs = sd.query_devices()
    except Exception:
        return None
    for i, d in enumerate(devs):
        if d.get("max_output_channels", 0) > 0 and "EMEET" in d["name"].upper():
            return i
    return None


def main(argv) -> int:
    if len(argv) < 3:
        print("usage: me2_music_worker.py <wav> <start_sample> [device]",
              file=sys.stderr)
        return 2
    wav, start = argv[1], int(argv[2])
    device = int(argv[3]) if len(argv) > 3 else None
    arr, sr = load(wav)
    start = max(0, min(start, len(arr) - 1))
    if device is None:
        device = find_m1a()
        if device is None:
            # NO silent default-sink fallback (see find_m1a).
            print("no M1A output device (USB wedge?) - reboot the Pi",
                  file=sys.stderr)
            return 3
    stream = sd.OutputStream(samplerate=sr, channels=1, dtype="int16",
                             device=device)
    stream.start()
    try:
        stream.write(arr[start:])  # blocks until the slice is drained
    except Exception as e:
        print(f"me2_music_worker: write failed: {e}", file=sys.stderr)
        return 1
    try:
        stream.stop()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
