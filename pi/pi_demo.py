"""ME2 Pi demo v2 — wake gate + command VCM + UI handoff, one process.

No torch on the device. Deps: numpy, onnxruntime, sounddevice. The asset
dir is the export_v2.py payload PLUS the wake extension (src/export_wake.py):
  model.onnx, meta.json, wake_model.onnx, wake_meta.json,
  mel_window.npy, mel_fb.npy, mel.py, pi_demo.py, me2_ui.py

Loop (wake mode — automatic when wake_model.onnx is present):
  rolling 3.0 s buffer, 1 s stride -> 2-class wake gate ("hey boots")
  (no RMS pre-gate — every window runs the wake model; muji, 10-03)
  on fire: ack beep -> 200 ms warm-up -> capture the command until
  1.0 s of RMS silence (max 3.5 s) -> right-pad to 3.0 s -> VCM infer
  -> fire beep + JSON line (printed and POSTed to the UI, /fire)
  3 s cooldown after a fire (the beep would otherwise self-trigger).

  python3 pi_demo.py --dir ~/me2 --list-devices
  python3 pi_demo.py --dir ~/me2 --once --device 1
  python3 pi_demo.py --dir ~/me2 --loop --device 1 \
      --post http://127.0.0.1:8330/fire
  --no-wake  : legacy RMS-gated loop (no wake model required)

Each fire = one JSON line:
  {"ts": ..., "cmd": "set_alarm", "conf": 0.97, "slot": "seven am",
   "slot_index": 26, "slot_alt": "seven pm", "slot_margin": 0.93,
   "infer_ms": 12.3, "wake_conf": 0.99, "e2e_ms": 842}
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

SR = 16000
IN_SAMPLES = 48000  # 3.0 s
CHUNK_S = 0.25      # wake-loop stride (snappy 1006)
FRAME_MS = 30       # endpointing frame


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def post(url: str, obj: dict) -> None:
    try:
        req = urllib.request.Request(
            url, data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=1.5)
    except Exception as e:
        print(f"  post {url} failed: {e}", file=sys.stderr)


def ui_speaking(base: str) -> bool:
    """True while the UI's TTS reply is on the air (self-trigger guard,
    instruction 16, live-verified 2026-10-02: the reply audio hits the
    M1A mic and the wake gate hears it — one reply drove 3 fake fires,
    incl. play_music slot=baby conf 0.9998). Fail-open: UI unreachable
    -> False -> fire normally."""
    try:
        with urllib.request.urlopen(base + "/ui_state",
                                    timeout=1.5) as r:
            return bool(json.load(r).get("speaking"))
    except Exception:
        return False


def vcm_bench_log(res: dict) -> None:
    """Append one fire to ~/vcm_benchmark/vcm.log (read by airimonda/vcm-benchmark).

    JSON line with the four fields its LineParser needs: intent, slot,
    infer_ms, audio_ms. Best-effort; never breaks the fire loop.
    """
    try:
        import os
        d = os.path.expanduser("~/vcm_benchmark")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "vcm.log"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"intent": res.get("cmd"), "slot": res.get("slot") or "",
                                "infer_ms": res.get("infer_ms"),
                                "audio_ms": res.get("audio_ms")}) + "\n")
    except Exception:
        pass


class Demo:
    def __init__(self, pkg_dir: Path):
        sys.path.insert(0, str(pkg_dir))
        import mel as mel_mod
        import onnxruntime as ort
        self.pkg = pkg_dir
        self.meta = json.loads((pkg_dir / "meta.json").read_text())
        self.mel = mel_mod.Mel(str(pkg_dir / "mel_window.npy"),
                               str(pkg_dir / "mel_fb.npy"))
        self.sess = ort.InferenceSession(str(pkg_dir / "model.onnx"),
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.classes = self.meta["classes"]
        self.parametric = set(self.meta["parametric"])
        self.slot_pos = {c: i + 1
                         for i, c in enumerate(self.meta["parametric"])}
        # optional wake gate
        self.wake_sess = None
        self.wake_thr = 0.5
        wm = pkg_dir / "wake_model.onnx"
        if wm.exists():
            meta = json.loads((pkg_dir / "wake_meta.json").read_text())
            self.wake_meta = meta
            self.wake_thr = float(meta.get("threshold", 0.75))
            self.wake_sess = ort.InferenceSession(
                str(wm), providers=["CPUExecutionProvider"])
            self.wake_in = self.wake_sess.get_inputs()[0].name

    def wake_conf(self, wav: np.ndarray) -> float:
        """wav (48000,) -> P(wake) from the 2-class gate."""
        logmel = self.mel(wav)
        out = self.wake_sess.run(
            None, {self.wake_in: logmel.astype(np.float32)})
        return float(softmax(out[0][0])[1])

    def infer(self, wav: np.ndarray):
        """wav (48000,) float32 [-1,1] -> command dict (slot None for
        non-parametric classes)."""
        t0 = time.perf_counter()
        logmel = self.mel(wav)
        outs = self.sess.run(None, {self.in_name: logmel.astype(np.float32)})
        cmd = int(np.argmax(outs[0]))
        cls = self.classes[cmd]
        conf = float(softmax(outs[0][0])[cmd])
        slot = slot_label = slot_alt = None
        slot_margin = None
        if cls in self.parametric:
            sp = softmax(outs[self.slot_pos[cls]][0])
            order = np.argsort(sp)[::-1]
            slot = int(order[0])
            slot_label = self.meta["slot_vocab"][cls][slot]
            # top-2 + margin -> UI low-conf slot badge / future slotcheck
            slot_alt = self.meta["slot_vocab"][cls][int(order[1])]
            slot_margin = round(float(sp[slot] - sp[int(order[1])]), 4)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {"cmd": cls, "conf": round(conf, 4), "slot": slot_label,
                "slot_index": slot, "slot_alt": slot_alt,
                "slot_margin": slot_margin, "infer_ms": round(dt_ms, 1),
                "audio_ms": round(len(wav) / SR * 1000.0, 1)}

    # -------------------------------------------------------------- capture
    _stream = None

    def _open(self, device):
        import sounddevice as sd
        if self._stream is None:
            self._stream = sd.InputStream(
                samplerate=SR, channels=1, dtype="int16", device=device,
                blocksize=int(SR * FRAME_MS / 1000.0))
            self._stream.start()  # NOT auto-started by __init__!
        return self._stream

    def _close(self):
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception:
            pass
        self._stream = None

    def capture(self, device=None, frames: int = IN_SAMPLES):
        """Blocking read from ONE persistent input stream (re-opening a
        USB stream per 30 ms frame wedged the EMEET M1A — the model got
        repeated garbage and classified everything as media_control).
        Self-heals: a stopped stream is reopened once before re-reading."""
        try:
            x, _overflowed = self._open(device).read(frames)
        except Exception:
            self._close()
            x, _overflowed = self._open(device).read(frames)
        return x[:, 0].astype(np.float32) / 32768.0

    def capture_command(self, device=None, min_rms: float = 0.004,
                        warmup_ms: int = 200, silence_ms: int = 700,
                        max_ms: int = 3500):
        """After the wake fires: listen for the command. Returns (48000,)
        right-aligned float32, or None if nothing was heard.

        Stops on FRAME COUNT, not wall clock: each blocking read yields
        exactly one 30 ms frame, so max_ms == audio captured. A wall-clock
        cap under-collected on the M1A (Pi read overhead) — real phrases
        were cut mid-word ('set a ta', 'dim the', 'remind me to ba').
        silence_ms: how long the user may pause mid-phrase before we end
        the capture (600 cut real pauses; 1000 is safer, +400 ms latency
        on short commands)."""
        frame = int(SR * FRAME_MS / 1000.0)
        for _ in range(int(warmup_ms / FRAME_MS)):
            self.capture(device, frames=frame)  # drop the 'boots' tail
        chunks = []
        silent_ms = 0
        voiced_ms = 0
        heard = False
        why = "cap"
        for _ in range(int(max_ms / FRAME_MS)):
            x = self.capture(device, frames=frame)
            rms = float(np.sqrt((x ** 2).mean()))
            chunks.append(x)
            if rms >= min_rms:
                heard = True
                voiced_ms += FRAME_MS
                silent_ms = 0
            elif heard:
                silent_ms += FRAME_MS
            if heard and silent_ms >= silence_ms:
                why = "silence"
                break
        # Need >= 300 ms of ACTUAL voiced audio: the VCM has no "nothing"
        # class, so a 30 ms noise blip would otherwise fire a
        # high-confidence garbage command (observed: silence ->
        # media_control 0.998). Shortest real phrase ~0.5 s.
        if not heard or voiced_ms < 300:
            return None
        wav = np.concatenate(chunks)
        print(f"  endpoint: {why} after {len(wav) / SR * 1000:.0f} ms "
              f"({voiced_ms} ms voiced)", flush=True)
        if len(wav) > IN_SAMPLES:
            wav = wav[-IN_SAMPLES:]
        else:
            pad = np.zeros(IN_SAMPLES - len(wav), dtype=np.float32)
            wav = np.concatenate([pad, wav])  # right-align: speech at tail
        return wav

    def beep(self, device=None, freq: float = 880.0, ms: int = 250,
             amp: float = 0.50, windowed: bool = True):
        """Ack/fire beep. Plays at the OUTPUT device's native rate (USB
        audio like the EMEET M1A often only does 48k, not our 16k SR) and
        never raises — a failed beep must not wedge the state machine.
        250 ms @ 50%: the old 120 ms @ 15% beep was inaudible on the M1A
        speaker in a normal room (a 1 s @ 90% test tone was clearly heard).
        amp/windowed are experiment knobs (beep_ladder.py)."""
        import sounddevice as sd
        try:
            sr_out = int(sd.query_devices(device, "output")[
                "default_samplerate"])
        except Exception:
            sr_out = 48000
        n = int(sr_out * ms / 1000.0)
        t = np.arange(n) / sr_out
        w = amp * 32767 * np.sin(2 * np.pi * freq * t)
        if windowed:
            w = w * np.hanning(n)
        w = w.astype(np.int16)
        try:
            sd.play(w, sr_out, device=device, blocking=True)
        except Exception as e:
            print(f"  beep failed: {e}", file=sys.stderr)

    def chirp(self, device, f0: float, f1: float, ms: int, amp: float = 0.90):
        """Linear frequency-sweep tone (the 'boop' of a cute little bot).
        90 % flat amplitude (50 % / hanning measured inaudible on the M1A
        at 500 ms — only >=650 ms @ 90 % passed the audibility ladder),
        5 ms click-free ramps at the ends."""
        import sounddevice as sd
        try:
            sr = int(sd.query_devices(device, "output")["default_samplerate"])
        except Exception:
            sr = 48000
        n = int(sr * ms / 1000.0)
        t = np.arange(n) / sr
        dur = ms / 1000.0
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * dur))
        w = amp * 32767 * np.sin(phase)
        r = max(1, int(0.005 * sr))
        w[:r] *= np.linspace(0.0, 1.0, r)
        w[-r:] *= np.linspace(1.0, 0.0, r)
        try:
            sd.play(w.astype(np.int16), sr, device=device, blocking=True)
        except Exception as e:
            print(f"  chirp failed: {e}", file=sys.stderr)

    def ack_beep(self, device=None):
        """Wake ack — the HIGHER rising two-note 'boop-boop' (~630 ms)."""
        self.chirp(device, 700.0, 1200.0, 300)
        time.sleep(0.03)
        self.chirp(device, 1000.0, 1600.0, 300)

    def fire_beep(self, device=None):
        """Command fired — the LOWER rising two-note (~630 ms)."""
        self.chirp(device, 500.0, 900.0, 300)
        time.sleep(0.03)
        self.chirp(device, 900.0, 1500.0, 300)

    def save_wav(self, path, wav: np.ndarray):
        """wav float32 [-1,1] -> 16 kHz mono int16 WAV (debug recording —
        the EXACT audio the gate/model consumed)."""
        import wave as wavmod
        x16 = (np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
               * 32767).astype(np.int16)
        with wavmod.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SR)
            f.writeframes(x16.tobytes())



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Pi payload dir (export out)")
    ap.add_argument("--once", action="store_true", help="single capture+infer")
    ap.add_argument("--loop", action="store_true", help="continuous mode")
    ap.add_argument("--device", type=int, default=None,
                    help="input device index (EMeet M1A)")
    ap.add_argument("--min-rms", type=float, default=0.004,
                    help="endpointing silence gate (0.004 ~= -48 dBFS)")
    ap.add_argument("--no-wake", action="store_true",
                    help="legacy RMS-gated loop, even if a wake model ships")
    ap.add_argument("--post", default=None,
                    help="fire URL, e.g. http://127.0.0.1:8330/fire")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--record", nargs="?", const="rec", default=None,
                    metavar="DIR",
                    help="save what the mic hears: wake windows + command "
                         "audio as WAVs in DIR (name has ts/class/conf)")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    if not (args.once or args.loop):
        ap.error("choose --once or --loop")
    demo = Demo(Path(args.dir))
    wake_on = demo.wake_sess is not None and not args.no_wake
    if args.record:
        Path(args.record).mkdir(parents=True, exist_ok=True)
        print(f"recording: {args.record}/ (wake windows + command audio)",
              flush=True)
    if args.post:
        state_url = args.post.rsplit("/", 1)[0] + "/state"
        ui_base = args.post.rsplit("/", 1)[0]
    else:
        state_url = None
        ui_base = None

    def fire(res):
        if ui_base is not None and ui_speaking(ui_base):
            print("  (suppressed — UI is speaking)", flush=True)
            return
        res = dict(res)
        res["ts"] = time.time()
        line = json.dumps(res)
        print(line, flush=True)
        vcm_bench_log(res)
        if args.post:
            post(args.post, res)

    if args.once:
        wav = demo.capture(args.device)
        res = demo.infer(wav)
        if demo.wake_sess is not None:
            res["wake_conf"] = round(demo.wake_conf(wav), 4)
        fire(res)
        return

    if not wake_on:
        print(f"loop: RMS-gated (no wake) device={args.device} "
              f"rms>{args.min_rms}", flush=True)
        cooldown = 0.0
        while True:
            try:
                if time.time() < cooldown:
                    time.sleep(0.5)
                    continue
                wav = demo.capture(args.device)
                rms = float(np.sqrt((wav ** 2).mean()))
                if rms < args.min_rms:
                    print(f". (rms {rms:.4f} < gate)", flush=True)
                    continue
                res = demo.infer(wav)
                print(f"rms {rms:.4f} ->", flush=True)
                fire(res)
                cooldown = time.time() + 3.0
            except KeyboardInterrupt:
                print("bye")
                return
            except Exception as e:
                print(f"error: {e}", file=sys.stderr)
                time.sleep(1.0)

    print(f"loop: WAKE mode (thr {demo.wake_thr}) device={args.device} "
          f"stride {CHUNK_S}s -> endpoint {args.min_rms}", flush=True)
    if state_url:
        post(state_url, {"mode": "idle", "t": time.time()})
    cooldown = 0.0
    chunk_n = int(SR * CHUNK_S)
    buf = []
    hb_last = time.time()
    while True:
        try:
            # liveness tripwire: a wedged sd.InputStream.read() emits
            # nothing forever, so a healthy loop must keep printing --
            # one 'hb' line every 15 s. The supervisor restarts on silence.
            _now = time.time()
            if _now - hb_last >= 15:
                hb_last = _now
                print("hb", flush=True)
            in_cooldown = time.time() < cooldown
            wav = demo.capture(args.device, frames=chunk_n)
            buf.append(wav)
            buf = buf[-IN_SAMPLES // chunk_n:]
            if in_cooldown or len(buf) < IN_SAMPLES // chunk_n:
                continue
            window = np.concatenate(buf)
            conf = demo.wake_conf(window)
            if conf < demo.wake_thr:
                continue
            if ui_base is not None and ui_speaking(ui_base):
                # TTS self-trigger guard (instruction 16): this wake
                # window contains the UI's spoken reply — skip the whole
                # cycle (no beep, no capture, no fire).
                print(f"WAKE conf={conf:.3f} — suppressed (speaking)",
                      flush=True)
                cooldown = time.time() + 1.0
                continue
            t_wake = time.time()
            print(f"WAKE conf={conf:.3f} — listening for command...",
                  flush=True)
            if args.record:
                demo.save_wav(Path(args.record) /
                              f"wake_{t_wake:.0f}_{conf:.2f}.wav", window)
            if state_url:
                post(state_url, {"mode": "awake", "t": t_wake})
            demo.ack_beep(args.device)
            cmd_wav = demo.capture_command(args.device, min_rms=args.min_rms)
            if cmd_wav is None:
                print("  (no command heard)", flush=True)
                if state_url:
                    post(state_url, {"mode": "idle", "t": time.time()})
                cooldown = time.time() + 2.0
                continue
            res = demo.infer(cmd_wav)
            res["wake_conf"] = round(conf, 4)
            res["e2e_ms"] = round((time.time() - t_wake) * 1000.0, 1)
            if args.record:
                demo.save_wav(Path(args.record) /
                              f"cmd_{t_wake:.0f}_{res['cmd']}_"
                              f"{res['conf']:.2f}.wav", cmd_wav)
            fire(res)
            demo.fire_beep(args.device)
            if state_url:
                post(state_url, {"mode": "idle", "t": time.time()})
            cooldown = time.time() + 3.0  # our beep would self-trigger
        except KeyboardInterrupt:
            print("bye")
            return
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            if state_url:  # never leave the UI stuck in 'awake'
                try:
                    post(state_url, {"mode": "idle", "t": time.time()})
                except Exception:
                    pass
            time.sleep(1.0)


if __name__ == "__main__":
    main()
