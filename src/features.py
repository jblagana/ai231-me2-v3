"""V3 feature extraction — the SINGLE mel implementation for HPC and Pi.

The Mel class is a verbatim port of V2's Pi-verified src/mel.py (bit-exact
vs V2 training; V1/V2 verify_pi parity pattern). Buffers
(data/mel_buffers/mel_window.npy (1024,), mel_fb.npy (513,80)) are
V2's SHIPPED buffers — the exact ones the reused wake model consumes, so
one mel implementation serves: HPC training, HPC eval, Pi runtime, and
the wake gate. Zero parity risk by construction; verify_pi_v3 still
re-checks on the device before the demo.

Locked spec (BENCHMARK.md / DECISIONS.md "Truncation policy v2"):
  waveform 16 kHz mono
    -> 3.0 s window, right-aligned to SPEECH END (policy v2):
       window ends at min(d, t1 + 1.0 s); t1 = last voiced frame from the
       frozen VAD audit (data/vad_audit/clip_vad_*.csv); left-pad silence;
       t1=None -> file-end right-align (v1 fallback, no-voiced clips)
    -> [train only] MUSAN noise mix in the WAVEFORM domain, SNR 5-25 dB
    -> STFT: reflect-pad 512, 151 frames @ hop 320, window (1024,)
    -> power -> 80-mel fb (513,80), 50-8000 Hz -> (80,151)
    -> clip 1e-5 -> log10 -> (x+5)/5 -> (80,150) (drop last frame)
    -> [train only] time-warp 0.95-1.05 (mel, tail-anchored, no OOS)
    -> [train only] gain +-10 dB (mel, +-0.2 normalized; no pure silence)
  -> (1, 80, 150) float32

Order (FROZEN): noise -> mel -> warp -> gain.
OOS lane: no time-warp; generated silence/noise-only clips at 3.0 s.
"""
import io
import wave
from pathlib import Path

import numpy as np

SR = 16000
WIN_S = 3.0
IN_SAMPLES = int(WIN_S * SR)          # 48000
N_FFT, HOP, PAD = 1024, 320, 512
N_MELS = 80
N_FRAMES = 150
TAIL_S = 1.0                          # = live silence_ms/1000 (policy v2)

# Augmentation (train only, FROZEN)
SNR_MIN, SNR_MAX = 5.0, 25.0          # waveform-domain MUSAN, dB
WARP_MIN, WARP_MAX = 0.95, 1.05       # mel time-warp (tail-anchored)
GAIN_NORM = 10.0 / 10.0 / 5.0         # +-10 dB -> +-1.0 log10 -> +-0.2 (x+5)/5
EPS = 1e-5


class Mel:
    """V2 Pi-verified numpy mel. wav (48000,) float32 in [-1,1]
    -> (1, 80, 150) float32. MUST stay bit-identical to the Pi port."""

    def __init__(self, buffers_dir="data/mel_buffers"):
        b = Path(buffers_dir)
        self.window = np.load(b / "mel_window.npy").astype(np.float32)
        self.fb = np.load(b / "mel_fb.npy").astype(np.float32)

    def __call__(self, wav):
        x = np.pad(wav, (PAD, PAD), mode="reflect")
        n_frames = (len(x) - N_FFT) // HOP + 1                  # 151
        idx = np.arange(N_FFT)[None, :] + (HOP * np.arange(n_frames))[:, None]
        frames = x[idx] * self.window                           # (151, 1024)
        spec = np.fft.rfft(frames, axis=1)                      # (151, 513)
        power = spec.real ** 2 + spec.imag ** 2
        mel = (power @ self.fb).T                               # (80, 151)
        logmel = (np.log10(np.clip(mel, EPS, None)) + 5.0) / 5.0
        return logmel[:, :N_FRAMES].astype(np.float32)[None]    # (1, 80, 150)


def decode_wav_bytes(b):
    """PCM wav bytes (8/16/32-bit) -> float32 mono in [-1,1], 16 kHz."""
    w = wave.open(io.BytesIO(b), "rb")
    ch, sw, n, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
    raw = w.readframes(nf)
    w.close()
    if sw == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        a = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    else:
        raise ValueError(f"unsupported sampwidth {sw}")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if n != SR:  # naive linear resample (documented V2 dependency)
        m = int(len(a) * SR / n)
        a = np.interp(np.linspace(0, len(a) - 1, m), np.arange(len(a)), a)
        a = a.astype(np.float32)
    return a


def load_wav(src):
    """wav bytes, .wav path, or .raw int16 path -> float32 mono 16 kHz."""
    if isinstance(src, (bytes, bytearray, io.BytesIO)):
        return decode_wav_bytes(bytes(src))
    p = Path(src)
    if p.suffix == ".raw":
        return np.fromfile(p, dtype=np.int16).astype(np.float32) / 32768.0
    return decode_wav_bytes(p.read_bytes())


def speech_end_window(wav, t1):
    """Policy v2: 3.0 s window right-aligned to SPEECH END, not file end.

    wav: full-clip waveform (n,) float32; t1: last voiced frame time (s),
    or None for the file-end fallback (no-voiced clips, flagged upstream).
    Window ends at min(d, t1 + TAIL_S); left-pad with silence.
    TAIL_S = 1.0 s = EXACTLY the shipping live geometry: V2 pi_demo.py
    capture_command(silence_ms=1000) ends the capture 1.0 s after the
    last voiced frame (V2's original 600 ms was raised because it cut
    real mid-phrase pauses; its docstring line is stale).
    """
    n_out = IN_SAMPLES
    d = len(wav)
    end = d if t1 is None else min(d, int((t1 + TAIL_S) * SR))
    seg = wav[max(0, end - n_out):end]
    out = np.zeros(n_out, dtype=np.float32)
    out[n_out - len(seg):] = seg[:n_out]
    return out


def mix_noise(wav, noise, snr_db):
    """Add `noise` (same length, float32) to `wav` at snr_db (waveform domain).
    SNR is window-total energy based: P_sig/P_noise = 10^(snr/10)."""
    p_sig = float((wav.astype(np.float64) ** 2).sum())
    p_noise = float((noise.astype(np.float64) ** 2).sum())
    if p_sig <= 0.0 or p_noise <= 0.0:
        return wav + (noise if p_sig <= 0.0 else 0.0)
    g = np.sqrt(p_sig / (p_noise * 10.0 ** (snr_db / 10.0)))
    return (wav + g * noise).astype(np.float32)


def time_warp_right(mel, r):
    """Mel time-warp, TAIL-ANCHORED (right-aligned): the last frame is fixed,
    the window is stretched/squeezed toward the start. mel (1,80,150)."""
    if abs(r - 1.0) < 1e-9:
        return mel
    x = mel[0]                                              # (80, 150)
    n = x.shape[1]
    pos = (n - 1) - ((n - 1) - np.arange(n, dtype=np.float32)) / r
    pos = np.clip(pos, 0, n - 1)
    warped = np.array([np.interp(pos, np.arange(n), x[m]) for m in range(x.shape[0])])
    return warped.astype(np.float32)[None]


class Features:
    """V3 feature pipeline. Eval path is clean (window + mel only).
    Train path adds waveform noise -> (mel) -> warp -> gain, in that order."""

    def __init__(self, buffers_dir="data/mel_buffers", noise_dir=None,
                 rng=None, load_noise=True):
        self.mel = Mel(buffers_dir)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.noise = []
        if noise_dir and load_noise:
            for p in sorted(Path(noise_dir).glob("*.npy")):
                a = np.load(p)
                if a.dtype != np.float32:
                    a = a.astype(np.float32) / 32768.0
                self.noise.append(a)
            self.noise_starts = np.array(
                [max(0, len(a) - IN_SAMPLES) for a in self.noise])

    # -- core paths ----------------------------------------------------
    def clean(self, wav, t1=None):
        """Eval/holdout path: policy-v2 window + mel, no augmentation."""
        return self.mel(speech_end_window(wav, t1))

    def train(self, wav, t1=None, oos=False, pure_silence=False):
        """Train path: window -> [waveform noise] -> mel -> [warp] -> [gain].
        oos: no time-warp (frozen). pure_silence: skip noise+gain (zeros)."""
        x = speech_end_window(wav, t1)
        if not pure_silence and self.noise:
            i = int(self.rng.integers(len(self.noise)))
            a = self.noise[i]
            s = int(self.rng.integers(self.noise_starts[i] + 1))
            snr = float(self.rng.uniform(SNR_MIN, SNR_MAX))
            x = mix_noise(x, a[s:s + IN_SAMPLES], snr)
        feat = self.mel(x)
        if not oos and not pure_silence:
            r = float(self.rng.uniform(WARP_MIN, WARP_MAX))
            feat = time_warp_right(feat, r)
        if not pure_silence:
            feat = feat + float(self.rng.uniform(-GAIN_NORM, GAIN_NORM))
        return feat

    def oos_silence(self):
        """Generated 3.0 s silence, labeled OUT_OF_SCOPE. Exactly 0.0."""
        return self.mel(np.zeros(IN_SAMPLES, dtype=np.float32))

    def oos_noise_only(self, snr_db=None):
        """3.0 s noise-only clip (level varied by gain only), labeled
        OUT_OF_SCOPE. No time-warp."""
        if not self.noise:
            return self.oos_silence()
        i = int(self.rng.integers(len(self.noise)))
        a = self.noise[i]
        s = int(self.rng.integers(self.noise_starts[i] + 1))
        return self.train(a[s:s + IN_SAMPLES], t1=None, oos=True)


def selftest(buffers_dir="data/mel_buffers"):
    """Local smoke test (no data files needed): synthetic audio only.
    python src/features.py --selftest [--buffers data/mel_buffers]"""
    F = Features(buffers_dir=buffers_dir, noise_dir=None)
    rng = np.random.default_rng(0)

    # 1) silence -> shape, finite, exactly 0.0
    z = F.clean(np.zeros(IN_SAMPLES, dtype=np.float32))
    assert z.shape == (1, N_MELS, N_FRAMES) and z.dtype == np.float32
    assert np.all(np.isfinite(z)) and np.allclose(z, 0.0), "silence != 0"

    # 2) late-onset tone (starts at 2.0 s) -> right-aligned: early frames 0
    t = np.arange(IN_SAMPLES) / SR
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    tone[: int(2.0 * SR)] = 0.0
    f = F.clean(tone.astype(np.float32))
    assert np.isfinite(f).all() and f[0, :, 149].max() > 0.5
    assert f[0, :, :25].max() < 1e-3, "late tone leaked into early frames"

    # 3) policy v2: 5.0 s clip, speech in [0, 1.4], t1=1.4 -> tone INSIDE
    #    the 3.0 s window (ends at 1.9 s), and file-end fallback LOSES it.
    long = np.zeros(int(5.0 * SR), dtype=np.float32)
    tl = np.arange(int(1.4 * SR)) / SR
    long[: int(1.4 * SR)] = 0.5 * np.sin(2 * np.pi * 440 * tl)
    f_anchor = F.clean(long, t1=1.4)
    f_file = F.clean(long, t1=None)
    # anchored window = [0, 1.9 s] right-aligned -> tone inside the window
    assert f_anchor[0].max() > 0.5, "speech-end anchor lost the tone"
    # file-end window = [2.0, 5.0 s] -> pure silence (the v1 bug)
    assert f_file[0].max() < 0.05, "file-end fallback should be ~silence"

    # 4) determinism: same input + seed -> identical features
    w = (0.3 * rng.standard_normal(IN_SAMPLES)).astype(np.float32)
    f1 = Features(buffers_dir=buffers_dir, rng=np.random.default_rng(7))
    f2 = Features(buffers_dir=buffers_dir, rng=np.random.default_rng(7))
    assert np.array_equal(f1.train(w, t1=2.5), f2.train(w, t1=2.5))

    # 5) warp: r=1.0 identity; r!=1.0 keeps the last frame (tail-anchored)
    base = F.clean(w)
    assert np.array_equal(time_warp_right(base, 1.0), base)
    warped = time_warp_right(base, 1.05)
    assert np.allclose(warped[0, :, -1], base[0, :, -1])

    # 6) train path without noise bank: warp+gain only, finite
    ft = f1.train(w, t1=2.5)
    assert ft.shape == base.shape and np.isfinite(ft).all()

    # 7) OOS lanes
    s = F.oos_silence()
    assert np.allclose(s, 0.0)
    print("selftest OK: shape (1,80,150) f32 | silence=0 | right-align OK |"
          " speech-end anchor OK | deterministic | tail-anchored warp OK")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    buf = "data/mel_buffers"
    if "--selftest" in args:
        if "--buffers" in args:
            buf = args[args.index("--buffers") + 1]
        selftest(buf)