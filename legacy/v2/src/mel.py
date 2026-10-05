"""Pi-side mel feature extractor — numpy only (no torch, no torchaudio).

Reproduces train_v2.wav_to_logmel EXACTLY using the shipped buffers
(mel_window.npy (1024,), mel_fb.npy (513,80)) written by src/export_v2.py.
Ported verbatim from V1's ~/vcm/mel.py, which was verified == training on
real clips (verify_pi.py) before V1 shipped — a feature mismatch is the #1
silent on-device accuracy killer, so V2 re-verifies the same way
(tools/verify_pi_v2.py) before we ship.

Pipeline (must match training bit-for-bit):
  wav (48000,) @16k float32 in [-1,1]
    -> reflect-pad 512 both sides
    -> frames (151, 1024), hop 320
    -> * window (1024,)
    -> rfft -> (151, 513) complex
    -> |.|^2 -> power (151, 513)
    -> @ fb (513, 80) -> mel (151, 80) -> transpose (80, 151)
    -> clamp(1e-5).log10() -> (x+5)/5
    -> [:, :150]  (right-aligned: drop the last frame)
  -> (1, 80, 150) float32
"""
import numpy as np

N_FFT, HOP, PAD = 1024, 320, 512
N_FRAMES = 150
IN_SAMPLES = 48000


class Mel:
    def __init__(self, window_path="mel_window.npy", fb_path="mel_fb.npy"):
        self.window = np.load(window_path).astype(np.float32)   # (1024,)
        self.fb = np.load(fb_path).astype(np.float32)           # (513, 80)

    def __call__(self, wav):
        # wav: (48000,) float32 in [-1,1]
        x = np.pad(wav, (PAD, PAD), mode="reflect")             # (49024,)
        n_frames = (len(x) - N_FFT) // HOP + 1                  # 151
        # frame matrix (n_frames, 1024)
        idx = np.arange(N_FFT)[None, :] + (HOP * np.arange(n_frames))[:, None]
        frames = x[idx]                                         # (151, 1024)
        frames = frames * self.window
        spec = np.fft.rfft(frames, axis=1)                      # (151, 513)
        power = spec.real ** 2 + spec.imag ** 2                 # (151, 513)
        mel = power @ self.fb                                   # (151, 80)
        mel = mel.T                                             # (80, 151)
        logmel = np.log10(np.clip(mel, 1e-5, None))
        logmel = (logmel + 5.0) / 5.0
        logmel = logmel[:, :N_FRAMES]                           # right-align
        return logmel.astype(np.float32)[None, :, :]            # (1, 80, 150)


def load_wav16k(path, in_samples=IN_SAMPLES):
    """Load a 16k mono clip (raw int16 or wav) -> (48000,) float32 in [-1,1],
    right-aligned (left-pad with silence if short)."""
    if path.endswith(".raw"):
        arr = np.fromfile(path, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        import soundfile as sf
        arr, sr = sf.read(path, dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            # naive linear resample (Pi has no scipy by default)
            n = int(len(arr) * 16000 / sr)
            arr = np.interp(np.linspace(0, len(arr) - 1, n),
                            np.arange(len(arr)), arr).astype(np.float32)
    if arr.size < in_samples:
        arr = np.pad(arr, (in_samples - arr.size, 0))  # right-align: pad LEFT
    return arr[:in_samples]
