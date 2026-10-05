import wave, numpy as np, os
for n in ["time_lead", "weather_lead", "pause", "next", "stop"]:
    p = n + ".wav"
    if not os.path.exists(p):
        print(n, "MISSING"); continue
    with wave.open(p, "rb") as wf:
        sr = wf.getframerate()
        a = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    seg = a[: int(sr * 0.05)]
    # leading silence: first sample index where |amp|>200
    nz = np.argmax(np.abs(a) > 200) if (np.abs(a) > 200).any() else -1
    print(n, "sr", sr, "dur", round(len(a) / sr, 2),
          "first8", a[:8].tolist(),
          "rms50ms", int(seg.std()),
          "lead_silence_s", round(nz / sr, 3) if nz >= 0 else "none")
