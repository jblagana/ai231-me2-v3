import wave, numpy as np, glob

for f in sorted(glob.glob(r"C:\Users\Jan\hpc_overnight\pi_rec\*.wav")):
    w = wave.open(f)
    sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    c = int(sr * 0.25)
    prof = [round(float(np.sqrt((x[i:i+c]**2).mean())), 4)
            for i in range(0, len(x), c)]
    print(f.split("\\")[-1])
    print(f"  sr={sr} max={float(np.abs(x).max()):.3f} rms={float(np.sqrt((x**2).mean())):.4f}")
    print(f"  0.25s-rms: {prof}")
    w.close()