"""Final Pi-path gate (port of V1's verify_pi.py): verified, not trusted.

A feature mismatch is the #1 silent on-device accuracy killer, so before
the chosen model ships to the Pi we re-check, on REAL clips:
  1. numpy mel (src/mel.py + shipped .npy buffers) == torch training mel
  2. torch model (checkpoint) == ONNX (onnxruntime CPU)
  3. full path: wav -> numpy mel -> ONNX  ==  wav -> torch mel -> torch
Both diffs must be < 1e-3.

Usage (HPC, vcm env), after selection:
  python tools/verify_pi_v2.py --pkg runs/v2a/<winner>_pi/me2 \
      --ckpt runs/v2a/<winner>/best.pt --data ~/vcm/data/raw_v2/train
  # --data: any dir tree containing 16k .raw clips (walks recursively)
  # without --data: synthetic clips (noise + tones) — still catches mel
  # buffer and export bugs, just not real-audio edge cases
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mel import Mel, IN_SAMPLES, load_wav16k  # noqa: E402
from models import build_model  # noqa: E402
from slots import PARAMETRIC  # noqa: E402
from train_v2 import wav_to_logmel  # noqa: E402

TOL = 1e-3


def synth_clips(n: int, seed: int = 0):
    """n synthetic (48000,) float32 clips: noise, tones, mixed."""
    rng = np.random.default_rng(seed)
    clips = []
    for i in range(n):
        kind = i % 3
        if kind == 0:
            w = (rng.standard_normal(IN_SAMPLES) * 0.1).astype(np.float32)
        elif kind == 1:
            t = np.arange(IN_SAMPLES) / 16000.0
            w = (0.3 * np.sin(2 * np.pi * 300 * t)
                 + 0.2 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
        else:
            w = (rng.standard_normal(IN_SAMPLES) * 0.05).astype(np.float32)
            burst = int(0.2 * IN_SAMPLES)
            t = np.arange(burst) / 16000.0
            w[10000:10000 + burst] += 0.4 * np.sin(2 * np.pi * 800 * t)
        clips.append(w)
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="Pi payload dir (from export_v2)")
    ap.add_argument("--ckpt", required=True, help="torch checkpoint the onnx came from")
    ap.add_argument("--data", default=None, help="dir tree with 16k .raw clips")
    ap.add_argument("--clips", type=int, default=8)
    args = ap.parse_args()

    pkg = Path(args.pkg)
    meta = json.loads((pkg / "meta.json").read_text())
    mel = Mel(str(pkg / "mel_window.npy"), str(pkg / "mel_fb.npy"))
    model = build_model(meta["model"], meta["n_classes"])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(pkg / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    if args.data:
        raws = sorted(Path(args.data).rglob("*.raw"))[:args.clips]
        clips = [load_wav16k(str(p)) for p in raws]
        src = f"{len(clips)} real .raw clips under {args.data}"
    else:
        clips = synth_clips(args.clips)
        src = f"{len(clips)} synthetic clips"

    print(f"verifying {meta['model']} ({meta['n_classes']}c) on {src}")
    worst_feat = worst_onnx = worst_full = 0.0
    for i, wav in enumerate(clips):
        feat_torch = wav_to_logmel(torch.from_numpy(wav))       # (1,80,150)
        feat_np = mel(wav)                                       # (1,80,150)
        worst_feat = max(worst_feat,
                         float((feat_torch - torch.from_numpy(feat_np)).abs().max()))
        with torch.no_grad():
            t_cmd, t_sf = model(feat_torch.unsqueeze(1))
            t_outs = [t_cmd] + [model.slot_logits(t_sf, c) for c in PARAMETRIC]
        o_outs = sess.run(None, {in_name: feat_np.astype(np.float32)})
        worst_onnx = max(worst_onnx,
                         max(float(np.abs(a.numpy() - b).max())
                             for a, b in zip(t_outs, o_outs)))
        # full path: numpy mel -> onnx vs torch mel -> torch
        worst_full = max(worst_full,
                         max(float(np.abs(np.asarray(a) - b.numpy()).max())
                             for a, b in zip(o_outs, t_outs)))
        print(f"  clip {i}: feat={float((feat_torch - torch.from_numpy(feat_np)).abs().max()):.2e}"
              f"  onnx={max(float(np.abs(a.numpy()-b).max()) for a, b in zip(t_outs, o_outs)):.2e}")

    ok = worst_feat < TOL and worst_onnx < TOL and worst_full < TOL
    print(f"\nWORST numpy-mel diff = {worst_feat:.2e}")
    print(f"WORST onnx-vs-torch  = {worst_onnx:.2e}")
    print(f"WORST full-path      = {worst_full:.2e}   (tol {TOL})")
    print(f"PI-PATH VERIFY: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
