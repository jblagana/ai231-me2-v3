"""V2 ONNX export — Pi-ready package (port of V1's export_a2.py).

Input = PRECOMPUTED log-mel (1, 80, 150). The mel is computed on the Pi in
numpy (src/mel.py) from the shipped mel_window.npy (1024,) + mel_fb.npy
(513,80). The graph is conv/pool/linear/batchnorm only — no FFT (the
exporter can't emit rfft) — and every V2 op is ONNX-native at opset 17,
including SubSpectralNorm (it is just reshape -> BatchNorm2d -> reshape).

Outputs (8): cmd_logits (1, n_classes) + one slot vector per PARAMETRIC
class. The Pi picks the slot output matching the predicted command.

Usage (HPC, vcm env):
  # final, after selection:
  python src/export_v2.py --model v2cnn --n-classes 11 \
      --ckpt runs/v2a/v2cnn_11c/best.pt --out runs/v2a/v2cnn_11c_pi/me2 \
      --wav <any 16k .raw>          # self-verifies on that clip
  # writes <out>/: model.onnx, mel_window.npy, mel_fb.npy, meta.json,
  #                mel.py, pi_demo.py   (one dir = the whole Pi payload)
  # --random : random weights (pipeline validation before training finishes)
  # --smoke  : both models x {11,10}c with random weights to a temp dir,
  #            ONNX-vs-torch + numpy-mel-vs-torch parity, exit 0/1
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

sys.path.insert(0, str(Path(__file__).parent))
from commands import CLASSES, CLASSES_10  # noqa: E402
from mel import Mel, IN_SAMPLES, N_FFT, HOP  # noqa: E402
from models import N_MELS, N_FRAMES, SR, build_model, MODEL_ZOO  # noqa: E402
from slots import PARAMETRIC, SLOT_VOCAB, SLOT_COUNTS  # noqa: E402
from train_v2 import wav_to_logmel  # noqa: E402

OPSET = 17


class Exporter(nn.Module):
    """Takes a precomputed log-mel (1, 80, 150) -> (cmd_logits, slot heads)."""

    def __init__(self, model_name: str, n_classes: int, ckpt: str = None):
        super().__init__()
        self.model = build_model(model_name, n_classes)
        if ckpt:
            self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()

    def forward(self, logmel: torch.Tensor):
        x = logmel.unsqueeze(1)  # (1, 1, 80, 150)
        cmd_logits, slot_feat = self.model(x)
        outs = [cmd_logits]
        for cls in PARAMETRIC:
            outs.append(self.model.slot_logits(slot_feat, cls))
        return tuple(outs)


def class_names(n_classes: int):
    return CLASSES if n_classes == len(CLASSES) else CLASSES_10


def write_mel_buffers(out_dir: Path):
    """Ship the EXACT training mel window + filterbank (same extraction as
    V1's export_a2.py — the numpy mel is validated against these in
    verify_pi_v2.py before shipping)."""
    ref = MelSpectrogram(sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
                         n_mels=N_MELS, f_min=50.0, f_max=8000.0)
    _ = ref(torch.randn(1, IN_SAMPLES))
    np.save(out_dir / "mel_window.npy", ref.spectrogram.window.numpy())
    np.save(out_dir / "mel_fb.npy", ref.mel_scale.fb.numpy())


def parity_check(exp: Exporter, onnx_path: Path, n: int = 4,
                 tol: float = 1e-3) -> float:
    """ONNX-vs-torch max abs diff over n random log-mel inputs."""
    import onnxruntime as ort
    exp.eval()  # guard: the legacy ONNX exporter leaves the module in train mode
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    worst = 0.0
    with torch.no_grad():
        for _ in range(n):
            x = torch.randn(1, N_MELS, N_FRAMES)
            t_out = [o.numpy() for o in exp(x)]
            o_out = sess.run(None, {in_name: x.numpy().astype(np.float32)})
            worst = max(worst, max(float(np.abs(a - b).max())
                                   for a, b in zip(t_out, o_out)))
    return worst


def export(args, out_dir: Path, ckpt: str = None, n_classes: int = 11,
           verify_wav: str = None) -> bool:
    exp = Exporter(args.model, n_classes, ckpt)
    onnx_path = out_dir / "model.onnx"
    print(f"exporting {args.model} ({n_classes}c) -> {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            exp, (torch.randn(1, N_MELS, N_FRAMES),), onnx_path,
            input_names=["logmel"],
            output_names=["cmd_logits"] + [f"slot_{c}" for c in PARAMETRIC],
            dynamic_axes=None, opset_version=OPSET,
            do_constant_folding=True, dynamo=False,
        )
    print(f"  onnx: {onnx_path.stat().st_size / 1024:.1f} KB")
    exp.eval()  # torch>=2.9 legacy exporter flips the module to TRAIN mode
    write_mel_buffers(out_dir)
    names = class_names(n_classes)
    meta = {
        "model": args.model, "n_classes": n_classes, "classes": names,
        "parametric": list(PARAMETRIC), "slot_vocab": SLOT_VOCAB,
        "slot_counts": SLOT_COUNTS,
        "outputs": ["cmd_logits"] + [f"slot_{c}" for c in PARAMETRIC],
        "opset": OPSET, "sr": SR, "in_samples": IN_SAMPLES,
        "window_s": 3.0, "n_mels": N_MELS, "n_frames": N_FRAMES,
        "ckpt": ckpt,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    # ship the Pi-side code with the weights: one dir = the whole payload
    shutil.copy(Path(__file__).parent / "mel.py", out_dir / "mel.py")
    shutil.copy(Path(__file__).parent.parent / "pi" / "pi_demo.py",
                out_dir / "pi_demo.py")
    # self-verify: ONNX == torch
    d = parity_check(exp, onnx_path)
    print(f"  onnx-vs-torch max diff: {d:.2e}  ({'OK' if d < 1e-3 else 'FAIL'})")
    ok = d < 1e-3
    if verify_wav:
        from mel import load_wav16k
        wav = load_wav16k(verify_wav)
        mel = Mel(str(out_dir / "mel_window.npy"),
                  str(out_dir / "mel_fb.npy"))(wav)
        feat_diff = float((wav_to_logmel(torch.from_numpy(wav))
                           - torch.from_numpy(mel)).abs().max())
        print(f"  numpy-mel-vs-torch-mel ({Path(verify_wav).name}): "
              f"{feat_diff:.2e}  ({'OK' if feat_diff < 1e-3 else 'FAIL'})")
        ok = ok and feat_diff < 1e-3
    return ok


def smoke() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for name in MODEL_ZOO:
            for nc in (11, 10):
                d = Path(td) / f"{name}_{nc}"
                d.mkdir(parents=True)
                print(f"=== {name} {nc}c (random weights) ===")
                ok = ok and export(
                    argparse.Namespace(model=name), d, ckpt=None, n_classes=nc)
    print("EXPORT SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_ZOO), default="v2cnn")
    ap.add_argument("--n-classes", type=int, default=11, choices=(11, 10))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None, help="output dir (Pi payload)")
    ap.add_argument("--wav", default=None, help="16k .raw/.wav for self-verify")
    ap.add_argument("--random", action="store_true",
                    help="export with random weights (pipeline validation)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        sys.exit(smoke())
    if not args.out:
        ap.error("--out required (or --smoke)")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.random and not args.ckpt:
        ap.error("--ckpt required unless --random")
    ok = export(args, out_dir, ckpt=None if args.random else args.ckpt,
                n_classes=args.n_classes, verify_wav=args.wav)
    print("EXPORT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

