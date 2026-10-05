"""Export the V3 wake-gate model to ONNX (the Pi's wake_model.onnx).

Ported from legacy/v2/src/export_wake.py onto the V3 stack (ratified
2026-10-05). 2-class model (0 = no_wake, 1 = wake "hey boots"); only the
command head is exported — the 6 slot heads are random/untrained and
unreachable from the output, so the tracer drops them.

Frozen export spec (matches the V2 wake_meta.json the Pi already reads):
  opset 17
  input  : precomputed log10 mel, (1, 80, 150)   [3.0 s, 16 kHz, 20 ms hop]
  output : wake_logits (1, 2)
  meta   : wake_meta.json with threshold 0.5 (pi_demo.py overrides from meta)

The Pi package is TWO ONNX models sharing ONE 80-mel feature:
  tools/export_onnx.py  -> models/me2_vcm_v3.onnx (command + slots)
  tools/export_wake.py  -> models/me2_vcm_v3_wake.onnx (this file)

Run on the box with torch + onnxruntime (HPC conda `vcm`):
  ~/.conda/envs/vcm/bin/python tools/export_wake.py \
      --checkpoint runs/v3w/bcresnet/vcm_wake_bcresnet_best.pt \
      --out models/me2_vcm_v3_wake.onnx --verify
  # --random : random weights (pipeline validation before training)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import IN_SAMPLES, N_FRAMES, N_MELS, SR  # noqa: E402
from models import build_model, param_count  # noqa: E402

CLASSES = ["no_wake", "wake"]
OPSET = 17


class WakeExporter(nn.Module):
    """Precomputed log-mel (1, 80, 150) -> wake logits (1, 2).

    The Pi feeds (1, 80, 150); the net wants (1, 1, 80, 150), so the
    channel dim is inserted here (same pattern as export_onnx.py).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, logmel: torch.Tensor):
        if logmel.dim() == 3:
            logmel = logmel.unsqueeze(1)
        cmd_logits, _ = self.model(logmel)
        return cmd_logits


def export(model_name: str, ckpt, out_path: Path, verify: bool) -> bool:
    model = build_model(model_name, 2)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        # strict=False: wake checkpoints may carry a DIFFERENT slot-head
        # vocab (the V2 wake ckpt has V2 slot heads). The wake graph only
        # uses backbone + cmd head, so slot heads are irrelevant — but
        # EVERYTHING else must load exactly.
        res = model.load_state_dict(sd, strict=False)
        bad = [k for k in res.missing_keys + res.unexpected_keys
               if not k.startswith("slot_heads.")]
        if bad:
            raise RuntimeError(f"non-slot keys mismatched: {bad}")
        if res.missing_keys or res.unexpected_keys:
            print(f"  note: slot heads replaced by fresh init "
                  f"({len(res.missing_keys)} keys) — unused by the wake graph")
    model.eval()
    print(f"exporting wake {model_name} -> {out_path}")
    exp = WakeExporter(model).eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            exp, (torch.randn(1, N_MELS, N_FRAMES),), out_path,
            input_names=["mel"], output_names=["wake_logits"],
            dynamic_axes=None, opset_version=OPSET,
            do_constant_folding=True, dynamo=False,
        )
    print(f"  onnx: {out_path.stat().st_size / 1024:.1f} KB")
    exp.eval()  # legacy exporter can flip the module to train mode
    meta = {
        "model": model_name, "n_classes": 2, "classes": CLASSES,
        "threshold": 0.5, "opset": OPSET, "sr": SR, "in_samples": IN_SAMPLES,
        "window_s": 3.0, "n_mels": N_MELS, "n_frames": N_FRAMES,
        "params": int(param_count(model)),
        "ckpt": str(ckpt) if ckpt else None,
    }
    (out_path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=1))
    print(f"  meta: {out_path.with_suffix('.meta.json')}")

    ok = True
    if verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print("verify skipped: onnxruntime not installed")
            return 0
        sess = ort.InferenceSession(str(out_path),
                                    providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        worst = 0.0
        with torch.no_grad():
            for _ in range(4):
                x = torch.randn(1, N_MELS, N_FRAMES)
                t_out = exp(x).numpy()
                o_out = np.asarray(sess.run(None, {in_name: x.numpy()}))
                worst = max(worst, float(np.abs(t_out - o_out).max()))
        print(f"  onnx-vs-torch max diff: {worst:.2e}  "
              f"({'OK' if worst < 1e-3 else 'FAIL'})")
        ok = worst < 1e-3
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v2cnn", "bcresnet"], default="bcresnet")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="models/me2_vcm_v3_wake.onnx")
    ap.add_argument("--random", action="store_true",
                    help="random weights (pipeline validation)")
    ap.add_argument("--verify", action="store_true",
                    help="ONNX Runtime vs torch parity")
    args = ap.parse_args()
    if not args.random and not args.checkpoint:
        ap.error("--checkpoint required unless --random")
    t0 = time.time()
    ok = export(args.model, None if args.random else args.checkpoint,
                Path(args.out), args.verify)
    print(f"EXPORT-WAKE: {'PASS' if ok else 'FAIL'} in {time.time() - t0:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
