"""V2 wake-gate ONNX export — adds wake_model.onnx + wake_meta.json to a
Pi package dir (the VCM payload from export_v2.py).

2-class model (0 = no_wake, 1 = wake). Only the command head is exported —
the slot heads are unreachable from the output, so the tracer drops them.
Same locked feature: precomputed log-mel (1, 80, 150), same mel buffers.

Usage (HPC, vcm env):
  python src/export_wake.py --model v2cnn \
      --ckpt runs/v2w/v2cnn/vcm_wake_v2cnn_best.pt --pkg <pi-pkg-dir>
  # writes <pkg>/wake_model.onnx + <pkg>/wake_meta.json, parity-checks
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

sys.path.insert(0, str(Path(__file__).parent))
from export_v2 import (OPSET, Exporter, parity_check,  # noqa: E402
                       write_mel_buffers)
from mel import IN_SAMPLES  # noqa: E402
from models import N_MELS, N_FRAMES, SR, build_model  # noqa: E402

CLASSES = ["no_wake", "wake"]


class WakeExporter(nn.Module):
    """Precomputed log-mel (1, 80, 150) -> wake logits (1, 2)."""

    def __init__(self, model_name: str, ckpt: str = None):
        super().__init__()
        self.model = build_model(model_name, 2)
        if ckpt:
            self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()

    def forward(self, logmel: torch.Tensor):
        cmd_logits, _ = self.model(logmel.unsqueeze(1))
        return cmd_logits


def export(model: str, ckpt, pkg_dir: Path) -> bool:
    exp = WakeExporter(model, ckpt)
    onnx_path = pkg_dir / "wake_model.onnx"
    print(f"exporting wake {model} -> {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            exp, (torch.randn(1, N_MELS, N_FRAMES),), onnx_path,
            input_names=["logmel"], output_names=["wake_logits"],
            dynamic_axes=None, opset_version=OPSET,
            do_constant_folding=True, dynamo=False,
        )
    print(f"  onnx: {onnx_path.stat().st_size / 1024:.1f} KB")
    exp.eval()
    write_mel_buffers(pkg_dir)  # idempotent; no-ops if VCM already wrote them
    meta = {
        "model": model, "n_classes": 2, "classes": CLASSES,
        "threshold": 0.5, "opset": OPSET, "sr": SR, "in_samples": IN_SAMPLES,
        "window_s": 3.0, "n_mels": N_MELS, "n_frames": N_FRAMES,
        "ckpt": str(ckpt) if ckpt else None,
    }
    (pkg_dir / "wake_meta.json").write_text(json.dumps(meta, indent=1))
    d = parity_check(exp, onnx_path)
    print(f"  onnx-vs-torch max diff: {d:.2e}  ({'OK' if d < 1e-3 else 'FAIL'})")
    return d < 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v2cnn", "bcresnet"], default="v2cnn")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pkg", required=True, help="Pi package dir to extend")
    ap.add_argument("--random", action="store_true")
    args = ap.parse_args()
    if not args.random and not args.ckpt:
        ap.error("--ckpt required unless --random")
    pkg = Path(args.pkg)
    pkg.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok = export(args.model, None if args.random else args.ckpt, pkg)
    print(f"EXPORT-WAKE: {'PASS' if ok else 'FAIL'} in {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
