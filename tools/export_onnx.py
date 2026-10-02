"""Export the V3 command+slot model to ONNX (the Pi package model).

Frozen export spec (DECISIONS / BENCHMARK):
  opset 17
  input  : precomputed log10 mel, (1, 80, 150)   [3.0 s, 16 kHz, 20 ms hop]
  output : command logits (20) + 6 slot logits, one per head (3 each)

The Pi package is TWO ONNX models (wake + command/slot) sharing ONE 80-mel
feature; this exports the command/slot model. The wake model is reused from V2
(JOURNAL "Capture policy").

Run on the box with torch + onnxruntime (HPC conda `vcm`):
  ~/.conda/envs/vcm/bin/python tools/export_onnx.py \
      --checkpoint runs/v3r1/checkpoint_best.pt \
      --out models/me2_vcm_v3.onnx --verify
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from commands import CLASSES            # noqa: E402
from models import build_model, param_count  # noqa: E402
from slots import PARAMETRIC, SLOT_COUNTS    # noqa: E402

N_CLASSES = len(CLASSES)                # 20


class ExportWrapper(nn.Module):
    """Single graph -> (command_logits, slot_<head1>, ..., slot_<head6>).

    The Pi feature is (B, 80, 150) log-mel (mel.py); the BC-ResNet consumes
    (B, 1, 80, 150), so insert the channel dim here. Keeping the ONNX input at
    (B, 80, 150) means the shipped Pi mel.py / pi_demo.py are unchanged.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        cmd, slot_feat = self.model(x)
        outs = [cmd]
        for name in PARAMETRIC:
            outs.append(self.model.slot_logits(slot_feat, name))
        return tuple(outs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="models/me2_vcm_v3.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", action="store_true",
                    help="ONNX Runtime vs torch parity (needs onnxruntime)")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    arch = ckpt.get("arch", "bcresnet")
    scale = ckpt.get("scale", 2)
    model = build_model(arch, N_CLASSES, scale=scale)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded {args.checkpoint}  arch={arch} scale={scale}  "
          f"params={param_count(model):,}  epoch={ckpt.get('epoch')}")

    wrapper = ExportWrapper(model).eval()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    example = torch.zeros(1, 80, 150)
    input_names = ["mel"]
    output_names = ["command_logits"] + [f"slot_{n}" for n in PARAMETRIC]

    # Prefer the legacy TorchScript exporter (dynamo=False): stable across
    # torch 2.x and does not require onnxscript (the dynamo exporter does).
    try:
        torch.onnx.export(
            wrapper, (example,), out,
            input_names=input_names, output_names=output_names,
            opset_version=args.opset, verbose=False,
            dynamo=False,
        )
    except TypeError:
        # torch < 2.1 has no `dynamo` kwarg (legacy exporter is the default).
        torch.onnx.export(
            wrapper, (example,), out,
            input_names=input_names, output_names=output_names,
            opset_version=args.opset, verbose=False,
        )
    size_kb = out.stat().st_size / 1024
    with torch.no_grad():
        touts = wrapper(example)
    print(f"wrote {out}  ({size_kb:.1f} KB)  opset={args.opset}  "
          f"output shapes {[tuple(t.shape) for t in touts]}")

    meta = {
        "arch": arch, "scale": scale, "n_classes": N_CLASSES,
        "heads": {n: SLOT_COUNTS[n] for n in PARAMETRIC},
        "feature": "log10 mel (1, 80, 150); 3.0 s, 16 kHz, 20 ms hop",
        "opset": args.opset, "input": input_names, "outputs": output_names,
        "params": int(param_count(model)), "size_kb": round(size_kb, 1),
        "checkpoint": str(args.checkpoint), "epoch": ckpt.get("epoch"),
        "git_hash": ckpt.get("git_hash"),
        "feature_policy": ckpt.get("feature_policy"),
    }
    with open(out.with_suffix(".meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out.with_suffix('.meta.json')}")

    if args.verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print("verify skipped: onnxruntime not installed")
            return 0
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        # input is fixed-shape (1, 80, 150) — the Pi always feeds one clip
        x = np.random.default_rng(0).standard_normal((1, 80, 150)).astype(np.float32)
        o_onnx = sess.run(None, {"mel": x})
        with torch.no_grad():
            o_torch = wrapper(torch.from_numpy(x))
        ok = True
        for i, name in enumerate(output_names):
            d = float(np.max(np.abs(np.asarray(o_onnx[i]) - o_torch[i].numpy())))
            ok = ok and d < 1e-4
            print(f"  parity {name}: max|d|={d:.2e}  {'OK' if d < 1e-4 else 'FAIL'}")
        print("PARITY:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
