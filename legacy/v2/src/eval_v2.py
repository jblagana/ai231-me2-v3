"""V2 eval — command acc + per-class + slot acc + confusions, on EVAL-SYN.

Port of V1's eval_a2.py (GPU feature precompute, per-class command + slot
breakdowns, top confusions, JSON report). Label space must match the
checkpoint: --merge-10 for 10-class checkpoints.

Usage (HPC):
  python src/eval_v2.py --model v2cnn --ckpt runs/v2a/v2cnn_11c/vcm_v2cnn_11c_best.pt \
      --data ~/vcm/data/raw_v2 --split eval
  python src/eval_v2.py --model bcresnet --merge-10 --ckpt runs/v2a/bcresnet_10c/... \
      --data ~/vcm/data/raw_v2 --split eval
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from commands import CLASSES, CLASSES_10  # noqa: E402
from slots import PARAMETRIC, SLOT_VOCAB  # noqa: E402
from models import build_model, N_MELS, SR  # noqa: E402
from train_v2 import (preload_raw,  # noqa: E402
                      precompute_base_features, VCMDatasetV2)
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v2cnn", "bcresnet"], default="v2cnn")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", default="data/raw_v2")
    ap.add_argument("--split", default="test",
                    help="manifest split: 'val' (selection) or 'test' "
                         "(sealed one-shot) — protocol 10-02")
    ap.add_argument("--merge-10", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--calib", action="store_true",
                    help="slot-confidence calibration: accept-err vs ask-rate "
                         "at margin thresholds (command-correct clips only)")
    ap.add_argument("--slot-only", action="store_true",
                    help="Run D checkpoint: SlotOnlyBCResNet (no command "
                         "head; true-class gate, slot metrics only)")
    ap.add_argument("--cmd-only", action="store_true",
                    help="v2f: CmdOnlyBCResNet — command metrics only")
    ap.add_argument("--pair-cmd", default=None,
                    help="v2f pipeline: command checkpoint; with "
                         "--slot-only adds PIPELINE slot acc (cmd correct "
                         "AND slot correct — what the Pi achieves)")
    args = ap.parse_args()

    classes = CLASSES_10 if args.merge_10 else list(CLASSES)
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"no checkpoint at {ckpt}")
        return 2
    root = Path(args.data)

    ds = VCMDatasetV2(root, args.split, augment=False, merge10=args.merge_10)
    print(f"eval split={args.split}  n={len(ds)}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    preload_raw(root)
    from torchaudio.transforms import MelSpectrogram
    gpu_mel = MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=320,
                             n_mels=N_MELS, f_min=50.0, f_max=8000.0).to(dev)
    precompute_base_features(ds, dev, gpu_mel)
    del gpu_mel
    va = DataLoader(ds, batch_size=args.batch, num_workers=0)

    if args.slot_only:
        from models import SlotOnlyBCResNet
        m = SlotOnlyBCResNet(len(classes)).to(dev)
    elif args.cmd_only:
        from models import CmdOnlyBCResNet
        m = CmdOnlyBCResNet(len(classes)).to(dev)
    else:
        m = build_model(args.model, len(classes)).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    m.eval()
    cmd_m = None
    if args.pair_cmd:
        # v2f pipeline: the command model that gates which slot head fires
        from models import CmdOnlyBCResNet
        cckpt = Path(args.pair_cmd)
        sd = torch.load(cckpt, map_location=dev)
        arch = (CmdOnlyBCResNet if "slot_heads.0.fc.weight" not in sd
                else BCResNet)
        cmd_m = arch(len(classes)).to(dev)
        cmd_m.load_state_dict(sd)
        cmd_m.eval()
        print(f"pair cmd model: {cckpt}")

    cmd_correct = cmd_total = 0
    slot_correct = slot_total = 0
    per_cls = defaultdict(lambda: [0, 0])
    slot_per_cls = defaultdict(lambda: [0, 0])
    confusion = Counter()
    slot_confusion = defaultdict(Counter)
    slot_pairs = defaultdict(Counter)  # cls -> Counter[(true, pred)] for ALL samples
    calib_data = defaultdict(list) if args.calib else None  # cls -> [(margin, ok)]

    pipe_correct = pipe_total = 0  # cmd correct AND slot correct (the Pi)
    t0 = time.time()
    with torch.no_grad():
        for x, y_cmd, y_slot in va:
            x, y_cmd, y_slot = x.to(dev), y_cmd.to(dev), y_slot.to(dev)
            cgate = None
            if args.cmd_only:
                pred = m(x).argmax(1)
                cmd_correct += (pred == y_cmd).sum().item()
                cmd_total += len(y_cmd)
                pred = pred.cpu()
                y_cmd = y_cmd.cpu()
                for i in range(len(y_cmd)):
                    t = classes[y_cmd[i]]
                    p = classes[pred[i]]
                    per_cls[t][1] += 1
                    if t == p:
                        per_cls[t][0] += 1
                    else:
                        confusion[(t, p)] += 1
                continue
            if args.slot_only:
                slot_feat = m(x)
                pred = y_cmd  # no command head — gate on true class
                if cmd_m is not None:
                    cgate = cmd_m(x).argmax(1)  # pipeline: real cmd pred
            else:
                cmd_logits, slot_feat = m(x)
                pred = cmd_logits.argmax(1)
                cmd_correct += (pred == y_cmd).sum().item()
                cmd_total += len(y_cmd)
                cgate = pred  # its own command head gates the pipeline
            for cls in PARAMETRIC:
                ci = classes.index(cls)
                mask = (y_cmd == ci) & (y_slot >= 0)
                if mask.any():
                    sl = m.slot_logits(slot_feat[mask], cls)
                    sp = sl.argmax(1)
                    ys = y_slot[mask]
                    slot_correct += (sp == ys).sum().item()
                    slot_total += int(mask.sum().item())
                    if cgate is not None:
                        pipe_correct += ((cgate[mask] == ci)
                                         & (sp == ys)).sum().item()
                        pipe_total += int(mask.sum().item())
                    slot_per_cls[cls][1] += int(mask.sum().item())
                    slot_per_cls[cls][0] += int((sp == ys).sum().item())
                    if args.calib:
                        i_pc = (pred[mask] == ci).nonzero().ravel()
                        if len(i_pc):
                            slp = torch.softmax(sl[i_pc], 1)
                            top2 = slp.topk(2, dim=1).values
                            for mg, c in zip((top2[:, 0] - top2[:, 1]).cpu(),
                                             (sp[i_pc] == ys[i_pc]).cpu()):
                                calib_data[cls].append((float(mg), float(c)))
                    sp = sp.cpu()
                    ys = ys.cpu()
                    for i in range(len(ys)):
                        t_lab = SLOT_VOCAB[cls][ys[i]]
                        p_lab = SLOT_VOCAB[cls][sp[i]]
                        slot_pairs[cls][(t_lab, p_lab)] += 1
                        if sp[i] != ys[i]:
                            slot_confusion[cls][(t_lab, p_lab)] += 1
            pred = pred.cpu()
            y_cmd = y_cmd.cpu()
            for i in range(len(y_cmd)):
                t = classes[y_cmd[i]]
                p = classes[pred[i]]
                per_cls[t][1] += 1
                if t == p:
                    per_cls[t][0] += 1
                else:
                    confusion[(t, p)] += 1
    dt = time.time() - t0

    cmd_acc = cmd_correct / max(cmd_total, 1)
    slot_acc = slot_correct / max(slot_total, 1)
    pipe_acc = (pipe_correct / pipe_total) if pipe_total else None
    kind = "slot-only" if args.slot_only else f"2head {args.model}"
    print(f"\n=== V2 EVAL ({args.split}, {kind}, "
          f"{'10c' if args.merge_10 else '11c'}) ===")
    if not args.slot_only:
        print(f"command acc: {cmd_acc:.4f}  ({cmd_correct}/{cmd_total})   "
              f"[v1i baseline: 0.8618]")
    print(f"slot acc:    {slot_acc:.4f}  ({slot_correct}/{slot_total})   "
          f"(parametric clips with a slot)")
    if pipe_total:
        print(f"PIPELINE slot: {pipe_acc:.4f}  ({pipe_correct}/{pipe_total})   "
              f"(cmd correct AND slot correct - the Pi)")
    print(f"eval time: {dt:.0f}s")

    if not args.slot_only:
        print("\nper-class command acc:")
        for cls in classes:
            c, t = per_cls[cls]
            print(f"  {cls:18s} {c / t if t else 0:.3f}  ({c}/{t})")
    else:
        per_cls = {}

    print("\nper-class slot acc:")
    for cls in PARAMETRIC:
        c, t = slot_per_cls[cls]
        print(f"  {cls:18s} {c / t if t else 0:.3f}  ({c}/{t})")

    if not args.slot_only:
        print("\ntop command confusions (true -> pred):")
        for (t, p), n in confusion.most_common(10):
            print(f"  {t:18s} -> {p:18s}  {n}")
    else:
        confusion = {}

    print("\ntop slot confusions (class: true -> pred):")
    for cls in PARAMETRIC:
        if slot_confusion[cls]:
            for (t, p), n in slot_confusion[cls].most_common(3):
                print(f"  {cls}: {t!r} -> {p!r}  {n}")

    for cls in PARAMETRIC:
        vocab = SLOT_VOCAB[cls]
        if len(vocab) > 10 or not slot_pairs[cls]:
            continue  # big vocabs (alarm 48) -> JSON only
        print(f"\nslot matrix {cls} (rows=true, cols=pred):")
        print("  " + " " * 10 + " ".join(f"{v[:8]:>9s}" for v in vocab))
        for t in vocab:
            cells = " ".join(f"{slot_pairs[cls].get((t, p), 0):>9d}"
                             for p in vocab)
            print(f"  {t[:10]:10s} {cells}")

    calib_json = None
    if args.calib:
        print("\n=== SLOT CONFIDENCE CALIBRATION (command-correct clips; "
              "accept if top1-top2 margin >= θ) ===")
        calib_json = {}
        all_data = []
        for cls in PARAMETRIC:
            all_data.extend(calib_data.get(cls, []))
        for label, data in [("ALL", all_data)] + \
                [(cls, calib_data.get(cls, [])) for cls in PARAMETRIC]:
            if not data:
                continue
            n = len(data)
            acc0 = sum(c for _, c in data) / n
            print(f"\n  {label:16s} n={n:5d}  base-err {1 - acc0:7.3f}")
            print(f"    θ      ask%   accept-err   ask-err")
            rows = {}
            for th in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
                acc = [c for mg, c in data if mg >= th]
                ask = [c for mg, c in data if mg < th]
                a_err = 1 - sum(acc) / len(acc) if acc else float("nan")
                q_err = 1 - sum(ask) / len(ask) if ask else float("nan")
                q_rate = len(ask) / n
                print(f"    {th:4.2f}   {q_rate:5.1%}   {a_err:10.3f}   "
                      f"{q_err:9.3f}")
                rows[f"{th:.2f}"] = {"ask_rate": round(q_rate, 4),
                                     "accept_err": round(a_err, 4) if acc
                                     else None,
                                     "ask_err": round(q_err, 4) if ask
                                     else None}
            calib_json[label] = {"n": n, "base_err": round(1 - acc0, 4),
                                 "by_threshold": rows}

    out = ckpt.parent / f"eval_{args.split}.json"
    out.write_text(json.dumps({
        "split": args.split, "n": cmd_total,
        "cmd_acc": cmd_acc if not args.slot_only else None,
        "slot_acc": slot_acc,
        "pipe_acc": pipe_acc,
        "pipe_n": pipe_total,
        "per_class_cmd": {k: {"acc": v[0] / v[1] if v[1] else 0,
                              "n": v[1]} for k, v in per_cls.items()},
        "per_class_slot": {k: {"acc": v[0] / v[1] if v[1] else 0,
                               "n": v[1]} for k, v in slot_per_cls.items()},
        "top_confusions": (
            {f"{t}->{p}": n for (t, p), n in confusion.most_common(20)}
            if not args.slot_only else {}),
        "slot_confusions": {
            cls: {f"{t}->{p}": n
                  for (t, p), n in slot_confusion[cls].most_common(50)}
            for cls in PARAMETRIC if slot_confusion[cls]
        },
        "slot_matrix": {
            cls: {t: {p: slot_pairs[cls].get((t, p), 0) for p in SLOT_VOCAB[cls]}
                  for t in SLOT_VOCAB[cls]}
            for cls in PARAMETRIC if slot_pairs[cls]
        },
        "calib": calib_json,
    }, indent=1))
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())