"""V3 wake-gate training — 2-class (0 = no_wake, 1 = wake "hey boots").

Ported from legacy/v2/src/train_wake.py onto the V3 stack (ratified
2026-10-05): same frozen 2-class recipe (V2 run: bcresnet, 30 epochs,
eval bal 0.970 / wake recall 0.94 / no-wake FPR 0.0), but features now
flow through src/features.py — the SINGLE numpy mel the Pi ships
(bit-exact vs the V2 shipped mel buffers) and the V3 frozen augmentation
(waveform MUSAN SNR 5-25 -> mel -> tail-anchored warp 0.95-1.05 -> gain
+-10 dB). The V2 trainer used a GPU torchaudio MelSpectrogram +
mel-domain noise mix; V3's is the numpy pipeline, so training and the
Pi consume the identical feature by construction.

Only the command head is trained/used; the 6 slot heads are random and
unreachable from the wake loss (the export drops them).

Data (speaker-disjoint 20/20, same 40 voices as the V2 VCM):
  wake (1):      {wake}/{split}/*.mp3 + .raw sibling   (make_wake_tts + decode)
  no_wake (0):   {cmd}/{split}/**/*.mp3 + .raw         (the V2 VCM clips)
                 + MUSAN noise .npy (train only)
Best checkpoint by BALANCED ACCURACY on eval.

Usage (HPC, vcm env, one per GPU):
  CUDA_VISIBLE_DEVICES=2 python tools/train_wake.py --model bcresnet \
      --wake ~/vcm/data/wake_v2 --cmd ~/vcm/data/raw_v2 \
      --noise ~/vcm/data/noise16k --out runs/v3w/bcresnet
  python tools/train_wake.py --smoke    # CPU, tiny subset, 2 epochs
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import (GAIN_NORM, IN_SAMPLES, Features, SR, load_wav,  # noqa: E402
                      mix_noise, time_warp_right)
from models import build_model, param_count  # noqa: E402

WAKE_CLASSES = ["no_wake", "wake"]

_FEAT = None
BASE_FEAT_CACHE = {}


def feat() -> Features:
    """Module-level numpy mel (the Pi's exact buffers)."""
    global _FEAT
    if _FEAT is None:
        _FEAT = Features(buffers_dir=str(ROOT / "data" / "mel_buffers"),
                         load_noise=False)
    return _FEAT


def load_noise_bank(noise_dir: Path, max_sec: float = 4.0) -> list:
    """MUSAN .npy (float32 16 kHz) -> list of waveform arrays (T,) float32."""
    bank = []
    for f in sorted(Path(noise_dir).glob("*.npy")):
        w = np.load(f).astype(np.float32)
        if w.shape[-1] > int(SR * max_sec):
            w = w[..., :int(SR * max_sec)]
        bank.append(w)
    print(f"noise bank: {len(bank)} MUSAN clips cached", flush=True)
    return bank


def _waveform(p: Path) -> np.ndarray:
    """Clip -> float32 mono 16 kHz: .raw via load_wav, MUSAN .npy direct."""
    if p.suffix == ".npy":
        return np.load(p).astype(np.float32)
    return load_wav(p.with_suffix(".raw"))


def _right_align3(wav: np.ndarray) -> np.ndarray:
    """V2 window: LAST 3.0 s, right-aligned (never crop the tail)."""
    if len(wav) < IN_SAMPLES:
        return np.pad(wav, (IN_SAMPLES - len(wav), 0))
    return wav[-IN_SAMPLES:]


class WakeDataset(Dataset):
    """(path, y) items, y in {0: no_wake, 1: wake}. The .mp3 is the name
    carrier; audio is its .raw sibling (s16le 16 kHz) — the V2 layout.
    Noise items are bare .npy paths, labeled no_wake, train only."""

    def __init__(self, split: str, wake_root: Path, cmd_root: Path,
                 noise_root: Path = None, augment: bool = False,
                 noise_bank: list = None):
        self.augment = augment
        self.noise_bank = noise_bank or []
        self.items = []
        for p in sorted((Path(wake_root) / split).glob("*.mp3")):
            if p.with_suffix(".raw").exists():
                self.items.append((p, 1))
        for p in sorted((Path(cmd_root) / split).rglob("*.mp3")):
            if p.with_suffix(".raw").exists():
                self.items.append((p, 0))
        if split == "train" and noise_root is not None:
            for f in sorted(Path(noise_root).glob("*.npy")):
                self.items.append((f, 0))
        random.shuffle(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def _base_feat(self, p: Path) -> torch.Tensor:
        key = str(p)
        f = BASE_FEAT_CACHE.get(key)
        if f is None:
            f = torch.from_numpy(feat().mel(_right_align3(_waveform(p))))
            BASE_FEAT_CACHE[key] = f
        return f

    def __getitem__(self, i):
        p, y = self.items[i]
        if not self.augment:
            return self._base_feat(p), y
        # V3 frozen order: waveform noise -> mel -> warp -> gain.
        wav = _right_align3(_waveform(p))
        if self.noise_bank:
            nz = random.choice(self.noise_bank)
            if len(nz) < IN_SAMPLES:
                nz = np.tile(nz, -(-IN_SAMPLES // len(nz)))[:IN_SAMPLES]
            else:
                nz = nz[:IN_SAMPLES]
            wav = mix_noise(wav, nz, random.uniform(5.0, 25.0))
        x = feat().mel(wav)
        x = time_warp_right(torch.from_numpy(x),
                            random.uniform(0.95, 1.05))
        x = x + random.uniform(-GAIN_NORM, GAIN_NORM)
        return torch.from_numpy(x), y


def _forward(m, x: torch.Tensor) -> torch.Tensor:
    """x (B, 80, 150) or (B, 1, 80, 150) -> cmd logits (B, 2)."""
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return m(x)[0]


def evaluate(m, loader, dev: torch.device):
    """Returns (bal_acc, wake_recall, nowake_fpr, confusion dict)."""
    m.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            pred = _forward(m, x).argmax(1)
            tp += int(((pred == 1) & (y == 1)).sum())
            fn += int(((pred == 0) & (y == 1)).sum())
            tn += int(((pred == 0) & (y == 0)).sum())
            fp += int(((pred == 1) & (y == 0)).sum())
    recall_wake = tp / max(tp + fn, 1)
    recall_nowake = tn / max(tn + fp, 1)
    bal = (recall_wake + recall_nowake) / 2.0
    return bal, recall_wake, fp / max(fp + tn, 1), \
        {"tp": tp, "fn": fn, "tn": tn, "fp": fp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v2cnn", "bcresnet"], default="bcresnet")
    ap.add_argument("--wake", default="data/wake_v2")
    ap.add_argument("--cmd", default="data/raw_v2")
    ap.add_argument("--noise", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/v3w")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU, 20-clip subsets, 2 epochs — pipeline check")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model={args.model}  device={dev}  wake={args.wake}  cmd={args.cmd}")
    feat()

    noise_bank = load_noise_bank(Path(args.noise)) if args.noise else []

    train_ds = WakeDataset("train", Path(args.wake), Path(args.cmd),
                           Path(args.noise) if args.noise else None,
                           augment=True, noise_bank=noise_bank)
    eval_ds = WakeDataset("eval", Path(args.wake), Path(args.cmd))
    if args.smoke:
        train_ds.items = train_ds.items[:20]
        eval_ds.items = eval_ds.items[:20]
        args.epochs = min(args.epochs, 2)
    n_wake = sum(1 for _, y in train_ds.items if y == 1)
    print(f"train n={len(train_ds)} (wake={n_wake} "
          f"no_wake={len(train_ds) - n_wake})  eval n={len(eval_ds)}")

    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=0)
    va = DataLoader(eval_ds, batch_size=args.batch, num_workers=0)

    m = build_model(args.model, 2).to(dev)
    print(f"params: {param_count(m):,}")
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    counts = torch.zeros(2)
    for _, y in train_ds.items:
        counts[y] += 1
    n = len(train_ds)
    w = (n / (2 * counts.clamp_min(1)))
    w = (w / w.mean()).to(dev)
    print("class weights:", {"no_wake": round(float(w[0]), 3),
                             "wake": round(float(w[1]), 3)})
    lossf = nn.CrossEntropyLoss(weight=w)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best_bal = 0.0
    for ep in range(1, args.epochs + 1):
        m.train()
        t0 = time.time()
        tot = cnt = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(_forward(m, x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            cnt += len(y)
        sched.step()
        bal, rec_w, fpr_n, cm = evaluate(m, va, dev)
        history.append({"epoch": ep, "train_loss": tot / max(cnt, 1),
                        "eval_bal_acc": bal, "eval_wake_recall": rec_w,
                        "eval_nowake_fpr": fpr_n, "confusion": cm,
                        "sec": time.time() - t0})
        tag_ = ""
        if bal > best_bal:
            best_bal = bal
            torch.save(m.state_dict(), out / f"vcm_wake_{args.model}_best.pt")
            tag_ = "  *best*"
        print(f"ep {ep:2d}  loss={tot/max(cnt,1):.4f}  bal={bal:.4f}  "
              f"wakeRec={rec_w:.4f}  nowakeFPR={fpr_n:.4f}  "
              f"({time.time() - t0:.1f}s){tag_}", flush=True)

    torch.save(m.state_dict(), out / f"vcm_wake_{args.model}.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    bal, rec_w, fpr_n, cm = evaluate(m, va, dev)
    print(f"WAKE-EVAL {args.model}: bal_acc={bal:.4f} wake_recall={rec_w:.4f} "
          f"nowake_fpr={fpr_n:.4f} cm={cm}  (best bal {best_bal:.4f}) "
          f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
