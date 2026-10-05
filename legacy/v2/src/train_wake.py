"""V2 wake-gate training — 2-class (0 = no_wake, 1 = wake "hey boots").

Same locked feature and recipe as the VCM (3.0 s RIGHT-aligned log-mel,
mel-domain jitter 0.95-1.05 + MUSAN SNR 5-25 aug, AdamW 1e-4 wd + cosine,
class-weighted CE). Only the command head is trained/used.

Data (all speaker-disjoint 20/20, same 40 voices as the VCM):
  wake (1):      {wake}/{split}/*.raw          (tools/make_wake_tts.py + decode)
  no_wake (0):   {cmd}/{split}/**/*.raw        (the VCM command clips)
                 + MUSAN noise .npy -> .raw (train only)
Eval is speaker-disjoint: eval-voice wake clips + eval-split commands.
Best checkpoint by BALANCED ACCURACY on eval.

Usage (HPC, vcm env, one per GPU):
  CUDA_VISIBLE_DEVICES=2 python src/train_wake.py --model v2cnn \
      --wake ~/vcm/data/wake_v2 --cmd ~/vcm/data/raw_v2 \
      --real-noise ~/vcm/data/noise16k --out runs/v2w/v2cnn
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

sys.path.insert(0, str(Path(__file__).parent))
from models import build_model, N_MELS, N_FRAMES, SR  # noqa: E402
from train_v2 import (NOISE_BANK, _resample_mel, load_noise_bank,  # noqa: E402
                      load_wav, preload_raw, wav_to_logmel)

BASE_FEAT_CACHE = {}


def materialize_noise(noise_dir: Path) -> Path:
    """MUSAN .npy (float32 16 kHz) -> s16le .raw siblings so load_wav() and
    the GPU precompute treat them like any other clip. Idempotent."""
    out = noise_dir.parent / (noise_dir.name + "_raw")
    out.mkdir(exist_ok=True)
    n = 0
    for f in sorted(Path(noise_dir).glob("*.npy")):
        raw_p = out / (f.stem + ".raw")
        if raw_p.exists() and raw_p.stat().st_size > 0:
            continue
        w = np.load(f)
        if w.shape[-1] > int(SR * 3.0):
            w = w[..., :int(SR * 3.0)]
        (w.astype(np.float32) * 32767.0).astype("<i2").tofile(raw_p)
        n += 1
    print(f"noise raw: {n} converted under {out}", flush=True)
    return out


class WakeDataset(Dataset):
    """(path, y) items, y in {0: no_wake, 1: wake}. Path carries a suffix so
    train_v2.load_wav finds its .raw sibling (noise items use a fake '.npy'
    name with a real .raw sibling from materialize_noise)."""

    def __init__(self, split: str, wake_root: Path, cmd_root: Path,
                 noise_raw_root: Path = None, augment: bool = False):
        self.augment = augment
        self._jit = (0.95, 1.05)
        self._snr = (5.0, 25.0)
        self.items = []
        for p in sorted((Path(wake_root) / split).glob("*.mp3")):
            if p.with_suffix(".raw").exists():
                self.items.append((p, 1))
        for p in sorted((Path(cmd_root) / split).rglob("*.mp3")):
            if p.with_suffix(".raw").exists():
                self.items.append((p, 0))
        if split == "train" and noise_raw_root is not None:
            for raw_p in sorted(Path(noise_raw_root).glob("*.raw")):
                self.items.append((raw_p.with_suffix(".npy"), 0))
        random.shuffle(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def _base_feat(self, p: Path) -> torch.Tensor:
        key = str(p.with_suffix(".raw"))
        feat = BASE_FEAT_CACHE.get(key)
        if feat is None:
            feat = wav_to_logmel(load_wav(p))
            BASE_FEAT_CACHE[key] = feat
        return feat

    def __getitem__(self, i):
        p, y = self.items[i]
        x = self._base_feat(p)
        if self.augment:
            rate = random.uniform(*self._jit)
            T2 = int(round(x.shape[-1] / rate))
            x = _resample_mel(x, T2)
            if x.shape[-1] < N_FRAMES:
                x = nn.functional.pad(x, (N_FRAMES - x.shape[-1], 0))
            elif x.shape[-1] > N_FRAMES:
                x = x[..., -N_FRAMES:]
            x = x + random.uniform(-6.0, 6.0) / 20.0
            snr_db = random.uniform(*self._snr)
            if NOISE_BANK:
                nz = random.choice(NOISE_BANK)
                W = N_FRAMES
                if nz.shape[-1] >= W:
                    start = random.randint(0, nz.shape[-1] - W)
                    nz = nz[..., start:start + W]
                else:
                    reps = -(-W // nz.shape[-1])
                    nz = nz.repeat(1, 1, reps)[..., :W]
            else:
                nz = torch.randn(1, N_MELS, N_FRAMES)
            sig_p = x.pow(2).mean().clamp_min(1e-8)
            nz_p = nz.pow(2).mean().clamp_min(1e-8)
            scale = torch.sqrt(sig_p / (nz_p * 10 ** (snr_db / 10.0)))
            x = x + nz * scale
        return x, y


def precompute(items, dev: torch.device, mel: torch.nn.Module) -> int:
    t0 = time.time()
    for p, _ in items:
        key = str(p.with_suffix(".raw"))
        if key in BASE_FEAT_CACHE:
            continue
        feat = wav_to_logmel(load_wav(p).to(dev), mel).cpu()
        BASE_FEAT_CACHE[key] = feat
    print(f"base features: {len(BASE_FEAT_CACHE)} cached (GPU {dev}) in "
          f"{time.time() - t0:.0f}s", flush=True)
    return len(BASE_FEAT_CACHE)


def evaluate(m, loader, dev: torch.device):
    """Returns (bal_acc, wake_recall, nowake_fpr, confusion dict)."""
    m.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            pred = m(x)[0].argmax(1)
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
    ap.add_argument("--model", choices=["v2cnn", "bcresnet"], default="v2cnn")
    ap.add_argument("--wake", default="data/wake_v2")
    ap.add_argument("--cmd", default="data/raw_v2")
    ap.add_argument("--real-noise", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/v2w")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model={args.model}  device={dev}  wake={args.wake}  cmd={args.cmd}")

    noise_raw = materialize_noise(Path(args.real_noise)) if args.real_noise \
        else None

    import train_v2
    preload_raw(Path(args.wake) / "train")
    preload_raw(Path(args.cmd) / "train")
    if noise_raw:
        preload_raw(noise_raw)
    if args.real_noise:
        load_noise_bank(Path(args.real_noise))

    train_ds = WakeDataset("train", Path(args.wake), Path(args.cmd),
                           noise_raw, augment=True)
    eval_ds = WakeDataset("eval", Path(args.wake), Path(args.cmd), None)
    n_wake = sum(1 for _, y in train_ds.items if y == 1)
    print(f"train n={len(train_ds)} (wake={n_wake} "
          f"no_wake={len(train_ds) - n_wake})  eval n={len(eval_ds)}")

    from torchaudio.transforms import MelSpectrogram
    gpu_mel = MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=320,
                             n_mels=N_MELS, f_min=50.0, f_max=8000.0).to(dev)
    precompute(train_ds.items, dev, gpu_mel)
    precompute(eval_ds.items, dev, gpu_mel)
    del gpu_mel

    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=0)
    va = DataLoader(eval_ds, batch_size=args.batch, num_workers=0)

    m = build_model(args.model, 2).to(dev)
    print(f"params: {sum(p.numel() for p in m.parameters()):,}")
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
            cmd_logits, _ = m(x)
            loss = lossf(cmd_logits, y)
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
