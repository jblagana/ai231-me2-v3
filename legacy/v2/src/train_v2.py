"""V2 train — two-head VCM (command + 7 slot heads), 3.0 s window.

Port of V1's validated A2 recipe (~/vcm/src/train_a2.py — v1g/a2 lineage:
right-aligned 3.0 s log-mel, mel-domain aug jitter 0.95-1.05 + gain +-6/20
+ MUSAN SNR 5-25, AdamW wd=1e-4 + cosine, class-weighted cmd CE + 0.5*slot
CE, best ckpt by EVAL-SYN cmd acc) with the V2 model zoo:

  --model v2cnn     primary (locked A2 arch, 109,890 p at 11c)
  --model bcresnet  experiment (arXiv 2106.04140, scale=2, 36,114 p)
  --merge-10        10-class label space (ask_weather+ask_time -> ask_question)

Data: {data}/{split}/{class}/*.raw (s16le @16 kHz) + manifest.jsonl
(speaker-disjoint; phrase in manifest drives slots.extract_slot).

Usage (HPC, one per GPU — Run A launches 4 of these in parallel):
  CUDA_VISIBLE_DEVICES=2 python src/train_v2.py --data ~/vcm/data/raw_v2 \
      --model v2cnn --epochs 30 --batch 32 --class-weights \
      --real-noise ~/vcm/data/noise16k --out runs/v2a/v2cnn_11c
  CUDA_VISIBLE_DEVICES=3 python src/train_v2.py ... --model v2cnn --merge-10 \
      --out runs/v2a/v2cnn_10c
  # v2e/v2e2 (10-02): 2-head, per-epoch eval + selection on VAL
  # (gated lexicographic, --cmd-gate 0.940 default)
  CUDA_VISIBLE_DEVICES=0 python src/train_v2.py --model bcresnet --merge-10 \
      --class-weights --epochs 60 --patience 8 --batch 32 --lr 1e-3 \
      --slot-w 0.5 \
      --data ~/vcm/data/raw_v2 --real-noise ~/vcm/data/noise16k \
      --save-all --out runs/v2e_bcresnet_10c
  # v2f (10-02): --cmd-only (command CE only) / --slot-only (Run D
  # recipe); the pair is judged by pipeline slot acc via
  # eval_v2.py --slot-only --pair-cmd <cmd ckpt> --split test
  (bcresnet same, --model bcresnet)
  python src/train_v2.py --smoke    # tiny pipeline check
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from commands import CLASSES, CLASSES_10, relabel_10  # noqa: E402
from slots import extract_slot, PARAMETRIC  # noqa: E402
from models import build_model, N_MELS, N_FRAMES, SR  # noqa: E402

CLASSES_11 = list(CLASSES)
SLOT_W = 0.5  # slot loss weight (V1 A2)
WINDOW_S = 3.0

RAW_CACHE = {}
BASE_FEAT_CACHE = {}
NOISE_BANK = []

_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=1024, hop_length=320, n_mels=N_MELS,
    f_min=50.0, f_max=8000.0,
)


def wav_to_logmel(wav: torch.Tensor, mel: torch.nn.Module = None) -> torch.Tensor:
    """(1, T) @16k -> (1, 80, 150) log-mel, RIGHT-aligned in the 3.0 s window.

    Right-aligned (pad LEFT) so the slot word — the LAST spoken word — always
    sits at the TAIL of the feature map, where the slot head reads. Clips
    longer than 3.0 s keep their LAST 3.0 s (never crop the tail).
    """
    if mel is None:
        mel = _MEL
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[-1] > SR * WINDOW_S:
        wav = wav[..., -int(SR * WINDOW_S):]
    m = mel(wav)
    logmel = torch.clamp(m, min=1e-5).log10()
    logmel = (logmel + 5.0) / 5.0
    if logmel.shape[-1] < N_FRAMES:
        pad = N_FRAMES - logmel.shape[-1]
        logmel = nn.functional.pad(logmel, (pad, 0))  # right-align: pad LEFT
    return logmel[..., :N_FRAMES]


def precompute_base_features(ds, dev: torch.device, mel: torch.nn.Module) -> int:
    """Every base log-mel once, on `dev` (GPU), into BASE_FEAT_CACHE (~1-2 min
    on an A100 for 47k clips; ~1-2 GB CPU RAM). Returns count cached."""
    t0 = time.time()
    for p, _, _, _ in ds.items:
        key = str(p)
        if key in BASE_FEAT_CACHE:
            continue
        feat = wav_to_logmel(load_wav(p).to(dev), mel).cpu()
        BASE_FEAT_CACHE[key] = feat
    print(f"base features: {len(BASE_FEAT_CACHE)} cached (GPU {dev}) in "
          f"{time.time() - t0:.0f}s", flush=True)
    return len(BASE_FEAT_CACHE)


def _resample_mel(x: torch.Tensor, T2: int) -> torch.Tensor:
    """Linearly resample a mel feature's frame axis to T2 frames."""
    T = x.shape[-1]
    if T2 <= 1:
        return x[..., :1]
    if T2 == T:
        return x
    src = torch.linspace(0.0, T - 1, T2)
    i0 = src.long().clamp(max=T - 1)
    i1 = (i0 + 1).clamp(max=T - 1)
    frac = (src - src.floor()).float().view(1, 1, -1)
    return x.index_select(2, i0) * (1 - frac) + x.index_select(2, i1) * frac


def load_wav(p: Path) -> torch.Tensor:
    raw_p = p.with_suffix(".raw")
    if raw_p.exists():
        arr = RAW_CACHE.get(str(raw_p))
        if arr is None:
            arr = (np.fromfile(str(raw_p), dtype=np.int16)
                   / 32768.0).astype(np.float32)
            RAW_CACHE[str(raw_p)] = arr
        return torch.from_numpy(arr).unsqueeze(0)
    raise FileNotFoundError(f"no .raw for {p} (run tools/decode_tts.py)")


def preload_raw(root: Path) -> int:
    t0 = time.time()
    n = 0
    for raw_p in sorted(root.rglob("*.raw")):
        key = str(raw_p)
        if key not in RAW_CACHE:
            RAW_CACHE[key] = (np.fromfile(str(raw_p), dtype=np.int16)
                              / 32768.0).astype(np.float32)
        n += 1
    print(f"preload: {n} raw files cached in RAM in {time.time() - t0:.1f}s",
          flush=True)
    return n


def load_noise_bank(noise_dir: Path, max_sec: float = 4.0) -> int:
    t0 = time.time()
    for f in sorted(Path(noise_dir).glob("*.npy")):
        w = torch.from_numpy(np.load(f)).unsqueeze(0)
        if w.shape[-1] > int(SR * max_sec):
            w = w[..., :int(SR * max_sec)]
        NOISE_BANK.append(wav_to_logmel(w).cpu())
    print(f"noise bank: {len(NOISE_BANK)} MUSAN clips cached in "
          f"{time.time() - t0:.0f}s", flush=True)
    return len(NOISE_BANK)


class VCMDatasetV2(Dataset):
    """(path, cls, phrase, slot_idx) items; slot_idx folded into the tuple so
    a single shuffle stays aligned with its clip."""

    def __init__(self, root: Path, split: str, augment: bool = False,
                 merge10: bool = False, jit=(0.95, 1.05)):
        self.root = root
        self.augment = augment
        self.merge10 = merge10
        self._jit = jit
        self._snr = (5.0, 25.0)
        self.items = []
        mf = root / "manifest.jsonl"
        if not mf.exists():
            raise FileNotFoundError(f"no manifest at {mf}")
        for line in mf.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e["split"] != split:
                continue
            p = root / e["file"]
            if not p.with_suffix(".raw").exists():
                continue
            slot_idx = -1
            if e["class"] in PARAMETRIC:
                sv, sidx = extract_slot(e["class"], e["phrase"])
                if sidx >= 0:
                    slot_idx = sidx
            self.items.append((p, e["class"], e["phrase"], slot_idx))

    def _base_feat(self, p: Path) -> torch.Tensor:
        key = str(p)
        feat = BASE_FEAT_CACHE.get(key)
        if feat is None:
            feat = wav_to_logmel(load_wav(p))
            BASE_FEAT_CACHE[key] = feat
        return feat

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        p, cls, phrase, slot_idx = self.items[i]
        x = self._base_feat(p)
        if self.augment:
            rate = random.uniform(*self._jit)
            T2 = int(round(x.shape[-1] / rate))
            x = _resample_mel(x, T2)
            if x.shape[-1] < N_FRAMES:
                x = nn.functional.pad(x, (N_FRAMES - x.shape[-1], 0))
            elif x.shape[-1] > N_FRAMES:
                x = x[..., -N_FRAMES:]
            x = x[..., :N_FRAMES]
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
        cls_label = relabel_10(cls) if self.merge10 else cls
        y_cmd = (CLASSES_10.index(cls_label) if self.merge10
                 else CLASSES_11.index(cls))
        return x, y_cmd, slot_idx


PARAMETRIC_VOCAB_IDX = None  # filled in main()/smoke() from slots.SLOT_VOCAB


def smoke() -> int:
    from slots import SLOT_VOCAB
    global PARAMETRIC_VOCAB_IDX
    PARAMETRIC_VOCAB_IDX = {k: {v: i for i, v in enumerate(vals)}
                            for k, vals in SLOT_VOCAB.items()}
    random.seed(0)
    for name in ("v2cnn", "bcresnet"):
        for merge in (False, True):
            classes = CLASSES_10 if merge else CLASSES_11
            m = build_model(name, len(classes))
            x, yc, ys = _fake_item()
            cmd, sf = m(torch.stack([x] * 8))
            assert cmd.shape == (8, len(classes)), cmd.shape
    print("train_v2 SMOKE: PASS (dataset loop needs HPC data; shapes here)")
    return 0


def _fake_item():
    """Synthetic (feature, cmd, slot) item for the shape smoke."""
    x = torch.zeros(1, N_MELS, N_FRAMES)
    return x, 0, -1


def main():
    global PARAMETRIC_VOCAB_IDX
    from slots import SLOT_VOCAB
    PARAMETRIC_VOCAB_IDX = {k: {v: i for i, v in enumerate(vals)}
                            for k, vals in SLOT_VOCAB.items()}
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw_v2")
    ap.add_argument("--model", choices=["v2cnn", "bcresnet"], default="v2cnn")
    ap.add_argument("--merge-10", action="store_true",
                    help="10-class label space (ask_weather+ask_time merged)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop: max epochs without improvement on "
                         "the val selection metric (0 = off; protocol "
                         "10-02: --epochs 60 --patience 8)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--slot-w", type=float, default=SLOT_W)
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--jit", default="0.95,1.05",
                    help="mel time-jitter 'lo,hi' (prosody diversity)")
    ap.add_argument("--slot-weights", default=None,
                    help="JSON {class: {slot: weight}} for slot oversampling")
    ap.add_argument("--init", default=None,
                    help="checkpoint to fine-tune from (same arch)")
    ap.add_argument("--resume", default=None,
                    help="continue from an ep ckpt (state_dict only): "
                         "loads weights, rebuilds fresh AdamW + "
                         "CosineAnnealingLR(T_max=--epochs) fast-forwarded "
                         "to the ckpt epoch, continues to --epochs; "
                         "prepends the out dir's history.json. Epoch is "
                         "parsed from the filename (..._epNNN.pt).")
    ap.add_argument("--save-all", action="store_true",
                    help="save a checkpoint every epoch (post-hoc selection)")
    ap.add_argument("--slot-only", action="store_true",
                    help="Run D: slot-only model — no command head/loss, "
                         "the slot CE is the entire loss (bcresnet)")
    ap.add_argument("--cmd-only", action="store_true",
                    help="v2f: command-only model — no slot heads/loss, "
                         "the command CE is the entire loss (bcresnet)")
    ap.add_argument("--eval-split", default="val",
                    help="split for per-epoch eval + selection "
                         "(protocol 10-02: 'val')")
    ap.add_argument("--cmd-gate", type=float, default=0.940,
                    help="2-head val selection gate (pre-registered 10-02): "
                         "max slot among epochs with cmd >= gate")
    ap.add_argument("--real-noise", default=None)
    ap.add_argument("--out", default="runs/v2a")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        return smoke()

    classes = CLASSES_10 if args.merge_10 else CLASSES_11
    tag = "10c" if args.merge_10 else "11c"
    print(f"model={args.model}  labelspace={tag}  data={args.data}")

    root = Path(args.data)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    preload_raw(root / "train")
    if args.real_noise:
        load_noise_bank(Path(args.real_noise))
    jit = tuple(float(v) for v in args.jit.split(","))
    train_ds = VCMDatasetV2(root, "train", augment=True,
                            merge10=args.merge_10, jit=jit)
    eval_ds = VCMDatasetV2(root, args.eval_split, augment=False,
                           merge10=args.merge_10)
    print(f"train n={len(train_ds)}  {args.eval_split} n={len(eval_ds)}")
    from torchaudio.transforms import MelSpectrogram
    gpu_mel = MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=320,
                             n_mels=N_MELS, f_min=50.0, f_max=8000.0).to(dev)
    precompute_base_features(train_ds, dev, gpu_mel)
    precompute_base_features(eval_ds, dev, gpu_mel)
    del gpu_mel

    if args.slot_weights:
        from torch.utils.data import WeightedRandomSampler
        sw = json.loads(Path(args.slot_weights).read_text())
        inv = {cls: {i: v for v, i in vals.items()}
               for cls, vals in PARAMETRIC_VOCAB_IDX.items()}
        w = []
        for _p, cls, _phrase, sidx in train_ds.items:
            if sidx >= 0:
                w.append(float(sw.get(cls, {})
                                .get(inv[cls][sidx], 1.0)))
            else:
                w.append(1.0)
        n_up = sum(1 for x in w if x > 1.0)
        print(f"slot oversampling: {n_up}/{len(w)} items upweighted "
              f"(max {max(w):.1f}, mean {sum(w) / len(w):.2f})")
        sampler = WeightedRandomSampler(w, num_samples=len(w),
                                        replacement=True)
        tr = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                        num_workers=0)
    else:
        tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=0)
    va = DataLoader(eval_ds, batch_size=args.batch, num_workers=0)

    if args.slot_only:
        from models import SlotOnlyBCResNet
        m = SlotOnlyBCResNet(len(classes)).to(dev)
        print("slot-only model (Run D): no command head — the slot CE is "
              "the only loss", flush=True)
    elif args.cmd_only:
        from models import CmdOnlyBCResNet
        m = CmdOnlyBCResNet(len(classes)).to(dev)
        print("cmd-only model (v2f): no slot heads — the command CE is "
              "the only loss", flush=True)
    else:
        m = build_model(args.model, len(classes)).to(dev)
        if args.init:
            m.load_state_dict(torch.load(args.init, map_location=dev))
            print(f"init: loaded {args.init}", flush=True)
    resume_ep = 0
    if args.resume:
        mmt = re.search(r"ep(\d+)\.pt$", Path(args.resume).name)
        if not mmt:
            raise SystemExit(
                f"--resume {args.resume}: cannot parse epoch from filename "
                "(need ..._epNNN.pt)")
        resume_ep = int(mmt.group(1))
        m.load_state_dict(torch.load(args.resume, map_location=dev))
        print(f"resume: loaded {args.resume} (ep {resume_ep}); fresh "
              f"AdamW + cosine(T_max={args.epochs}) fast-forwarded to "
              f"ep {resume_ep}", flush=True)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params:,}")
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if resume_ep:
        for _ in range(resume_ep):
            sched.step()
    if args.class_weights:
        counts = torch.zeros(len(classes))
        for it in train_ds.items:
            c = relabel_10(it[1]) if args.merge_10 else it[1]
            counts[classes.index(c)] += 1
        n = len(train_ds)
        w = n / (len(classes) * counts)
        w = (w / w.mean()).to(dev)
        print("class weights:", {classes[i]: round(float(w[i]), 3)
                                 for i in range(len(classes))})
        cmd_lossf = nn.CrossEntropyLoss(weight=w)
    else:
        cmd_lossf = nn.CrossEntropyLoss()
    slot_lossf = nn.CrossEntropyLoss()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    if resume_ep:
        prior = out / "history.json"
        if prior.exists():
            try:
                history = json.loads(prior.read_text())
                print(f"resume: prepended {len(history)} prior history "
                      f"epochs from {prior}", flush=True)
            except Exception as e:
                print(f"resume: WARNING could not load prior history "
                      f"({e}); starting fresh", flush=True)
    best_cmd = 0.0
    best_gated = -1.0  # 2-head: best val slot among gate-passing epochs
    best_any = -1.0    # 2-head: best val slot overall (guard fallback)
    best_seen = -1.0   # patience tracker: best val selection metric so far
    stalled = 0
    for ep in range(max(1, resume_ep + 1), args.epochs + 1):
        m.train()
        t0 = time.time()
        tot = cnt = 0
        for x, y_cmd, y_slot in tr:
            x, y_cmd, y_slot = x.to(dev), y_cmd.to(dev), y_slot.to(dev)
            opt.zero_grad()
            if args.slot_only:
                # Run D: no command head/loss — the slot CE is the entire
                # loss (sum of the per-class terms present in the batch,
                # same semantics as the 2-head recipe without the cmd term)
                slot_feat = m(x)
                loss = None
                n_slot = 0
                for cls in PARAMETRIC:
                    ci = classes.index(cls)
                    mask = (y_cmd == ci) & (y_slot >= 0)
                    if mask.any():
                        sl = m.slot_logits(slot_feat[mask], cls)
                        l = slot_lossf(sl, y_slot[mask])
                        loss = l if loss is None else loss + l
                        n_slot += int(mask.sum().item())
                if loss is None:
                    continue  # batch without a slot (rare at batch 32)
                loss.backward()
                opt.step()
                tot += loss.item() * n_slot
                cnt += n_slot
                continue
            if args.cmd_only:
                loss = cmd_lossf(m(x), y_cmd)
                loss.backward()
                opt.step()
                tot += loss.item() * len(y_cmd)
                cnt += len(y_cmd)
                continue
            cmd_logits, slot_feat = m(x)
            loss = cmd_lossf(cmd_logits, y_cmd)
            for cls in PARAMETRIC:
                ci = classes.index(cls)
                mask = (y_cmd == ci) & (y_slot >= 0)
                if mask.any():
                    sl = m.slot_logits(slot_feat[mask], cls)
                    loss = loss + args.slot_w * slot_lossf(sl, y_slot[mask])
            loss.backward()
            opt.step()
            tot += loss.item() * len(y_cmd)
            cnt += len(y_cmd)
        sched.step()
        m.eval()
        cmd_correct = cmd_total = 0
        slot_correct = slot_total = 0
        with torch.no_grad():
            for x, y_cmd, y_slot in va:
                x, y_cmd, y_slot = x.to(dev), y_cmd.to(dev), y_slot.to(dev)
                if args.slot_only:
                    slot_feat = m(x)
                    pred = y_cmd  # no command head — gate on true class
                elif args.cmd_only:
                    cmd_logits = m(x)
                    pred = cmd_logits.argmax(1)
                    cmd_correct += (pred == y_cmd).sum().item()
                    cmd_total += len(y_cmd)
                    continue
                else:
                    cmd_logits, slot_feat = m(x)
                    pred = cmd_logits.argmax(1)
                    cmd_correct += (pred == y_cmd).sum().item()
                    cmd_total += len(y_cmd)
                for cls in PARAMETRIC:
                    ci = classes.index(cls)
                    mask = (y_cmd == ci) & (y_slot >= 0)
                    if mask.any():
                        sl = m.slot_logits(slot_feat[mask], cls)
                        sp = sl.argmax(1)
                        slot_correct += (sp == y_slot[mask]).sum().item()
                        slot_total += int(mask.sum().item())
        cmd_acc = cmd_correct / max(cmd_total, 1)
        slot_acc = slot_correct / max(slot_total, 1)
        history.append({"epoch": ep, "train_loss": tot / max(cnt, 1),
                        "eval_cmd_acc": cmd_acc, "eval_slot_acc": slot_acc,
                        "sec": time.time() - t0})
        tag_ = ""
        if args.slot_only or args.cmd_only:
            sel = slot_acc if args.slot_only else cmd_acc
            if sel > best_cmd:
                best_cmd = sel
                torch.save(m.state_dict(),
                           out / f"vcm_{args.model}_{tag}_best.pt")
                tag_ = "  *best*"
        else:
            # 2-head (v2e/v2e2): pre-registered lexicographic selection
            # (10-02) — max slot among epochs at the val cmd gate;
            # _bestslot = ungated fallback if no epoch passes the gate.
            if cmd_acc >= args.cmd_gate and slot_acc > best_gated:
                best_gated = slot_acc
                torch.save(m.state_dict(),
                           out / f"vcm_{args.model}_{tag}_best.pt")
                tag_ = "  *best*"
            if slot_acc > best_any:
                best_any = slot_acc
                torch.save(m.state_dict(),
                           out / f"vcm_{args.model}_{tag}_bestslot.pt")
                tag_ = tag_ or "  *bestslot*"
        if args.save_all:
            torch.save(m.state_dict(),
                       out / f"vcm_{args.model}_{tag}_ep{ep:02d}.pt")
        cmd_s = "n/a" if args.slot_only else f"{cmd_acc:.3f}"
        print(f"ep {ep:2d}  loss={tot/max(cnt,1):.4f}  "
              f"cmd={cmd_s}  slot={slot_acc:.3f}  "
              f"({time.time() - t0:.1f}s){tag_}", flush=True)
        # pre-registered patience stop (protocol 10-02): halt when the
        # val selection metric has not improved for --patience epochs
        sel_metric = cmd_acc if args.cmd_only else slot_acc
        if sel_metric > best_seen:
            best_seen = sel_metric
            stalled = 0
        else:
            stalled += 1
        if args.patience and stalled >= args.patience:
            print(f"early stop @ ep {ep:2d} — no {args.eval_split} "
                  f"improvement for {args.patience} epochs", flush=True)
            break

    torch.save(m.state_dict(), out / f"vcm_{args.model}_{tag}.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    if args.slot_only or args.cmd_only:
        print(f"saved -> {out}  (best {args.eval_split} "
              f"{'slot' if args.slot_only else 'cmd'} acc {best_cmd:.3f})")
    else:
        print(f"saved -> {out}  (gated best {args.eval_split} slot "
              f"{best_gated:.3f} @ cmd>={args.cmd_gate}; max slot "
              f"{best_any:.3f}"
              f"{' — GUARD FAILED (no epoch at gate)' if best_gated < 0 else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
