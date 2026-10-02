"""V3 training + evaluation harness (run 1).

FROZEN protocol (BENCHMARK.md; re-freeze 2026-10-02, JFS rev 6947f130):
  model      bcresnet scale=2 (PRIMARY) / v2cnn (fallback)
  optimizer  AdamW lr=1e-3, weight_decay=1e-4, cosine over total steps
  batch      32
  loss       L = L_cmd + 1.0 * L_slot
             L_cmd : class-weighted CE, frozen weights w = n_max/n_c
             L_slot: plain CE over the batch's slotted clips (0 if none)
  selection  max slot accuracy (marginal over slotted TEST clips) among
             epochs with TEST command accuracy >= 0.990; if none qualify,
             best command accuracy + "GUARD FAILED" logged
  holdout    ONE-SHOT: auto-evaluated once on the selected model; a flag
             file in the run dir refuses a second evaluation
  numerals   REPORT ONLY, evaluated once on the selected model (t1 = file-end
             fallback — no numerals VAD audit; documented in DECISIONS)

TEST metrics per epoch: command acc (overall + per-class recall/precision),
per-head + overall slot acc (marginal; conditional-on-cmd-correct reported
alongside), OOS false-fire rate, 6x3x3 slot confusions, SNR 20/10/5 dB
command acc (seeded deterministic waveform MUSAN mix, cached in the run
dir), >3.0 s vs <=3.0 s diagnostic, OOS softmax stats (basis for the
pre-registered confidence threshold).

HPC:
  ~/.conda/envs/vcm/bin/python tools/train.py --run v3r1
  ~/.conda/envs/vcm/bin/python tools/train.py --smoke     # 100-500 clips
"""
import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from commands import CLASSES                        # noqa: E402
from dataset import OOS_IDX, V3Dataset, class_weight_vector  # noqa: E402
from features import (IN_SAMPLES, Features, decode_wav_bytes,  # noqa: E402
                      mix_noise)
from models import build_model, param_count          # noqa: E402
from slots import PARAMETRIC, SLOT_COUNTS            # noqa: E402

N_CLASSES = len(CLASSES)                             # 20
HEAD_IDX = {name: i for i, name in enumerate(PARAMETRIC)}
CMD_GATE = 0.990
FEATURE_POLICY_VERSION = (
    "policy-v2: 3.0s window right-aligned to min(d, t1+1.0s); 16 kHz; 80-mel "
    "log10; 150 frames; VAD 10ms rms>max(1e-3,0.10*peak) (tools/vad_audit.py); "
    "no-voiced -> file-end fallback; dataset = JFS rev 6947f130 default; "
    "slot read region 2.0s (10-02 amend)")


def git_hash(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short",
                               "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TeeLog:
    """print -> stdout + runs/<run>/train.log."""

    def __init__(self, path: Path):
        self.fh = open(path, "a", buffering=1)

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.fh.write(s + "\n")

    def close(self):
        self.fh.close()


class IndexSubset:
    """Deterministic index subset of a V3Dataset (smoke mode)."""

    def __init__(self, ds, idx):
        self.ds, self.idx = ds, list(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.ds[self.idx[i]]


def collate(batch):
    xs, cmds, heads, slots, metas = [], [], [], [], []
    for x, cmd, slot_cls, slot_idx, meta in batch:
        assert x.shape == (1, 80, 150) and x.dtype == np.float32
        xs.append(x)
        cmds.append(cmd)
        heads.append(HEAD_IDX[slot_cls] if slot_cls is not None else -1)
        slots.append(slot_idx)
        metas.append(meta)
    return {"x": torch.from_numpy(np.stack(xs)),
            "cmd": torch.tensor(cmds, dtype=torch.long),
            "head": torch.tensor(heads, dtype=torch.long),
            "slot": torch.tensor(slots, dtype=torch.long),
            "meta": metas}


def slot_loss(model, slot_feat, head, slot, assert_shapes=False):
    """Plain CE over the batch's slotted clips; 0.0 when none (frozen)."""
    n = int((head >= 0).sum())
    if n == 0:
        return slot_feat.new_zeros(())
    total = None
    for h, name in enumerate(PARAMETRIC):
        msk = head == h
        if not bool(msk.any()):
            continue
        logits = model.slot_logits(slot_feat[msk], name)
        if assert_shapes:
            assert logits.shape == (int(msk.sum()), SLOT_COUNTS[name])
        ce = F.cross_entropy(logits, slot[msk], reduction="sum")
        total = ce if total is None else total + ce
    return total / n


def precompute_clean(ds, features, path: Path, log):
    """Deterministic clean features + slot supervision for an eval split.

    Cached in the run dir. Returns (x, y, oos, head, slot, dur) tensors.
    Large splits (numerals) are computed with a fork Pool.
    """
    keys = ("x", "y", "oos", "head", "slot", "dur")
    if path.exists():
        z = np.load(path)
        log(f"  loaded cached {path.name} ({z['x'].shape[0]} clips)")
        return tuple(torch.from_numpy(z[k]) for k in keys)
    n = len(ds)
    x = np.zeros((n, 1, 80, 150), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    oos = np.zeros(n, dtype=np.int64)
    head = np.full(n, -1, dtype=np.int64)
    slot = np.full(n, -1, dtype=np.int64)
    dur = np.zeros(n, dtype=np.float64)
    t0 = time.time()

    def work(i):
        xi, yi, sc, si, meta = ds[i]
        return (i, xi, yi, HEAD_IDX[sc] if sc is not None else -1, si,
                meta["duration_s"], int(meta["oos"]))

    if n > 5000:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(16) as pool:
            got = pool.imap_unordered(work, range(n))
    else:
        got = map(work, range(n))
    for (i, xi, yi, hi, si, di, oi) in got:
        x[i] = xi
        y[i] = yi
        head[i] = hi
        slot[i] = si
        dur[i] = di
        oos[i] = oi
    np.savez_compressed(path, x=x, y=y, oos=oos, head=head, slot=slot,
                        dur=dur)
    log(f"  precomputed {path.name} in {time.time() - t0:.0f}s")
    return tuple(torch.from_numpy(a) for a in (x, y, oos, head, slot, dur))


def precompute_robust(ds, features, snrs, run_dir: Path, seed: int, log):
    """Seed-deterministic waveform MUSAN mix per SNR (cached).

    Per (snr): rng = default_rng(seed + snr); each clip gets one noise clip
    + one start offset from that rng — identical across epochs and runs with
    the same seed. Requires the filtered >=3.0 s bank (data/noise16k_3s).
    """
    out = {}
    bank = features.noise
    if not bank:
        raise ValueError("robust precompute needs the MUSAN npy bank")
    for snr in snrs:
        p = run_dir / f"test_snr{snr}.npz"
        keys = ("x", "y", "oos", "head", "slot", "dur")
        if p.exists():
            z = np.load(p)
            log(f"  loaded cached {p.name}")
            out[snr] = tuple(torch.from_numpy(z[k]) for k in keys)
            continue
        n = len(ds)
        rng = np.random.default_rng(seed + int(snr))
        x = np.zeros((n, 1, 80, 150), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        oos = np.zeros(n, dtype=np.int64)
        head = np.full(n, -1, dtype=np.int64)
        slot = np.full(n, -1, dtype=np.int64)
        dur = np.zeros(n, dtype=np.float64)
        t0 = time.time()
        for i in range(n):
            kind, base, cmd, _, _, meta = ds.samp[i]
            assert kind == "real"
            b = ds.bank["bytes"][ds.bank["idx"][base]].as_py()
            wav = decode_wav_bytes(b["bytes"] if isinstance(b, dict) else b)
            xi = int(rng.integers(len(bank)))
            a = bank[xi]
            s = int(rng.integers(len(a) - IN_SAMPLES + 1))
            noisy = mix_noise(wav, a[s:s + IN_SAMPLES], float(snr))
            x[i] = features.clean(noisy, t1=meta["t1"])
            y[i] = cmd
            oos[i] = int(meta["oos"])
            dur[i] = meta["duration_s"]
        np.savez_compressed(p, x=x, y=y, oos=oos, head=head, slot=slot,
                            dur=dur)
        log(f"  precomputed {p.name} in {time.time() - t0:.0f}s")
        out[snr] = tuple(torch.from_numpy(a) for a in
                         (x, y, oos, head, slot, dur))
    return out



@torch.no_grad()
def evaluate_feats(model, x, y, head, slot, metas, device, chunk=512):
    """Full TEST metric set from precomputed features.

    x (N,1,80,150) float32 cpu; y (N,) cmd labels; head (N,) parametric-head
    index or -1; slot (N,) slot index (-1 when head == -1); metas list of
    sample meta dicts (for the duration diagnostic).
    """
    model.eval()
    N = x.shape[0]
    logits = torch.empty(N, N_CLASSES, dtype=torch.float32)
    sf_list, sf_h, sf_i = [], [], []
    for s in range(0, N, chunk):
        cl, sf = model(x[s:s + chunk].to(device))
        logits[s:s + chunk] = cl.cpu()
        m = head[s:s + chunk] >= 0
        if bool(m.any()):
            sf_list.append(sf[m].cpu())
            sf_h.append(head[s:s + chunk][m])
            sf_i.append(torch.nonzero(m).flatten() + s)

    pred = logits.argmax(1)
    m = {}
    m["n"] = int(N)
    m["cmd_acc"] = float((pred == y).float().mean())
    per = {}
    for c, name in enumerate(CLASSES):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        per[name] = {"n": int((y == c).sum()),
                     "recall": tp / (tp + fn) if tp + fn else None,
                     "precision": tp / (tp + fp) if tp + fp else None}
    m["cmd_per_class"] = per

    oos_m = y == OOS_IDX
    m["oos_n"] = int(oos_m.sum())
    if m["oos_n"]:
        m["oos_reject"] = float((pred[oos_m] == OOS_IDX).float().mean())
        m["oos_false_fire"] = 1.0 - m["oos_reject"]
    else:
        m["oos_reject"] = m["oos_false_fire"] = None

    # slot supervision (marginal over slotted clips; conditional alongside)
    slotted = head >= 0
    slot_pred = torch.full_like(slot, -1)
    if int(slotted.sum()) and sf_list:
        sf_all = torch.cat(sf_list)
        h_all = torch.cat(sf_h)
        i_all = torch.cat(sf_i)
        for h, name in enumerate(PARAMETRIC):
            hh = h_all == h
            if bool(hh.any()):
                lh = model.slot_logits(sf_all[hh].to(device), name).cpu()
                slot_pred[i_all[hh]] = lh.argmax(1)
    s_m = slotted
    m["slot_n"] = int(s_m.sum())
    if m["slot_n"]:
        m["slot_acc"] = float((slot_pred[s_m] == slot[s_m]).float().mean())
        cond = s_m & (pred == y)
        m["slot_acc_cond"] = float((slot_pred[cond] == slot[cond]).float().mean())
        per_h, conf = {}, {}
        for h, name in enumerate(PARAMETRIC):
            hh = s_m & (head == h)
            per_h[name] = {"n": int(hh.sum()),
                           "acc": float((slot_pred[hh] == slot[hh]).float().mean())
                           if int(hh.sum()) else None}
            cm = np.zeros((3, 3), dtype=int)
            tt = slot[hh].numpy()
            pp = slot_pred[hh].numpy()
            for i in range(len(tt)):
                cm[tt[i], pp[i]] += 1
            conf[name] = cm.tolist()
        m["slot_per_head"] = per_h
        m["slot_conf"] = conf
    else:
        m["slot_n"] = 0
        m["slot_acc"] = m["slot_acc_cond"] = None
        m["slot_per_head"] = {n: {"n": 0, "acc": None} for n in PARAMETRIC}
        m["slot_conf"] = {n: [[0] * 3 for _ in range(3)] for n in PARAMETRIC}

    if metas:
        durs = np.array([meta["duration_s"] for meta in metas])
        ok = (pred == y).numpy()
        for key, long_ in (("dur_long", True), ("dur_short", False)):
            dm = durs > 3.0 if long_ else durs <= 3.0
            m[f"{key}_n"] = int(dm.sum())
            m[f"{key}_acc"] = float(ok[dm].mean()) if int(dm.sum()) else None

    p = torch.softmax(logits, dim=1)
    p_cmd = p.clone()
    p_cmd[:, OOS_IDX] = -1.0
    p_cmd_max = p_cmd.max(dim=1).values
    if m["oos_n"]:
        m["oos_softmax"] = {
            "p_oos_mean": float(p[:, OOS_IDX][oos_m].mean()),
            "p_cmd_max_mean": float(p_cmd_max[oos_m].mean()),
            "p_oos_ge_cmd_frac": float((p[:, OOS_IDX][oos_m] >=
                                        p_cmd_max[oos_m]).float().mean()),
        }
    else:
        m["oos_softmax"] = None
    return m



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["bcresnet", "v2cnn"], default="bcresnet")
    ap.add_argument("--run", default=None, help="run name (runs/<name>/)")
    ap.add_argument("--epochs", type=int, default=100,
                    help="frozen protocol: 100 (see BENCHMARK.md)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--device", default="auto",
                    help="auto | cuda | cuda:0 | cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--buffers", default="data/mel_buffers")
    ap.add_argument("--noise", default="data/noise16k_3s",
                    help="MUSAN npy bank (>= 3.0 s clips)")
    ap.add_argument("--arrow-dir",
                    default="/data/ai231/airimonda___ai231-me2-voice-commands/"
                            "default/0.0.0")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--vad-dir", default="data/vad_audit")
    ap.add_argument("--gen-oos", default="auto",
                    help="0 | auto (= real train OOS count) | int")
    ap.add_argument("--snr", default="20 10 5")
    ap.add_argument("--skip-robust", action="store_true")
    ap.add_argument("--skip-numerals", action="store_true")
    ap.add_argument("--force-holdout", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=200,
                    help="smoke: train subset size")
    ap.add_argument("--limit-eval", type=int, default=100,
                    help="smoke: test subset size")
    return ap.parse_args()


def smoke_indices(ds, limit):
    """Deterministic subset that still covers OOS + slot-supervised clips."""
    oos_i = [i for i in range(len(ds))
             if ds.samp[i][0] == "real" and ds.samp[i][5]["oos"]][:60]
    slot_i = [i for i in range(len(ds)) if ds.samp[i][4] >= 0][:60]
    seen = set(oos_i) | set(slot_i)
    rest = [i for i in range(len(ds)) if i not in seen]
    idx = oos_i + slot_i + rest
    return sorted(set(idx[:max(int(limit), len(idx) and (len(oos_i) + len(slot_i)))]))


def main():
    args = parse_args()
    set_seeds(args.seed)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    run_name = args.run or (time.strftime("smoke_%Y%m%d_%H%M%S") if args.smoke
                            else "v3r1")
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = TeeLog(run_dir / "train.log")
    gh = git_hash(ROOT)
    log(f"=== V3 train: {args.arch} scale=2 | run={run_name} | git={gh} | "
        f"device={device} | seed={args.seed} | smoke={args.smoke} ===")
    log(f"feature policy: {FEATURE_POLICY_VERSION}")

    # ---------------- data (frozen: JFS rev 6947f130 default) -----------
    f_train = Features(args.buffers, args.noise,
                       rng=np.random.default_rng(args.seed + 1))
    f_eval = Features(args.buffers, None,
                      rng=np.random.default_rng(args.seed + 2))
    man = []
    with open(Path(args.manifest_dir) / "manifest_train.csv") as fh:
        man = list(csv.DictReader(fh))
    n_real_oos = sum(1 for r in man if r["command"] == "OUT_OF_SCOPE")
    if args.gen_oos == "auto":
        gen_oos = n_real_oos
    else:
        gen_oos = int(args.gen_oos)
    ds_kw = dict(arrow_dir=args.arrow_dir, manifest_dir=args.manifest_dir,
                 vad_dir=args.vad_dir)
    train_ds = V3Dataset("train", f_train, augment=True, n_gen_oos=gen_oos,
                         **ds_kw)
    test_ds = V3Dataset("test", f_eval, augment=False, **ds_kw)
    log(train_ds.summary())
    log(test_ds.summary())
    w = class_weight_vector(Path(args.manifest_dir) / "manifest_train.csv")
    log(f"class weights (frozen w=nmax/n_c): OOS={w[OOS_IDX]:.3f} "
        f"min={w.min():.3f} max={w.max():.3f}")
    log(f"generated OOS lanes: {gen_oos} "
        f"({gen_oos // 2} silence + {gen_oos - gen_oos // 2} noise-only)")

    train_eff, test_eff = train_ds, test_ds
    if args.smoke:
        train_eff = IndexSubset(train_ds, smoke_indices(train_ds, args.limit))
        test_eff = IndexSubset(test_ds, smoke_indices(test_ds, args.limit_eval))
        log(f"smoke: train {len(train_eff)} | test {len(test_eff)}")

    # ---------------- model / optimizer --------------------------------
    model = build_model(args.arch, N_CLASSES, scale=2).to(device)
    log(f"model: {args.arch} scale=2  params={param_count(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    total_steps = max(1, math.ceil(len(train_eff) / args.batch)) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=args.lr * 0.01)
    w_t = torch.from_numpy(w).to(device)
    g_train = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        train_eff, batch_size=args.batch, shuffle=True, collate_fn=collate,
        num_workers=(2 if args.smoke else args.workers), generator=g_train,
        drop_last=False)

    # ---------------- eval feature caches ------------------------------
    x_t, y_t, oos_t, head_t, slot_t, dur_t = precompute_clean(
        test_eff, f_eval, run_dir / "test_clean.npz", log)
    robust = {}
    if not args.skip_robust and not args.smoke:
        snrs = [int(s) for s in args.snr.split()]
        robust = precompute_robust(test_ds, f_train, snrs, run_dir,
                                   args.seed, log)
    log(f"ready: train {len(train_eff)} | test {len(test_eff)} | "
        f"snr {list(robust)} | epochs {args.epochs} | batch {args.batch}")


    # ---------------- smoke unit checks --------------------------------
    if args.smoke:
        assert int(oos_t.sum()) > 0, "smoke test subset has no OOS clips"
        assert int(train_ds.counts[OOS_IDX]) > 0
        model.train()
        xf = torch.zeros(8, 1, 80, 150, device=device)
        cl, sf = model(xf)
        h_none = torch.full((8,), -1, dtype=torch.long, device=device)
        s_dummy = torch.zeros(8, dtype=torch.long, device=device)
        assert float(slot_loss(model, sf, h_none, s_dummy)) == 0.0, \
            "slot-loss masking"
        h_mix = torch.tensor([0, 0, 1, -1, 2, 3, 4, 5], device=device)
        s_mix = torch.tensor([0, 1, 2, 0, 0, 1, 2, 0], device=device)
        ls = slot_loss(model, sf, h_mix, s_mix, assert_shapes=True)
        assert torch.isfinite(ls) and float(ls.detach()) > 0.0
        for name in PARAMETRIC:
            assert model.slot_logits(sf, name).shape == (8, SLOT_COUNTS[name])
        log("smoke checks: slot-loss masking OK | per-head (B,3) OK | "
            "cmd (B,20) OK (asserted in loop)")

    # ---------------- train loop ----------------------------------------
    metas_t = [{"duration_s": float(d), "oos": bool(o)}
               for d, o in zip(dur_t.numpy(), oos_t.numpy())]
    best = None       # eligible (cmd acc >= gate), max slot acc
    best_cmd = None   # overall best cmd acc (fallback)
    epoch_rows = []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        s_l = s_c = s_s = 0.0
        n_b = 0
        for b in loader:
            x = b["x"].to(device)
            opt.zero_grad(set_to_none=True)
            cmd_logits, slot_feat = model(x)
            if args.smoke:
                assert cmd_logits.shape == (x.shape[0], N_CLASSES)
                assert torch.isfinite(cmd_logits).all()
            l_cmd = F.cross_entropy(cmd_logits, b["cmd"].to(device),
                                    weight=w_t)
            l_slot = slot_loss(model, slot_feat, b["head"].to(device),
                               b["slot"].to(device),
                               assert_shapes=args.smoke)
            loss = l_cmd + 1.0 * l_slot
            if args.smoke:
                assert torch.isfinite(loss), "NaN loss"
            loss.backward()
            opt.step()
            sched.step()
            s_l += loss.item()
            s_c += l_cmd.item()
            s_s += l_slot.item()
            n_b += 1

        m = evaluate_feats(model, x_t, y_t, head_t, slot_t, metas_t, device)
        for snr in sorted(robust):
            rx, ry, ro, rh, rs, rd = robust[snr]
            rm = evaluate_feats(
                model, rx, ry, rh, rs,
                [{"duration_s": float(d)} for d in rd.numpy()], device)
            m[f"snr{snr}_cmd_acc"] = rm["cmd_acc"]
            m[f"snr{snr}_oos_ff"] = rm["oos_false_fire"]

        entry = {"epoch": epoch,
                 "train": {"loss": s_l / n_b, "cmd": s_c / n_b,
                           "slot": s_s / n_b, "sec": round(time.time() - t0, 1)},
                 "metrics": m,
                 "eligible": bool(m["cmd_acc"] >= CMD_GATE)}
        epoch_rows.append(entry)

        is_best = False
        if entry["eligible"]:
            if best is None or (m["slot_acc"] is not None and (
                    best["metrics"]["slot_acc"] is None
                    or m["slot_acc"] > best["metrics"]["slot_acc"]
                    or (m["slot_acc"] == best["metrics"]["slot_acc"]
                        and m["cmd_acc"] > best["metrics"]["cmd_acc"]))):
                best = entry
                is_best = True
        if best_cmd is None or m["cmd_acc"] > best_cmd["metrics"]["cmd_acc"]:
            best_cmd = entry

        ckpt = {"model_state": model.state_dict(), "arch": args.arch,
                "scale": 2, "config": vars(args),
                "class_weights": [float(v) for v in w],
                "metrics": m, "epoch": epoch, "git_hash": gh,
                "feature_policy": FEATURE_POLICY_VERSION}
        torch.save(ckpt, run_dir / f"checkpoint_epoch{epoch}.pt")
        if is_best:
            torch.save(ckpt, run_dir / "checkpoint_best.pt")

        def _f(v):
            return "  -  " if v is None else f"{v:.4f}"
        snr_s = "  ".join(f"snr{s}:{m[f'snr{s}_cmd_acc']:.4f}"
                          for s in sorted(robust))
        log(f"epoch {epoch:2d}/{args.epochs}  {entry['train']['sec']}s  "
            f"L {entry['train']['loss']:.4f} "
            f"(cmd {entry['train']['cmd']:.4f} + slot "
            f"{entry['train']['slot']:.4f})  TEST cmd {m['cmd_acc']:.4f}  "
            f"slot {_f(m['slot_acc'])}  oos-ff {m['oos_false_fire']:.3f}  "
            f"{snr_s}")

    # ---------------- selection (FROZEN rule) ---------------------------
    guard = "OK" if best is not None else "GUARD FAILED"
    sel = best if best is not None else best_cmd
    if guard == "GUARD FAILED":
        log("GUARD FAILED: no TEST epoch reached cmd acc >= "
            f"{CMD_GATE}; falling back to best command accuracy")
    import shutil
    shutil.copyfile(run_dir / f"checkpoint_epoch{sel['epoch']}.pt",
                    run_dir / "checkpoint_best.pt")
    m_sel = sel["metrics"]
    log(f"selection: {guard} — epoch {sel['epoch']}  "
        f"cmd {m_sel['cmd_acc']:.4f}  slot {m_sel['slot_acc']}  "
        f"oos-ff {m_sel['oos_false_fire']:.3f}  "
        f"(wall {time.time() - t_start:.0f}s)")


    # ---------------- holdout ONE-SHOT (frozen protocol) ----------------
    m_ho = None
    if not args.smoke:
        flag = run_dir / "holdout.flag"
        if flag.exists() and not args.force_holdout:
            log("holdout: flag present — already evaluated once; SKIPPED "
                "(one-shot rule; --force-holdout overrides, do not use)")
        else:
            model.load_state_dict(torch.load(
                run_dir / "checkpoint_best.pt", map_location=device
            )["model_state"])
            ho_ds = V3Dataset("holdout", f_eval, augment=False, **ds_kw)
            log(ho_ds.summary())
            xh, yh, oh, hh, sh, dh = precompute_clean(
                ho_ds, f_eval, run_dir / "holdout_clean.npz", log)
            m_ho = evaluate_feats(
                model, xh, yh, hh, sh,
                [{"duration_s": float(d), "oos": bool(o)}
                 for d, o in zip(dh.numpy(), oh.numpy())], device)
            flag.write_text(f"evaluated {time.strftime('%Y-%m-%dT%H:%M:%S')}"
                            f" on checkpoint_best (epoch {sel['epoch']})\n")
            log(f"HOLDOUT ONE-SHOT: cmd {m_ho['cmd_acc']:.4f} "
                f"(n={m_ho['n']}, cmd clips={m_ho['n'] - m_ho['oos_n']})  "
                f"oos reject {m_ho['oos_reject']:.3f} (n={m_ho['oos_n']})  "
                f"slot {m_ho['slot_acc']}")

    # ---------------- numerals (report only) ----------------------------
    m_num = None
    if not args.smoke and not args.skip_numerals:
        num_ds = V3Dataset("numerals", f_eval, augment=False, **ds_kw)
        log(f"numerals (REPORT ONLY, no VAD audit -> file-end t1 fallback): "
            f"n={len(num_ds)}  vad-missing={num_ds.n_vad_missing}")
        xn, yn, on, hn, sn, dn = precompute_clean(
            num_ds, f_eval, run_dir / "numerals_clean.npz", log)
        m_num = evaluate_feats(
            model, xn, yn, hn, sn,
            [{"duration_s": float(d)} for d in dn.numpy()], device)
        log(f"NUMERALS: cmd {m_num['cmd_acc']:.4f} (n={m_num['n']})  "
            f"oos-ff {m_num['oos_false_fire']}  slot {m_num['slot_acc']}")

    # ---------------- results + CSVs ------------------------------------
    with open(run_dir / "results.json", "w") as fh:
        json.dump({
            "run": run_name, "arch": args.arch, "scale": 2,
            "config": vars(args), "git_hash": gh,
            "feature_policy": FEATURE_POLICY_VERSION,
            "class_weights": {c: float(w[i]) for i, c in enumerate(CLASSES)},
            "n_train_real": int(train_ds.counts.sum()),
            "n_gen_oos": gen_oos, "n_test": len(test_ds),
            "selection": {"rule": f"max slot acc among TEST epochs with "
                                  f"cmd acc >= {CMD_GATE}; else best cmd acc",
                          "guard": guard, "epoch": sel["epoch"],
                          "cmd_acc": m_sel["cmd_acc"],
                          "slot_acc": m_sel["slot_acc"]},
            "holdout_one_shot": m_ho, "numerals_report_only": m_num,
            "epochs": epoch_rows,
        }, fh, indent=2, default=float)
    with open(run_dir / "per_class_test.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["class", "n", "recall", "precision"])
        for c, name in enumerate(CLASSES):
            row = m_sel["cmd_per_class"][name]
            wr.writerow([name, row["n"],
                         "" if row["recall"] is None else f"{row['recall']:.4f}",
                         "" if row["precision"] is None
                         else f"{row['precision']:.4f}"])
    with open(run_dir / "slot_confusion_test.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["head", "true", "pred", "count"])
        for name in PARAMETRIC:
            for ti in range(3):
                for pi in range(3):
                    wr.writerow([name, ti, pi,
                                 m_sel["slot_conf"][name][ti][pi]])
    log(f"wrote {run_dir / 'results.json'} + per-class/slot-confusion CSVs")

    if args.smoke:
        log(f"SMOKE RESULT (unfitted {args.epochs}-epoch, subset): "
            f"cmd {m_sel['cmd_acc']:.4f} (n={m_sel['n']})  "
            f"slot {m_sel['slot_acc']}  oos-ff {m_sel['oos_false_fire']}")
        log("smoke OK: (B,1,80,150) in | (B,20) cmd | (B,3) per head | "
            "slot masking | OOS labels | no NaN losses")
    else:
        log(f"done: {run_dir}  (selection {guard}, epoch {sel['epoch']})")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

