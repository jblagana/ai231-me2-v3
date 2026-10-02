"""V3 model zoo — carried over from V2 (two architectures, both two-head:
command + 6 slot heads for the V3 schema). Only change vs V2: the
pinned-budget smoke check retuned to the V3 slot vocab (6 heads / 18
values). V2's deployed model = bcresnet 10c (v2a). V3 (ratified
2026-10-02): 20 command classes (19 intents + OUT_OF_SCOPE), bcresnet
PRIMARY (30,134 p @20c) / v2cnn fallback (98,022 p @20c).

Both consume the LOCKED feature: (B, 1, 80, 150) log10 mel, 3.0 s window,
16 kHz, hop 20 ms. The window is baked into the weights (DECISIONS.md) —
any architecture change must use the same feature.

1. v2cnn (FALLBACK — locked budget) — V1's A2 architecture
    (~/vcm/src/model_a2.py):
   3-block CNN 32/64/128 (3x3 pad1, BN, ReLU, MaxPool2 x3) -> (B,128,10,18);
   command head: global max-pool over all 18 time cells -> Dropout(0.3) ->
   Linear(128, n_class); slot heads: max-pool over the last
   SLOT_TAIL_CELLS=12 cells (~2.0 s; 10-02 amendment: the V1-validated 1.33 s
   tail was sized for short V1/V2 slot words, but V3's multi-word slot values
   + the live 1.0 s silent tail must both fit the read region) ->
   per-class Linear(128, N_k), 6 heads (V3 slot vocab: 18 values). 96,861 params at 11 classes (conv 93,120 + cmd FC 1,419 +
   slot FC 2,322); 97,893 at 19 classes.

2. bcresnet (PRIMARY — ratified 2026-10-02; proposed 2026-09-30) — BC-ResNet, "Broadcasted Residual
   Learning for Efficient Keyword Spotting" (arXiv 2106.04140, Interspeech
   2021, Kim et al.). Faithful port of the MIT reference implementation
   (github.com/re9ulus/BC-ResNet): most residual functions are 1D TEMPORAL
   convs (depthwise (1,3) with dilation) applied to the freq-mean of the
   preceding path's output and BROADCAST back across all freq bins
   (the broadcasted residual connection); depthwise (3,1) freq path plus
   1x1 channel-change transitions with stride-2 time downsampling.
   Scaling rule (paper): multiply all channels by `scale`; we use scale=2
   (the reference's SOTA config: 0.962 val / 0.960 test on 35-class Speech
   Commands). Documented deviations from the reference:
   - input is our locked 80-mel/150-frame log10 feature (reference: 40-mel,
     log2, variable-length padded);
   - final pooling: the reference's bare `squeeze()` only works when the
     freq axis collapses to 1; ours is global-avg over F x T for the command
     head, and max-pool over the last 100 time cells for the slot heads
     (2.0 s read region = live 1.0 s silent tail + full multi-word slot
     value; 10-02 amendment, time-equivalent to v2cnn's 12-cell tail);
   - dw_conv uses groups=C (true depthwise; identical to the reference at
     scale=1 where C=groups=20).
   - SubSpectralNorm (arXiv 2103.13620) included (F=40 divisible by 5);
     use_subspectral=False falls back to plain BatchNorm2d.

Usage:
  python src/models.py --smoke   # all 4 configs: shapes + param counts
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from slots import PARAMETRIC, SLOT_COUNTS  # noqa: E402

N_MELS = 80
SR = 16000
WINDOW_S = 3.0
HOP_S = 0.020
N_FRAMES = int((WINDOW_S - HOP_S) / HOP_S) + 1  # 150
SLOT_TAIL_CELLS = 12         # v2cnn: 1 cell ~ 167 ms -> read region ~ 2.0 s
BC_SLOT_TAIL_CELLS = 100     # bcresnet: 1 cell ~ 20 ms  -> read region ~ 2.0 s
                             # (10-02 amendment, instruction 16: V1's
                             # validated 1.33 s tail was sized for V1/V2's
                             # short slot words. V3's slot values are
                             # multi-word phrases (0.3-1.2 s; all 18
                             # templates end in {slot}) and the frozen live
                             # geometry leaves a 1.0 s silent tail (demo
                             # silence stop). Read region = tail + full slot
                             # value = 1.0 + 1.0 = 2.0 s, time-equivalent in
                             # both nets. Zero parameter change; no
                             # live/demo change. Probes: tools/slot_tail_check.py,
                             # tools/slot_len_check.py.)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# v2cnn — fallback (locked budget)
# ---------------------------------------------------------------------------

class V2CNN(nn.Module):
    """A2 two-head CNN (V1 heritage `~/vcm/src/model_a2.py`), 6 slot heads
    (V3 schema).

    forward(x) -> (cmd_logits (B, n_classes), slot_feat (B, 128)).
    slot_logits(slot_feat, cls) -> (B, N_c) for one parametric class.
    """

    def __init__(self, n_classes: int, n_mels: int = N_MELS, dropout: float = 0.3,
                 slot_tail_cells: int = SLOT_TAIL_CELLS):
        super().__init__()
        self.slot_tail_cells = slot_tail_cells
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cmd_pool = nn.AdaptiveMaxPool2d(1)
        self.cmd_fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, n_classes))
        self.slot_pool = nn.AdaptiveMaxPool2d(1)
        self.slot_heads = nn.ModuleDict(
            {c: nn.Linear(128, n) for c, n in SLOT_COUNTS.items()})

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Shared conv features: (B,1,F,T) -> (B,128,10,18)."""
        return self.conv(x)

    def forward(self, x: torch.Tensor):
        h = self.conv(x)
        cmd_logits = self.cmd_fc(self.cmd_pool(h).flatten(1))
        hs = h[:, :, :, -self.slot_tail_cells:]
        slot_feat = self.slot_pool(hs).flatten(1)
        return cmd_logits, slot_feat

    def slot_logits(self, slot_feat: torch.Tensor, cls: str) -> torch.Tensor:
        return self.slot_heads[cls](slot_feat)


# ---------------------------------------------------------------------------
# bcresnet — PRIMARY (arXiv 2106.04140)
# ---------------------------------------------------------------------------

BC_DROPOUT = 0.1


class SubSpectralNorm(nn.Module):
    """BN over grouped sub-bands of the freq axis (arXiv 2103.13620)."""

    def __init__(self, channels: int, sub_bands: int = 5, eps: float = 1e-5):
        super().__init__()
        self.sub_bands = sub_bands
        self.bn = nn.BatchNorm2d(channels * sub_bands, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, Fq, T = x.size()
        x = x.view(N, C * self.sub_bands, Fq // self.sub_bands, T)
        x = self.bn(x)
        return x.view(N, C, Fq, T)


class BCNormalBlock(nn.Module):
    """Broadcasted residual block (paper sec. 3.1).

    x1 = depthwise (3,1) freq conv + norm (the 2D part)
    x2 = freq-mean(x1) -> depthwise (1,3) temporal (dilated) + BN + SiLU
         -> 1x1 channel mix + dropout -> broadcast across freq bins
    out = ReLU(x + x1 + x2)
    """

    def __init__(self, n_chan: int, *, dilation: int = 1,
                 dropout: float = BC_DROPOUT, use_subspectral: bool = True):
        super().__init__()
        norm = (SubSpectralNorm(n_chan, 5) if use_subspectral
                else nn.BatchNorm2d(n_chan))
        # Explicit padding (ONNX portability: onnxruntime rejects
        # dilation + auto_pad=SAME; for these kernels same == these pads)
        self.f2 = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, (3, 1), padding=(1, 0), groups=n_chan),
            norm,
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, (1, 3), padding=(0, dilation), groups=n_chan,
                      dilation=(1, dilation)),
            nn.BatchNorm2d(n_chan),
            nn.SiLU(),
            nn.Conv2d(n_chan, n_chan, 1),
            nn.Dropout2d(dropout),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_freq = x.shape[2]
        x1 = self.f2(x)
        x2 = self.f1(x1.mean(dim=2, keepdim=True)).repeat(1, 1, n_freq, 1)
        return self.activation(x + x1 + x2)


class BCTransitionBlock(nn.Module):
    """Channel change (1x1) + optional stride-2 time downsampling.

    Residual skip is the f2 output (1x1 projects the channels), matching the
    reference: out = ReLU(f2(x) + broadcast(f1(freq-mean(f2(x))))).
    """

    def __init__(self, in_chan: int, out_chan: int, *, dilation: int = 1,
                 stride: int = 1, dropout: float = BC_DROPOUT,
                 use_subspectral: bool = True):
        super().__init__()
        if stride == 1:
            conv = nn.Conv2d(out_chan, out_chan, (3, 1), groups=out_chan,
                             padding=(1, 0))
        else:
            conv = nn.Conv2d(out_chan, out_chan, (3, 1), stride=(stride, 1),
                             groups=out_chan, padding=(1, 0))
        norm = (SubSpectralNorm(out_chan, 5) if use_subspectral
                else nn.BatchNorm2d(out_chan))
        self.f2 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            conv,
            norm,
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, (1, 3), padding=(0, dilation),
                      groups=out_chan, dilation=(1, dilation)),
            nn.BatchNorm2d(out_chan),
            nn.SiLU(),
            nn.Conv2d(out_chan, out_chan, 1),
            nn.Dropout2d(dropout),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.f2(x)
        n_freq = x.shape[2]
        x1 = self.f1(x.mean(dim=2, keepdim=True)).repeat(1, 1, n_freq, 1)
        return self.activation(x + x1)


class BCResNet(nn.Module):
    """BC-ResNet (arXiv 2106.04140) — V2 two-head adaptation.

    Layout (s = scale): input_conv(1->16s, (5,5), stride (2,1))
      t1(16s->8s) n11
      t2(8s->12s, dil 2, stride 2) n21
      t3(12s->16s, dil 4, stride 2) n31 n32 n33
      t4(16s->20s, dil 8) n41 n42 n43
      dw_conv(5,5) onexone(->32s)
    Axis convention (as in the reference): stride-2 downsampling hits the
    FREQ axis, time is preserved (only trimmed by the final dw 5x5). Final
    feature (B, 32s, 6, 146): F 80 -> 40 -> 20 -> 10 -> 6 (this is exactly
    why the reference's squeeze() works on 40-mel input: 40 -> 1); T 150 ->
    146. This is the paper's core design — 1D temporal convs with dilations
    1/2/4/8 running over the FULL time axis, freq compressed to sub-bands,
    broadcasted residual across them.
    Command head: global avg over F x T -> Linear(32s, n_class).
    Slot heads:   per-channel max-pool over the last BC_SLOT_TAIL_CELLS=100
    time cells (~2.0 s; 10-02 amendment — must cover the live 1.0 s silent
    tail plus the full multi-word slot value; time-equivalent to v2cnn's
    12-cell tail) -> Linear(32s, N_k) x6.
    """

    def __init__(self, n_classes: int, *, scale: int = 2,
                 dropout: float = BC_DROPOUT, use_subspectral: bool = True,
                 slot_tail_cells: int = BC_SLOT_TAIL_CELLS):
        super().__init__()
        s = scale
        self.slot_tail_cells = slot_tail_cells
        self.input_conv = nn.Conv2d(1, 16 * s, (5, 5), stride=(2, 1), padding=2)
        self.t1 = BCTransitionBlock(16 * s, 8 * s, dropout=dropout,
                                    use_subspectral=use_subspectral)
        self.n11 = BCNormalBlock(8 * s, dropout=dropout,
                                 use_subspectral=use_subspectral)
        self.t2 = BCTransitionBlock(8 * s, 12 * s, dilation=2, stride=2,
                                    dropout=dropout,
                                    use_subspectral=use_subspectral)
        self.n21 = BCNormalBlock(12 * s, dilation=2, dropout=dropout,
                                 use_subspectral=use_subspectral)
        self.t3 = BCTransitionBlock(12 * s, 16 * s, dilation=4, stride=2,
                                    dropout=dropout,
                                    use_subspectral=use_subspectral)
        self.n31 = BCNormalBlock(16 * s, dilation=4, dropout=dropout,
                                 use_subspectral=use_subspectral)
        self.n32 = BCNormalBlock(16 * s, dilation=4, dropout=dropout,
                                 use_subspectral=use_subspectral)
        self.n33 = BCNormalBlock(16 * s, dilation=4, dropout=dropout,
                                 use_subspectral=use_subspectral)
        self.t4 = BCTransitionBlock(16 * s, 20 * s, dilation=8, dropout=dropout,
                                    use_subspectral=use_subspectral)
        self.n41 = BCNormalBlock(20 * s, dilation=8, dropout=dropout,
                                 use_subspectral=use_subspectral)
        self.n42 = BCNormalBlock(20 * s, dilation=8, dropout=dropout,
                                 use_subspectral=use_subspectral)
        self.n43 = BCNormalBlock(20 * s, dilation=8, dropout=dropout,
                                 use_subspectral=use_subspectral)
        self.dw_conv = nn.Conv2d(20 * s, 20 * s, (5, 5), groups=20 * s)
        self.onexone_conv = nn.Conv2d(20 * s, 32 * s, 1)
        self.cmd_fc = nn.Linear(32 * s, n_classes)
        self.slot_heads = nn.ModuleDict(
            {c: nn.Linear(32 * s, n) for c, n in SLOT_COUNTS.items()})

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)
        x = self.t2(x)
        x = self.n21(x)
        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)
        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)
        x = self.dw_conv(x)
        x = self.onexone_conv(x)
        return x

    def forward(self, x: torch.Tensor):
        h = self.features(x)
        cmd_logits = self.cmd_fc(h.mean(dim=(2, 3)))
        hs = h[:, :, :, -self.slot_tail_cells:]
        # per-channel max over (F, tail slice) -> (B, 32s); same slot
        # semantics as v2cnn. (The V2 port's max-over-C-per-time-cell
        # variant only matched Linear(32s) by the 64==64 coincidence and
        # breaks for any other tail width.)
        slot_feat = hs.amax(dim=(2, 3))  # (B, 32s)
        return cmd_logits, slot_feat

    def slot_logits(self, slot_feat: torch.Tensor, cls: str) -> torch.Tensor:
        return self.slot_heads[cls](slot_feat)


class SlotOnlyBCResNet(BCResNet):
    """Run D (boss GO 10-01): slot-only BCResNet — NO command head, NO
    command loss. forward(x) -> slot_feat (B, 32s) only; the 7 class-
    conditional slot heads are the sole output. Trained on the identical
    raw_v2 corpus (same 3 s window / 64-cell tail / labels): the class
    label only routes WHICH head is supervised, contributing zero
    "command-ness" to the gradient. The frozen v2a command model gates
    the class upstream on the Pi (3rd ONNX in the package).
    """

    def __init__(self, n_classes: int, **kwargs):
        super().__init__(n_classes, **kwargs)
        del self.cmd_fc  # no command head -> no command gradient, ever

    def forward(self, x: torch.Tensor):
        h = self.features(x)
        hs = h[:, :, :, -self.slot_tail_cells:]
        slot_feat = hs.amax(dim=(2, 3))  # (B, 32s) — see BCResNet
        return slot_feat


class CmdOnlyBCResNet(BCResNet):
    """v2f command model (10-02): bcresnet with ONLY the command head —
    no slot heads, no slot loss. The dedicated command gate of the v2f
    two-model pipeline (the v2f slot model owns the slots).
    """

    def __init__(self, n_classes: int, **kwargs):
        super().__init__(n_classes, **kwargs)
        del self.slot_heads

    def forward(self, x: torch.Tensor):
        h = self.features(x)
        return self.cmd_fc(h.mean(dim=(2, 3)))


MODEL_ZOO = {
    "v2cnn": V2CNN,
    "bcresnet": BCResNet,
}


def build_model(name: str, n_classes: int, **kwargs):
    return MODEL_ZOO[name](n_classes, **kwargs)


def smoke() -> int:
    ok = True
    x = torch.randn(4, 1, N_MELS, N_FRAMES)
    for name in MODEL_ZOO:
        for nc in (11, 10):
            m = build_model(name, nc)
            cmd, sf = m(x)
            n = param_count(m)
            feat = m.features(x)
            # v2cnn slot_feat: (B, C) after slot-pool + flatten; bcresnet:
            # (B, C, F) — the per-head Linear flattens the non-batch dims
            feat_c = feat.shape[1]
            shapes_ok = (cmd.shape == (4, nc)
                         and sf.shape[0] == 4 and feat_c in sf.shape
                         and all(m.slot_logits(sf, c).shape == (4, SLOT_COUNTS[c])
                                  for c in PARAMETRIC))
            under_budget = n < 1_000_000
            ok = ok and shapes_ok and under_budget
            print(f"{name}-{nc}c: feat {tuple(feat.shape)}  cmd {tuple(cmd.shape)}"
                  f"  slot_feat {tuple(sf.shape)}  params {n:,}  "
                  f"shapes={'OK' if shapes_ok else 'FAIL'}  "
                  f"<1M={'OK' if under_budget else 'OVER'}")
    # pinned check: v2cnn must stay the locked A2 conv budget; the V3
    # slot vocab (6 heads / 18 values) sets the slot-FC total
    m = V2CNN(11)
    n = param_count(m)
    conv = param_count(m.conv)
    cmdp = param_count(m.cmd_fc)
    slotp = sum(param_count(h) for h in m.slot_heads.values())
    pinned = (n == 96_861 and conv == 93_120 and cmdp == 1_419 and slotp == 2_322)
    ok = ok and pinned
    print(f"v2cnn-11c pinned budget: total {n:,} / conv {conv:,} / "
          f"cmd {cmdp:,} / slot {slotp:,}  "
          f"{'OK' if pinned else 'FAIL (must stay 96,861)'}")
    # rough CPU latency (HPC CPU here; RPi will be slower — ONNX is the real check)
    m = V2CNN(11).eval()
    with torch.no_grad():
        for _ in range(10):
            m(x)
        t0 = time.perf_counter()
        for _ in range(100):
            m(x)
        dt = (time.perf_counter() - t0) / 100
    print(f"v2cnn CPU forward (batch 4, this machine): {dt*1000:.1f} ms -> "
          f"~{dt*1000/4:.2f} ms per 3 s clip")
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sys.exit(smoke() if args.smoke else 0)



