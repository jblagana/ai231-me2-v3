# BENCHMARK.md — AI 231 ME2 V3

Canonical V3 run numbers live here. Pre-registered targets go in BEFORE any
run launches (V2 rule).

## V2 heritage (reference baselines)
| Model | EVAL-SYN (speaker-disjoint TTS) | EVAL-REAL (authentic) |
|---|---|---|
| V1 a2 (11c) | 0.889 (run v1i: 0.8618) | **0.2289** (baseline to beat) |
| v2a (bcresnet 10c, deployed in V2) | 0.9990 command | pending in V2 |

v2a slot detail: slot acc 0.5512 (9,757/17,700); weak heads = alarm/timer
(confusable number slots); Run B / B′ / D (slot-focused retrains) all FAILED
their pre-registered gates → lesson: with this corpus, slot accuracy is a
DATA problem (phrase coverage), not a retrain problem.

## V3 targets — PRE-REGISTERED 2026-10-02 (before the first run; V2 rule)

### Locked setup
- **Model: bcresnet scale=2 — PRIMARY (ratified 10-02, boss; V2a lineage)** —
  20 command classes, 6 slot heads (18 values), 30,134 params @20c, ~120 KB
  ONNX. Fallback: v2cnn (98,022 p @20c) — same locked feature, so the two
  remain interchangeable.
- **Feature (locked):** 3.0 s / 16 kHz / 80-mel log10 / 150 frames →
  (B, 1, 80, 150). Both arches consume it. Windowing (v2, FROZEN):
  right-align to VAD speech end — window = 3.0 s ending at min(d, t1+1.0 s),
  t1 from data/vad_audit/clip_vad_*.csv (VAD params frozen in
  tools/vad_audit.py); file-end right-align only as no-voiced fallback.
  (VAD audit: 0.6 / 1.7 / 23.3% of train / test / holdout carry ≥ 3.0 s
  trailing silence — v1 file-end right-align made the holdout gate
  unreachable; see DECISIONS "Truncation policy v2".)
- **Corpus (re-frozen 10-02, JFS rev 6947f130):** master Gold Dataset — train 10,733 (20
  classes incl. OUT_OF_SCOPE 270) / test 4,443 (OOS 76) / holdout 202 (OOS
  16) / numerals 66,390 (separate bench). Speaker-disjoint, verified ∅
  overlap.
- **Loss:** command head = class-weighted CE (weights below, FROZEN from
  train counts); 6 slot heads = plain CE (333 per value),
  active only for that intent's clips. No balanced sampler (avoids double
  correction).
- **Combined loss (FROZEN): L = L_cmd + 1.0 · L_slot.** Both terms are
  batch means — the slot term is averaged only over the batch's slotted
  clips with a valid canonical slot value (0 if none; the 158 NaN-slot
  train clips are command-only). Slot CE unweighted. Rationale: at
  inference a wrong command voids the slot, so command stays primary; the
  weighted command CE is already ~2× raw scale on average (fixed 3.147× /
  OOS 3.881×) vs the easy 3-class slot CE; V2's slot-focused retrains
  (B/B′/D) showed pushing slot loss harder doesn't fix data-limited slots.
  λ frozen for run 1; FIRST knob for run 2 (λ → 2.0) if slot gates are
  missed while command gates pass. Both terms logged per epoch.
- **Augmentation (train only, FROZEN):** time-warp 0.95–1.05 (right-aligned,
  tail preserved), gain ±10 dB, waveform-domain MUSAN SNR 5–25 dB. OOS
  augmentation: 270 OOS + generated silence +
  noise-only clips labeled OUT_OF_SCOPE (no time-warp; run-1 pre-registered: 135 gen silence + 135 gen noise-only) — the anti-silent-
  fire data half (weight 3.881 is the loss half).
- **Selection (FROZEN):** lexicographic on TEST per epoch — best = max slot
  accuracy among epochs with command accuracy ≥ 0.990; if no epoch passes,
  best command accuracy (GUARD FAILED logged).

### Frozen class-weight vector (w = 1/n_c, normalized so max class = 1.0)
| Class | n_train | weight |
|---|---:|---:|
| TEMPERATURE | 1,048 | 1.000 |
| TIMER | 1,036 | 1.012 |
| BRIGHTNESS | 1,032 | 1.016 |
| ALARM | 1,022 | 1.025 |
| CREATE_REMINDER | 1,015 | 1.033 |
| COLOR | 999 | 1.049 |
| 11 fixed intents | 333 each | 3.147 |
| CALL | 326 | 3.215 |
| MESSAGE | 322 | 3.255 |
| OUT_OF_SCOPE | 270 | **3.881** |

### Run-1 targets
| Metric | Split | Target |
|---|---|---|
| Command accuracy (overall) | test | ≥ 0.990 |
| Command accuracy (per-class macro) | test | ≥ 0.980 |
| Slot accuracy (overall, slotted clips) | test | ≥ 0.850 |
| Slot accuracy (worst head) | test | ≥ 0.750 on every head |
| OOS false-fire (OOS clips → any command) | test | ≤ 10% (≥ 68/76 rejected) |
| Holdout command acc (186 command clips) | holdout, ONE-SHOT | ≥ 0.950 |
| Holdout OOS rejection (16 clips) | holdout, ONE-SHOT | ≥ 13/16 (80% floor) |
| RPi 4B ONNX CPU, per 3 s clip | device | ≤ 100 ms (stretch ≤ 50 ms) |

Rationale: v2a hit 0.9990 command on EVAL-SYN (synthetic, 10c); the V3 test
mixes real + synthetic across 20 classes, so 0.990 overall is the honest
gate. V2's slot acc 0.5512 was on 7 heads / 119 values incl. open numeric
slots; V3 is 6 heads × 3 whitelisted values — ≥ 0.85 overall is the run-1
gate (reassess after run 1, before any retrain). OUT_OF_SCOPE weight 3.881
is the anti-silent-fire mechanism (V2 live bug).

### Protocol
Tune on TEST (freely). HOLDOUT is touched exactly ONCE, after the final
model is selected — the fixed demo benchmark (13 fixed × 3 variations +
6 slotted × 9 rendered phrases + 16 OOS). Numerals = number-robustness
report only, never part of the main score. Confidence gate: the
deployment no-action threshold is pre-registered from the OOS softmax
curve on TEST before run 1 (V2 set it ad hoc — flagged in the
clean-slate review). Robustness (group protocol): test command accuracy
also reported at SNR 20 / 10 / 5 dB (seeded waveform MUSAN mixing,
deterministic) alongside clean. Truncation diagnostic (per run): test
command accuracy on clips with original duration > 3.0 s vs ≤ 3.0 s —
empirical basis for any future window-length call (no window change in
run 1/2; the ratified wake reuse is locked to 150 frames).
