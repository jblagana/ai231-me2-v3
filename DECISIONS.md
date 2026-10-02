# DECISIONS.md — AI 231 ME2 V3

Ratified decisions (boss = Jan) + the why. New decision → add an entry here
AND in `INSTRUCTIONS.md`.

## Training epoch budget: 30 -> 100 (boss question 2026-10-02; ratified)
- The FROZEN selection rule (BENCHMARK.md) is epoch-count-agnostic: it picks,
  per epoch, the best TEST epoch = max slot-acc among TEST cmd acc >= 0.990,
  else best cmd-acc (GUARD FAILED). So "check train/TEST divergence for model
  selection" is ALREADY the protocol; more epochs just enlarge the candidate
  pool and expose the gap — they cannot hurt selection (it never picks an
  overfit late epoch).
- **Run 1 uses --epochs 100** (was 30). Rationale: (1) a 100-epoch curve gives a
  clear, defensible train-loss-falls / TEST-plateaus divergence to SHOW in the
  submission; (2) ~3x headroom over the pre-registered 30 at no risk; (3) it is a
  single self-contained frozen number, so it stays reproducible.
- **500 rejected.** A 30,134-param model on 10.7k clips asymptotes well before
  500; each epoch past the TEST plateau is wasted (selection will never pick it).
  Wall-clock is ~linear in epochs and data-loading bound (the arrow IPC stream is
  single-reader); 500 is ~5x a 100-epoch run = multi-day. The cosine LR schedule
  is baked to total steps (ceil(N/32) x epochs), so a 500-epoch plan pushes the
  useful low-LR region to the very end and burns GPU wandering an overfit plateau.
  Do NOT "start at 30 and extend to 100 later" — the schedule is set at launch.
- Early-stop/patience was considered and REJECTED for run 1 to keep the harness
  deterministic + reproducible (a submission). If run 1's TEST curve shows a hard
  plateau by epoch ~40, that is the evidence 100 was already generous.

## Submission packaging (public, one-command, MIT) — 2026-10-02
- Repo made push-ready: `LICENSE` (MIT), reproduction-first `README.md`,
  `requirements.txt` / `requirements-train.txt` / `requirements-pi.txt`,
  `scripts/{fetch_data.py, train.sh, smoke.sh, build_slides.py}`,
  `tools/manifest_stats.py` (exact slide numbers: clips/hours/speakers/source),
  `tools/export_onnx.py` (opset-17; input mel (1,80,150); outputs cmd(20) +
  6x slot(3); --verify torch<->ONNX parity), `.gitignore` hardened (no 4 GB
  corpus, no *.pt/*.onnx, no third_party drift), `SUBMISSION.md` tracker.
- Slides: `slides_data.json` = single source of truth (every field + status
  known/pending/estimate + source); `scripts/build_slides.py` renders
  `slides/me2_v3_submission.pptx` (the 2 required template slides + appendix).
  Pending numbers are amber and filled post-training (SUBMISSION.md section 6).
- Architecture correction for the deck: the template's "Causal encoder /
  KV-cache / streaming" box does NOT match the model. Both models are
  fixed-window BC-ResNet KWS classifiers (wake = V2-reused; command/slot =
  scale=2) sharing one 80-mel / 3.0 s feature — no streaming / KV-cache.
- "Baseline of comparable size": in-repo `v2cnn` fallback (98,022 p, same
  feature) + V2 v2a (bcresnet 10c, 0.9990 EVAL-SYN) as the lineage baseline.
  (Optional extra: one `--arch v2cnn` run for a same-data size comparison.)

## Taxonomy (FINAL — group meeting 2026-10-02, instruction 7; supersedes
the 10-class V2 port)
- **19 intents** per Sir Mark's Dataset Schema (Option B) — the group's
  ratified basis for all labeling (schema sheet:
  docs.google.com/spreadsheets/d/1HZ7Q58S-konS5XWUE6FUGC-S-C8qAoSPaVbIP4NYSU0;
  machine-readable: github.com/markandrian30/AI231 MEX2/Data labels.json +
  slots.json):
  - 13 fixed (3 phrase variations each): PLAY_MUSIC, VOLUME_UP,
    VOLUME_DOWN, NEXT, PAUSE, STOP, LIGHT_ON, LIGHT_OFF, CALL, MESSAGE,
    LIST_REMINDERS, WEATHER, TIME
  - 6 variable (3 templates × 3 slot values each): BRIGHTNESS
    (20/60/100 percent), COLOR (red/blue/green), TEMPERATURE
    (18/22/26 degrees), TIMER (10 seconds/30 seconds/1 minute),
    ALARM (6 AM/8 AM/9 PM), CREATE_REMINDER (drink water/study/exercise)
- **Slots: 6 heads, 18 values** (`src/slots.py`, verbatim from
  slots.json). CONFIRMED against the master Gold Dataset manifest
  (2026-10-02): the manifest carries EXACTLY these 18 slot values — the
  earlier "~93 slot-value classes" report was the phrase-bucket count,
  not slot values. Surface-form deltas normalized at load: alarm ":00"
  rendering ("6:00 AM" ↔ "6 AM") and title-cased color/reminder values
  ("Red" ↔ "red").
- **Command head = 20 classes: 19 intents + OUT_OF_SCOPE** (ratified
  2026-10-02, boss) — the rejection class is trained on the manifest's OOS
  clips (201 train / 47 test / 10 holdout) and fixes V2's live silent-fire
  bug. It has no canonical phrase (off-vocabulary by definition).
- The 3 phrase variations per intent (Option B wording) are the **fixed
  demo benchmark** (master dataset holdout split, 196 clips, one-shot).
- Rules: closed vocabulary (schema values only); no locations in the
  19-intent phrasebook; group recordings stay class-private (consent).

## Model arch (locked 2026-10-02 — carried from V2; retuned to the V3 schema)
- **Feature (locked):** 3.0 s window, 16 kHz, 80-mel log10, 150 frames
  (20 ms hop) → (B, 1, 80, 150). The window is baked into the weights.
- **v2cnn (FALLBACK — locked budget):** 3-block CNN 32/64/128 (3×3 pad1,
  BN, ReLU, MaxPool2 ×3); command head = global max-pool → Dropout(0.3) →
  Linear; slot heads = max-pool over the last 12 cells (~2.0 s read region
  — 10-02 amendment, instruction 16: the V1-validated 1.33 s tail was sized
  for short V1/V2 slot words; V3's slot values are multi-word (0.3–1.2 s;
  all 18 templates end in {slot}) and the live 1.0 s silent tail must also
  fit the read region: 1.0 + 1.0 = 2.0 s. Post-hoc corpus stat 10-02 on the
  re-frozen revision (slot clips 5,994 / 2,538 / 108 train/test/holdout):
  V3 commands are short (median voiced span 1.5 s), so under the OLD 1.33 s
  region the MEDIAN slot value was 0% visible — 61-65% of slot clips had NO
  slot audio inside the read region; under the 2.0 s region median 75-77%
  visible, ~36% fully inside, ~10% still invisible (sub-1 s commands).
  Slot starts estimated from word-count proportion (no word-level
  alignment in the data). The old region was V1/V2-sized and simply wrong
  for V3 phrase lengths), 6 per-class Linears. 96,861
  params @11c /
  98,022 @20c (18 slot values).
- **bcresnet (PRIMARY — ratified 2026-10-02, boss; V2's winner):** BC-ResNet
  (arXiv 2106.04140), scale=2, 29,549 params @11c / 30,134 @20c (V3 slot
  vocab: 6 heads / 18 values), ~120 KB ONNX. V2's deployed v2a = bcresnet
  10c (EVAL-SYN command acc 0.9990; V2's 36,114-param figure included the
  old 7-head/119-value slot block). Both arches consume the same locked
  feature, so primary/fallback remain interchangeable.
- **bcresnet input deviation (vs the paper, deliberate):** the paper's
  official repo feeds 40-mel log-mel, ~100 frames (1.0 s GSC clips @ 10 ms
  hop; 30 ms window, n_fft 512). Its fixed freq-halving chain (5×5 head
  stride-2 + two stride-2 stages) lands exactly on F=1, so its classifier
  squeezes to 1-D time. Ours feeds the locked 80-mel / 150-frame / 3.0 s
  feature (shared with v2cnn, 20 ms hop): the same chain lands at F=6, so
  the command head global-averages over F×T instead of squeezing, and the
  6 slot heads read the last 100 time cells (~2.0 s; 10-02 amendment —
  see the v2cnn entry; zero parameter change). Same block schedule, dilations
  1/2/4/8, SubSpectralNorm(5) — only the input grid and the heads differ.
- Inference artifact: **ONNX Runtime CPU** on RPi 4B. No ASR, no LLM, no
  cloud at inference — hard spec. Training: HPC n002/n003 now; group DGX
  cluster (SHARCHPC) when the dataset path is confirmed.

## Clean-Slate Review (2026-10-02) — V1/V2 pipeline audit → V3 deltas
Method: line-by-line audit of the V1/V2 workspace (`AI231 ME2 V2`:
src/mel.py, train_v2.py, export_v2.py, train_wake.py,
tools/make_runb_weights.py, DECISIONS/BENCHMARK) against the V3 goal,
disregarding path dependence.

**Flags — unreasonable / problematic V1/V2 components (ranked):**
1. **No trained reject class (V1+V2).** The VCM shipped firing commands on
   silence; mitigated by an external wake gate + ad-hoc confidence
   threshold, not by learning. V2's #1 live bug. → V3: 20th OOS class +
   silence/noise OOS augmentation (below).
2. **Open-numeric slot vocab: 119 values / 7 heads** (48 alarm marks, 24
   timer values, 13 temperatures). A 30–100K-param KWS cannot
   disambiguate 119 phonetically close spoken numbers from 3–9 phrases
   each — slot acc 0.5512, all slot-focused retrains failed. Structurally
   wrong task for the budget. → V3: 18 whitelisted values (3/head).
3. **V1: selection on EVAL-SYN only** while EVAL-REAL sat at 0.229 —
   selection set ≠ deployment distribution. → V3: mixed real+synthetic test
   selection + one-shot holdout.
4. **V1: train/val leak** (the 0.33→0.98 collapse). V2's leak-free master +
   V3's verified speaker-disjoint splits fix it; keep the verification as a
   standing gate.
5. **v2f two-model pipeline** (command-only gate + slot-only; 3 ONNX with
   the wake gate). Decoupled a coupled two-head task to enable retrains;
   the frozen v2a upstream gate is pipeline complexity a single 20-class
   two-head model obviates. Do not carry to V3.
6. **Run B slot weights from the EVAL error matrix** (1+2·err, cap 4) —
   retraining against the selection set's errors; its own pre-registered
   gate showed it could not pass (B/B′/D all failed). Wrong lever; the
   right one was schema/data (V3 did that).
7. **MEL-domain noise mixing** — energy computed in the log domain; an
   approximation of waveform mixing. Tolerable in V2; V3 mixes in the
   waveform domain (raw audio is on hand).
8. **±3 dB gain augmentation** (V2's ±6/20 in normalized log10) — narrow
   vs real mic/voice gain variance. → V3: ±10 dB.
9. **Pi naive-linear resample (np.interp)** — only safe with a 16 kHz
   locked mic; hidden dependency. V3: assert/lock 16 kHz capture on the Pi.
10. **Confidence gate threshold ad-hoc** ("below threshold → no action",
    no pre-registered operating point). → V3: pre-register from the OOS
    curve on test.
11. **"VAD" = 600 ms RMS silence timeout** — pragmatic demo scope;
    limitation under babble/TV noise (no true VAD/echo cancellation).
    Documented, not fixed.

**V3 deltas from the audit (ratification pending):**
1. **Selection:** lexicographic on TEST — max slot acc among epochs with
   cmd acc ≥ 0.990; fallback best cmd (guard logged). V2's v2e gated
   protocol, aligned to V3's pre-registered gate.
2. **OOS augmentation (train only):** 201 OOS clips + 11 empty-transcript
   noise clips + generated silence + noise-only (SNR 5–25) all labeled
   OUT_OF_SCOPE. No time-warp on OOS.
3. **Augmentation (train only):** time-warp 0.95–1.05 (right-aligned, tail
   preserved), gain ±10 dB, waveform-domain MUSAN SNR 5–25.
4. **λ stays 1.0** (V2's validated 0.5 was paired with per-value slot
   weights; V3's heads are balanced 3-value, so either is fine — frozen at
   1.0).
5. **Deployment (ratified 2026-10-02 — boss: reuse V2's wake gate).**
   V2's wake bcresnet (2-class "hey boots"; balanced 0.9990, wake recall
   0.94, no-wake FPR 0.0000, ONNX parity verified; ckpt
   `~/ai231_me2_v2/runs/v2w/bcresnet/vcm_wake_bcresnet_best.pt`, export
   via V2's export_wake.py). No retraining needed: no_wake = any
   non-wake audio, V3's new phrases included. Pi package = **2 ONNX
   sharing one numpy mel + buffers**: wake_model.onnx (V2) +
   me2_vcm_v3.onnx (opset 17, precomputed-mel input, 20c cmd + 6 slot
   outputs) + mel_window/mel_fb buffers + mel.py + verify_pi_v3 bit-exact
   parity gate. Capture (V2-validated): rolling 3.0 s → wake → RMS 600 ms
   silence timeout → right-pad → VCM → confidence gate → beep/LED.
   Confidence operating point pre-registered. The 20-class OOS head =
   second rejection layer inside the command window (defense in depth vs
   V2's silent-fire bug).
6. **Feature pipeline: computation UNCHANGED, engineering changed.**
   Locked / V2-validated / shared by both ONNXs: 16 kHz, 48,000-sample
   window, right-aligned keep-last-3.0-s / left-pad, Hann 1024 / hop 320,
   80-mel f50–8000, power → log10 (clip 1e-5) → (x+5)/5, 151 → 150
   frames; numpy Pi port + shipped buffers; no per-clip normalization; no
   masking/SpecAugment. Changes: (a) noise mixed in the WAVEFORM domain
   (V2: mel domain) — mel computed on the fly per sample, train
   base-feature cache dropped (kept for clean eval splits); (b) gain
   ±3 → ±10 dB; (c) time-warp 0.95–1.05 stays mel-domain, applied AFTER
   the noise (order: noise → mel → warp → gain), tail-preserving
   right-align; (d) new OOS data path: MUSAN bank (732 clips,
   `~/vcm/data/noise16k`) + generated silence → mel → OUT_OF_SCOPE, no
   time-warp.

**Keep — V1/V2 best practices (verified in code, carry to V3):**
Pi numpy mel with shipped buffers + bit-exact parity verification (feature
mismatch = the #1 silent on-device accuracy killer); right-aligned
keep-last-3.0-s + left-pad (the slot is the last word and must sit at the
tail the slot head reads); (x+5)/5 deterministic normalization (no
train/Pi distribution shift); base-feature precompute + in-RAM augmentation;
AdamW lr 1e-3 / wd 1e-4 + cosine, batch 32, patience early stop;
pre-registration + at-most-one targeted retrain; speaker-disjoint splits;
no spoken ack (beep/LED — echo protection); tmpfs ring buffer (SD
bottleneck); ONNX graph = conv/pool/linear/BN only (mel outside the graph,
no FFT op needed).

## Feature grid: why NOT the paper's 40-mel (boss question, 2026-10-02)
Q: remake the data on the paper's features (40-mel, 1.0 s, 10 ms hop,
log-mel)? A: **No — the locked 80-mel / 3.0 s / 150-frame / 20 ms / log10
grid stays.**
1. Nothing to remake: features are computed at load time, not recorded —
   the 81,686 HF clips are raw audio.
2. **Ratified wake reuse is incompatible with any grid change:** the V2
   wake bcresnet consumes exactly (1, 80, 150) with the shipped mel
   buffers (export_wake.py) — a new grid retires the wake gate (boss
   decision 10-02).
3. **1.0 s window kills the slot mechanism:** slot = last spoken word;
   slot heads read the last ~1.3 s tail; corpus median ≈ 1.9 s and 14.2%
   of train clips exceed 3.0 s (duration audit).
4. **40-mel = lower frequency resolution** — fine for the paper's
   10-class 1-s KWS (F→1 collapse), a liability for 20 phonetically close
   intents + 18 spoken number values. We have headroom (v2cnn 98K on the
   same grid).
5. **log-mel (ln) vs log10 is a constant 2.303× rescale** — zero
   information difference, first conv + BatchNorm absorb it. (x+5)/5 is
   just calibrated to log10's [−5, 0] clip range; ln would only change it
   to (x+11.51)/11.51.
6. **20 ms hop is V2-Pi-validated** (bit-exact numpy parity, shipped
   buffers) and halves on-device STFT compute vs 10 ms; 50 fps temporal
   density per second is unchanged in spirit (150f/3.0 s vs 100f/1.0 s).
What we already take from the paper: the architecture (bcresnet scale=2)
as-is; documented geometry deviation (freq axis lands F=6 not F=1) is
benign — the halving chain still works.
**Pre-registered:** a feature-grid change is NOT a run-2 lever (run-2
knob order: λ → v2cnn fallback → augmentation/data); last-resort only if
run-1 command gates fail — and even then 3.0 s / 150 frames are kept.

## Truncation policy v2: right-align to speech end (2026-10-02, data-driven)
VAD clip-anatomy audit (tools/vad_audit.py; 10 ms RMS frames; voiced =
RMS > max(1e-3, 10% of peak frame RMS); t1 = last voiced frame; per-clip
t1 saved in data/vad_audit/clip_vad_{split}.csv):

| split | n | RISK: speech ends > 3.0 s before EOF |
|---|---:|---:|
| train | 10,682 | 0.6% (65) |
| test | 4,418 | 1.7% (77, all real_voice) |
| holdout | 196 | **17.9% (35, all real_voice)** |

The case flagged by the boss — [0–1.5 s command][1.5–5 s silence] (fixed
5.0 s group-recording capture) — is REAL: under v1 (right-align to file
end) those clips feature as pure silence labeled as a command.
Consequences if left unfixed: (a) 65 train clips teach silence=command
(silent-fire); (b) the one-shot holdout gate ≥ 0.950 becomes
mathematically unreachable — 35/196 = 17.9% of the demo benchmark scored
against silence windows (ceiling ≈ 0.82). V2 never saw this case (its
corpus was uniform 3.0 s TTS).

**Policy v2 (FROZEN): right-align to speech end, not file end.**
- Window = 3.0 s ending at min(d, t1 + 1.0 s) — keeps 1.0 s trailing
  silence = EXACTLY the shipping live-capture geometry: V2 pi_demo.py
  `capture_command(silence_ms=1000, max_ms=3500)` ends 1.0 s after the
  last voiced frame (AMENDED 10-02, boss: match the demo VAD — the
  original 600 ms was raised by V2 because it cut real mid-phrase
  pauses; its docstring line is stale).
  Front-pad silence if the window starts before 0.
- t1 from the audit CSV (deterministic; VAD params frozen in
  tools/vad_audit.py). No-voiced clips (1 in train): file-end right-align
  fallback + flag in the manifest.
- Applies to every manifest clip incl. OOS speech; the synthesized OOS
  silence/noise lane is unaffected (generated at exactly 3.0 s).
- **Outliers: NOT removed.** The policy subsumes all 177 risk clips; only
  the 1 no-voiced clip is flagged. Long spans (10.3% of train > 2.5 s,
  max 9.0 s) keep tail priority (slot = last word).
- Pi: no VAD on the live path (capture timeout already anchors the window
  to speech end); the file-replay demo path MUST apply the same speech-end
  anchor (t1 from the holdout CSV).

## Dataset plan (sheet corpus, 2026-10-02) — SUPERSEDED by the final
## Corpus section below (group meeting: use ALL datasets; master Gold
Dataset is the source of truth). Kept for the record.
Sheet: `data/ai231_me2_dataset.xlsx` (10 sources). Verdicts:

| Source | Verdict | Keep / trim / remove |
|---|---|---|
| SLURP (real, CC BY-NC 4.0 audio) | KEEP — core real corpus | Trim: locations, slot values outside the 119 (open times/temps), non-mapping intents. Role: EVAL-REAL + phrasing mining. Note: ships slurp_real + slurp_synth dirs — don't double-count vs own TTS |
| SLURP-based TTS rows (10k + 9k design) | KEEP — synthetic training backbone | Fix (a) "20 labels" → reconcile to the 10-class lock first; (b) 5 phrases/label is thin → expand to the full 236-phrase phrasebook before mass TTS; (c) keep the speaker-disjoint split (24/3/3 incl. Filipino-English refs) |
| SynTTS-Commands Official (CosyVoice 2, MIT) | KEEP, 2 classes | cmd 1 = play_music (14k), cmd 8 = media_control (53k → CAP for class balance). Remove: 5 vendor wake words, 3 call words (unless make_call grows), "Set volume to 50%" (no volume slot) |
| jbuchner Synthetic Speech Commands (Kaggle) | KEEP a sliver | Keep single words stop/on/off (+up/down optional) → media_control/control_lights/dim steps. Remove: left/right/forward/backward/go/no. Single-word augmentation only |
| Fluent Speech (real, 97 spk) | KEEP, trimmed | Keep: set/turn temperature → set_temperature (Celsius 16–28 only), volume/brightness → media_control/dim_lights. Trim: location slots, out-of-vocab values, non-mapping intents. 97 speakers → speaker-disjoint EVAL-REAL slice |
| Timers and Such (real) | KEEP — only real timer/alarm audio | Trim to locked vocab: timer minutes → 20 values + 1–4 h; alarm → 48 half-hour marks (most drop); reminder topics → 10 locked (phrasings still mined); play intent genres → "any" |
| Snips SLU (real, 5,886 clips, MIT) | KEEP, trimmed | Keep: lights on/off (strip locations), brightness in-tens → dim_lights (mine phrasings), thermostat set-temp, get_time → ask_question. Remove: ALL color commands, fan speed, arbitrary values, locations |
| Common Voice (~200k, real) | KEEP — not as commands | (a) whisper-mine phrasings, esp. Filipino-accent; (b) background-noise/OOD augmentation. Don't train on unlabeled audio |
| MLEnd Spoken Numerals (real) | KEEP — auxiliary | Number-robustness eval + slot-tail augmentation. Trim numbers outside slot ranges |
| Google Drive file (Ubalde) | UNKNOWN | Needs a description before any verdict |

Cross-cutting: (1) slot whitelist (119 values) is a HARD filter on all real
audio; (2) speaker-disjoint splits across every source, manifests committed;
(3) per-class balance caps; (4) ADD a no_command/silence class (SLURP noise
rows + CV background) — fixes V2's live "silent fire" bug; (5) "no location"
rule applied to every source.

## Data policy (RATIFIED 2026-10-02 — Option B, boss: "optB"; still valid)
- **TRAIN = mixed real + synthetic** — as embodied in the master Gold
  Dataset's train split (10.7k clips across all sources).
- Hard rules (non-negotiable):
  - Speaker-disjoint splits per source; train ∩ eval = ∅, verified by
    manifest (speaker IDs) BEFORE training — V1's leak (0.33→0.98 fake
    swing) is the standing warning.
  - Slot whitelist = the 18 schema values (CONFIRMED complete in the
    master manifest) = hard filter for slot-head supervision.
  - Per-class balance caps (the manifest's `bucket` column already
    balances; verify per-class counts before launch).
- EVAL: test (4.42k) for selection reporting; **holdout (196) = the fixed
  demo benchmark, touched ONCE**. Verified shape: 13 fixed intents × 3
  phrase variations + 6 slotted intents × 9 rendered template×value
  phrases + 10 OUT_OF_SCOPE clips (94 phrase pairs, 5 speakers).
- Numerals split (66.4k MLEnd) stays separate (number-robustness bench).

## Corpus (FINAL — group meeting 2026-10-02: "use ALL collected +
generated datasets")
- **Master Gold Dataset** (Ma'am Ailene's collation — source of truth for
  the class and for Prof. Atienza's review):
  - HuggingFace: `airimonda/ai231-me2-voice-commands` — **81,686 clips /
    3.03 GB**; splits: train 10.7k / test 4.42k / holdout 196 /
    numerals 66.4k
  - Google Drive: `ai231-me2-gold-dataset` (master folder + master
    manifest file); DGX cluster access path TBA (SHARCHPC email pending)
  - Manifest columns: audio, file, transcript, command (19 intents),
    variation (v1–v3), slot_value, out_of_scope, bucket, speaker_id,
    source, is_synthetic, accent_group, numerals, duration_s,
    transcript_source, variation_match (exact/close), whisper_check,
    whisper_transcript, note
- **Sources inside (all in scope; licenses per the HF card):**
  | Source | License |
  |---|---|
  | SLURP | CC BY 4.0 |
  | Google Speech Commands v2 | CC BY 4.0 |
  | Common Voice 19 (en) | CC0 |
  | Fluent Speech Commands | Fluent public license (non-commercial academic) |
  | SNIPS SLU | see dataset terms |
  | Timers and Such | other-open |
  | MLEnd spoken numerals | see dataset terms |
  | Multi-Sensor Voice Command (Xela's VCM) | CC BY 4.0, DOI 10.48804/IEKKVZ (GDPR download-tracking applies) |
  | Group synthetic set (markandrian30/AI231 MEX2/Data; 100 speakers: 84 LibriSpeech + 16 SilencioPH) | group's own; refs CC BY 4.0 / SilencioPH terms |
  | Group recordings (students, Xela S1–S5) | group's own — do not share outside class without consent |
- **Upstream QA already applied:** whisper large-v3-turbo cross-check on
  all group recordings (accent mishearings kept + noted; else
  out_of_scope), 100 clips relabeled, 11 train clips = background noise
  (empty transcript), 749 low-similarity (<0.80) synthetic files flagged
  (Sir Mark's set), speaker-disjoint 80/10/10 inside Sir Mark's set.

## Master Gold Dataset — ingested + verified on HPC (2026-10-02)
- HPC repo `~/vcm_v3` (git, main). Dataset cache
  `~/vcm_v3/data/hf_cache` (6.2 GB); text-only manifests
  `~/vcm_v3/data/manifests/manifest_{train,test,holdout,numerals}.csv`;
  verification log `~/vcm_v3/runs/verify_gold_20261002_0408.log`
  (tool: tools/verify_gold_dataset.py; detail pass:
  tools/inspect_manifests.py).
- Splits (verified): train 10,682 / test 4,418 / holdout 196 /
  numerals 66,390 — total 81,686 = the HF card's count.
- **Speaker disjointness VERIFIED**: train 315 / test 121 / holdout 5
  speakers; train∩test = train∩holdout = test∩holdout = ∅.
- All 19 intents present in every split (20th `command` value =
  OUT_OF_SCOPE: train 201 / test 47 / holdout 10).
- Balance: fixed intents 333 train / 141 test per class; slotted
  999–1,048 train (111 per canonical phrase + real-data "(other slot
  value)" buckets: TIMER 37, ALARM 23, TEMPERATURE 49, BRIGHTNESS 33,
  CREATE_REMINDER 16) / 423 test per class.
- Slot values: exactly the schema's 18 (474 each in train+test);
  surface forms differ (alarm ":00", title case) — normalized at load.
- Durations: min 0.40 s / median 1.94 s / max 10.51 s — clips longer
  than 3.0 s are truncated at feature time (expected, documented).
- **20th class: RATIFIED 2026-10-02 (boss)** — OUT_OF_SCOPE is the 20th
  command class (201 train / 47 test / 10 holdout clips; class weight
  5.214, pre-registered in BENCHMARK.md). The 11 empty-transcript noise
  clips label as OUT_OF_SCOPE too.

## Live benchmark SOP (ratified 2026-10-02 — instruction 17:
github.com/airimonda/vcm-benchmark)
- The class's unified evaluation is now the boss's harness: laptop plays
  "wake word, command" (3 takes of our recorded "hey boots"; default 0.8 s gap),
  the Pi listens + logs, the laptop scores + samples telemetry; unattended after a
  4-step guided setup (~1 h, 206 trials full). Outputs report.md per the boss's
  6-section spec (19-intent + 93-command tables with acc + 95% CI, P/R/F1/F2,
  FAR/FRR/misfire; pipeline; slot exact/L1/relative/Metaphone/char; Pi telemetry
  incl. latency p50/p95/p99 + RTF + FLOPs via `--model`; top-10 confusions;
  per-intent P/R/F1/F2).
- Holdout source = the FROZEN 196 (harness downloads HF
  `airimonda/ai231-me2-voice-commands` split `holdout`) + 10 no-wake false-wake
  control trials. The one-shot demo benchmark is now live audio; BENCHMARK.md
  gates still apply to it.
- Pi contract (verified in `vcmbench/pi.py`): NEWEST
  `~/vcm_benchmark/<id>_<date-time>.log` (new file per assistant start); one line
  per decision carrying intent + slot + infer_ms + audio_ms (sound check BLOCKS
  without both); intent NAMES (our 19 == harness INTENTS verbatim; OUT_OF_SCOPE =
  reject); ≤1 command line per trial; wake lines feed wake detection rate.
- Slot print rule: the canonical `src/slots.py` strings. The harness PARSES slot
  text (TIMER s / ALARM min on a 24 h circle / degrees / percent / color word /
  text), so "6 AM" vs "6:00 AM" and "red" vs "Red" score exact-match.
- V3 demo must add: the benchmark log writer (audio_ms = 3000 — fixed 3.0 s
  window), a wake log line, beep-only mock mode for the run, and a measured
  `--wake-gap` (our wake-end→capture-start ≈ ≤1 s stride + beep + 200 ms warm-up
  > default 0.8 s).
- Post re-freeze (10-02): the harness's HF holdout download now returns the NEW
  202-clip revision (186 command + 16 OOS) — it scores exactly our re-frozen
  holdout; full run ≈ 212 trials (202 + 10 no-wake), not the old 206.

## Data source: class-shared JFS cache /data/ai231 (instruction 18, 2026-10-02)
- V3 trains from the shared JFS cache `/data/ai231` (= /mnt/jfs_hpc/data/ai231 —
  a `datasets` builder cache of HF `airimonda/ai231-me2-voice-commands`), NOT
  from the home-dir cache. Config: `default` only (the load.py pattern) —
  `supplemental_synth` / `synthetic_negatives` / `supplemental_fil` are outside
  the freeze and are never loaded.
- The JFS copy is a NEWER revision than our previous pull (HF re-push 2026-10-02
  10:53 PHT, rev 6947f130 vs our 10-01 23:29 PHT pull, rev 25111444):
  holdout 196→202 (OOS 10→16), train 10,682→10,733 (OOS 201→270),
  test 4,418→4,443 (OOS 47→76), numerals unchanged (66,390). New
  `group_synthetic_oos` source; SLURP/SNIPS/FluentSpeechCommands clips replaced
  in bulk; ~59 holdout clips re-taken; +11 real group-recording holdout clips.
- Consequence: our manifests (full-split exports of the old revision) + VAD
  audits (t1 cache) are one generation stale — 517 train / 226 test / 59 holdout
  manifest files do not resolve against the new arrow. Re-export + re-audit
  before training. Holdout re-freeze 196→202 is one-shot clean (no V3 model has
  seen either holdout); BENCHMARK.md to be updated — done, see next section.

## Re-freeze executed: dataset pinned to JFS rev 6947f130 (boss "yes go", 10-02)
- The freeze now pins dataset revision `6947f130` (`default` config, JFS
  `/data/ai231`): train 10,733 (OOS 270) / test 4,443 (OOS 76) / holdout
  202 (OOS 16) / numerals 66,390. Manifests re-exported (tools/
  export_manifests.py); VAD t1 re-audited in FULL with the same frozen VAD
  params (old audit kept at data/vad_audit/oldrev_25111444/). Committed on
  HPC 97fc4ba.
- Frozen class weights recomputed from the new train (w = nmax/n_c,
  nmax=1048): **OOS 5.214 → 3.881** (OOS grew 201→270), CALL 3.147 → 3.215
  (326), MESSAGE 3.147 → 3.255 (322); all other classes unchanged. Note: the
  anti-silent-fire loss weight dropped because the new revision adds 69
  `group_synthetic_oos` train clips; the OOS data lanes still exist and the
  post-first-run confidence-threshold pre-registration re-derives from the
  new OOS softmax curve.
- Target adjustments (mechanical — same floors, new denominators): OOS
  false-fire ≥ 68/76 (10% floor); holdout OOS rejection **≥ 13/16** (80%
  floor carried from 8/10 — boss may raise); holdout command clips stay 186.
  Demo bench is now 16 OOS, not 10.
- Run-1 generated OOS lanes: 135 silence + 135 noise-only (= real OOS count
  270), pre-registered in BENCHMARK.md.
- Numerals (report-only split) gets NO VAD audit: eval uses the file-end t1
  fallback. Acceptable — report-only, never part of the main score.
- MUSAN training bank = `data/noise16k_3s` (649 of 732 V2 bank clips, ≥3.0 s):
  the frozen waveform-domain SNR mix (src/features.py) requires full-length
  noise; the 83 shorter clips would broadcast-fail. V2 truncated in the MEL
  domain — different pipeline, no V2 change.
- HPC env split: conda `vcm` (torch 2.14+cu130, A100) runs train.py; repo
  `.venv` (numpy/pandas/pyarrow only) runs the data tools (export/audit/
  dataset smoke).

