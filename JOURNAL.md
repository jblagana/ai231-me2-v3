# JOURNAL.md — ME2 V3: the story so far, dated

## Phase 0 — Dataset inventory + scaffold (2026-10-02)

- Workspace: `C:\Users\Jan\Desktop\AI 231 ME2 V3` (new). V2 heritage repo is
  read-only at `C:\Users\Jan\Desktop\AI231 ME2 V2`.
- Group dataset sheet (Google Sheets, link-restricted → downloaded copy at
  `data/ai231_me2_dataset.xlsx`; inspected with `tools/inspect_dataset.py`):
  3 tabs — **Datasets** (13 entries, 6 contributors), **Benchmarks** (empty),
  **Device** (empty). 10 distinct sources (see DECISIONS.md table).
- Source contents verified 10-02 (web):
  - SynTTS-Commands-Official = 23-class EN media-control KWS set (CosyVoice 2,
    8,100 speakers, 111 h, MIT). Sheet row "14k cmd 1 + 53k cmd 8" maps to
    play_music + media_control under the 10-class numbering.
  - SLURP (EMNLP 2020): textual CC BY 4.0, audio CC BY-NC 4.0 (~6 GB, Zenodo),
    ships slurp_real + slurp_synth; speaker metadata encodes gender +
    native/non-native (usrid "FE-488" style) — useful for disjoint splits.
  - Snips SLU (HF): 5,886 clips, 575 MB, MIT, speaker demographics
    (age/gender/country), near/far-field sources; heavy brightness + color
    phrasing variety.
  - jbuchner Kaggle page is JS-only (not bot-readable) — verify exact 10
    single words on download.
- Taxonomy + arch locked (boss 10-02): V2's 10 classes + 119 slots / 7 heads
  + V2 model zoo. `src/commands.py` GENERATED from V2's phrasebook
  (`tools/make_v3_commands.py`, verbatim 236 phrases); `src/slots.py` +
  `src/models.py` carried over verbatim.
- V2 reference numbers (carried into BENCHMARK.md): V1 a2 = 0.889 EVAL-SYN
  (v1i 0.8618) / 0.2289 EVAL-REAL; v2a (bcresnet 10c, V2's deployed model) =
  0.9990 EVAL-SYN command, 0.5512 slot acc — the live weakness was
  confusable number slots (alarm/timer heads), not command recognition.

## Phase 1 — Scope ratification (2026-10-02)

- Boss ratified (instruction 6): (1) **10-class lock is final** (sheet's
  "20 labels" wording moot); (2) unnamed Drive file **IGNORED**; (3) data
  policy = **Option B** (mixed real+synthetic training, speaker-disjoint,
  manifest-verified zero overlap).
- Corpus slimmed (boss: "theyre a lot" — agent rec ratified): TRAIN on 5
  (SLURP real, SLURP-TTS, Timers & Such, Snips, MUSAN noise); the other 5
  sheet sources (SynTTS-Official, jbuchner, Fluent, Common Voice, MLEnd)
  move to the EVAL/BENCH track → fills the sheet's empty Benchmarks tab,
  keeps real-eval honest, preserves every contributor's work. Rationale in
  DECISIONS.md "Corpus scope".

## Phase 2 — Group meeting: final scope (2026-10-02)

- Meeting summary ingested (`data/meeting_summary_2026-10-02.txt`; links
  recovered via tools/docx_links.py). Final decisions:
  1. **19 intents** per Sir Mark's Dataset Schema (Option B) — 13 fixed +
     6 variable (18 slot values). Supersedes the 10-class V2 port.
  2. **ALL collected + generated datasets used** for train/val/test —
     supersedes the 5-train/5-bench slimming.
  3. **Master Gold Dataset** (Ma'am Ailene's collation): HF
     airimonda/ai231-me2-voice-commands (81,686 clips / 3.03 GB;
     train 10.7k / test 4.42k / holdout 196 / numerals 66.4k) + Drive
     ai231-me2-gold-dataset + master manifest; DGX path TBA (SHARCHPC
     email pending).
  4. **Fixed demo benchmark = the 3 phrase variations** per intent
     (Option B wording) — holdout split, 196 clips, one-shot.
- Source facts verified (web, 10-02): Sir Mark's set = 100 speakers
  (84 LibriSpeech + 16 SilencioPH Filipino-English), 80/10/10
  speaker-disjoint, clean + light-noise conditions, 18,600 → 17,851
  after 0.80-similarity filtering (749 flagged); HF manifest =
  whisper large-v3-turbo QA on group recordings, out_of_scope flag, ~93
  slot-value classes, per-source licenses; Xela's file = Multi-Sensor
  Voice Command (CC BY 4.0, DOI 10.48804/IEKKVZ).
- src/ rebuilt on the final schema: commands.py (93 phrases = 13×3 +
  6×3×3), slots.py (6 heads / 18 values), models.py unchanged (19c
  config). tools/make_v3_commands.py deleted (superseded).
- Open: slot vocab 18 vs ~93 (default 18); out_of_scope/no_command as a
  20th class (V2 silent-fire lesson); DGX access path.

## Phase 3 — HPC bootstrap + Gold Dataset verification (2026-10-02)

- `~/vcm_v3` created on n002 (git init, kickoff commit 6e3fd21); full V3
  workspace synced (docs, src, tools, meeting summary, dataset xlsx).
  Ops notes learned the hard way: (1) conda is not on the non-interactive
  PATH (init block in ~/.bashrc; base = /opt/miniconda3) → used a project
  venv (`~/vcm_v3/.venv` from me2's python 3.11, datasets 5.0.1) instead
  of touching the me2 training env; (2) Windows CRLF in the screen-run
  bash script killed it instantly → LF + `bash -n` before any screen
  launch; (3) PowerShell mangles `$()`/`$?`/`2>` inside double-quoted ssh
  strings → scripts go via local file + scp, never inline.
- Master Gold Dataset pulled (HF → data/hf_cache, 6.2 GB) and VERIFIED
  (log runs/verify_gold_20261002_0408.log): **VERDICT PASS**.
  - Splits: train 10,682 / test 4,418 / holdout 196 / numerals 66,390.
  - Speaker-disjoint: train 315 / test 121 / holdout 5 speakers; all
    pairwise overlaps ∅ (the V1 leak lesson, satisfied).
  - 19 intents present in every split; OUT_OF_SCOPE: 201/47/10.
  - Balance: 111 clips per canonical phrase in train (fixed 333/class;
    slotted 999–1,048 incl. 164 real "(other slot value)" clips); test
    141/423 per class.
  - Slot values: EXACTLY the schema's 18 (474 each in train+test) — the
    "~93 slot-value classes" open item is **CLOSED** (that figure was
    the bucket count). Surface deltas: alarm "6:00 AM" vs "6 AM", title
    case — normalize at load (src/slots.py).
  - Holdout shape (one-shot demo bench): 13 fixed × 3 variations +
    6 slotted × 9 rendered phrases + 10 OOS = 196.
  - Durations 0.40–10.51 s (median 1.94) — >3.0 s clips truncated at
    feature time.
- Remaining data call: `no_command`/`out_of_scope` as a 20th class vs
  noise treatment. DGX path still TBA.

## Phase 4 — Ratifications: bcresnet primary + 20th class (2026-10-02)

- Boss ratified: (1) **bcresnet scale=2 = PRIMARY** (V2a lineage; v2cnn
  becomes the locked-budget fallback). (2) **OUT_OF_SCOPE = 20th command
  class** (201 train / 47 test / 10 holdout; class weight 5.214).
- src/commands.py: CLASSES = 20 (93 phrases unchanged; OUT_OF_SCOPE has no
  canonical phrase — off-vocabulary by definition). @20c params: bcresnet
  30,134 / v2cnn 98,022 (verified locally).
- BENCHMARK.md: run-1 targets pre-registered (cmd ≥ 0.990 overall /
  0.980 macro on test; slot ≥ 0.850 overall / 0.750 worst head; OOS
  false-fire ≤ 10%; holdout one-shot ≥ 0.950 on 186 clips + ≥ 8/10 OOS;
  RPi ≤ 100 ms/clip). Frozen w = 1/n_c weight vector in the file.
- Combined loss pre-registered: **L = L_cmd + 1.0·L_slot** (both batch
  means; slot over valid slotted clips only, unweighted CE). λ=1.0 frozen
  for run 1 — first run-2 knob if slot gates miss with command passing.
- Duration audit vs the locked 3.0 s window (tools/duration_check.py,
  log: runs/duration_audit_20261002.log): **not all clips fit** —
  ≤ 3.0 s: train 85.8% (max 10.51 s) / test 88.1% (max 9.38) /
  holdout 61.2% (max 5.98; 73/76 long clips are real_voice capped at
  5.0 s) / numerals 99.7%. Truncation policy is the next call before
  feature extraction — recommendation: right-align (keep the LAST 3.0 s
  so the ~1.3 s slot tail always lands on the final spoken word; silence
  pad in front of short clips), same rule on every split.

## Phase 5 — Clean-slate review: V1/V2 audit (2026-10-02)

- Boss: "without regard for v1 and v2, what should be done in v3? flag any
  unreasonable components/method in the v1 and v2 pipeline." Audited the V2
  workspace code end-to-end (mel.py, train_v2.py, export_v2.py,
  train_wake.py, make_runb_weights.py, DECISIONS/BENCHMARK).
- 11 flags, ranked (DECISIONS.md "Clean-Slate Review"): no trained reject
  class (the silent-fire bug), 119-value open-numeric slot vocab, V1
  SYN-only selection (REAL 0.229), V1 train/val leak, v2f two-model
  pipeline, Run B error-matrix slot weights, mel-domain noise mixing, ±3 dB
  gain range, Pi naive resample, ad-hoc confidence threshold,
  RMS-timeout "VAD".
- V3 deltas (ratification pending): lexicographic TEST selection (cmd ≥
  0.990 → max slot); OOS augmentation (silence + noise labeled
  OUT_OF_SCOPE); frozen augmentation (time-warp 0.95–1.05, gain ±10 dB,
  waveform-domain MUSAN 5–25); λ stays 1.0; single two-head ONNX + Pi
  numpy-mel parity gate; wake word out of scope (PTT or always-on 20-class
  — boss call).
- Truncation policy settled by the audit: V2's validated right-align /
  keep-last-3.0-s / left-pad is exactly the recommended rule — carry over.
- Ratified (boss): **reuse V2's wake bcresnet** (ckpt + verified numbers
  on HPC; Pi = 2 ONNXs sharing one numpy mel). Feature computation stays
  locked/unchanged; engineering changes: waveform-domain noise mix, gain
  ±10 dB, mel on the fly (train base cache dropped), OOS silence/noise
  data path; SNR 20/10/5 robustness reporting on test (group protocol).
- Built `docs/pipeline.html` — full training pipeline (S1 data prep →
  S2 features (2 lanes) → S3 train → S4 lexicographic selection → S5
  validation/eval + target table → S6 export/package) + end-to-end
  on-device system (B1 capture → B2 numpy mel → B3 wake gate → B4
  capture window → B5 VCM → B6 decision gates → B7 actuation → B8
  feedback/UI + budget/invariants). All values verified against
  src/commands.py, src/slots.py, BENCHMARK.md. Converted to light-mode
  conference theme (print CSS) per boss request.
- Boss asked: remake data on the paper's 40-mel + paper methodology
  (log-mel)? — **No** (DECISIONS.md "Feature grid"): wake reuse requires
  (1,80,150); 1.0 s window kills the slot tail; 40-mel = lower freq
  resolution for 20 close intents + 18 spoken numbers; ln vs log10 =
  constant 2.303× rescale (no info difference); nothing to remake
  (features are computed, not recorded). Grid change pre-registered as
  NOT a run-2 lever — last resort only.
- Boss asked: benefit of extending the window beyond 3.0 s? — gains are
  small (longer prefixes for 14.2% of train / 38.8% of holdout) vs hard
  blockers (wake reuse locked to 150 frames, Pi compute +33%/s). Answered
  with a pre-registered truncation diagnostic in BENCHMARK (test cmd acc
  > 3.0 s vs ≤ 3.0 s per run) so the call rests on run-1 data.
- Boss asked: check/remove outliers + what if the command is at 0–3 s and
  3–10 s is silence? — built tools/vad_audit.py (energy VAD, deterministic)
  and ran it on all splits. RISK clips (speech ends >3.0 s before EOF →
  v1 window = silence labeled command): train 0.6% / test 1.7% / holdout
  **17.9%** (fixed 5.0 s real-voice capture). Outliers NOT removed;
  **truncation policy v2 FROZEN**: right-align to VAD speech end (window
  ends at t1+0.5 s, matches the 600 ms live-capture timeout); t1 cached in
  data/vad_audit/clip_vad_*.csv. Without the fix the one-shot holdout
  gate ≥ 0.950 was unreachable (ceiling ≈ 0.82).

## Phase 6 — Feature extractor implemented (2026-10-02)
- Boss asked: what's next, are we cleaning data? — answered: cleaning/audit
  is complete (verification, duration, VAD anatomy). Implementation starts:
  **`src/features.py` written and verified** (commit 251d42b).
- `Mel` is a verbatim port of V2's Pi-verified numpy mel, and V2's
  SHIPPED buffers were copied into `data/mel_buffers/` — one mel
  implementation now serves HPC training/eval, the Pi runtime, and the
  reused V2 wake gate (parity by construction; verify_pi_v3 re-checks
  on-device before the demo).
- Policy-v2 speech-end windowing implemented (end = min(d, t1 + 0.5 s),
  left-pad; t1=None → file-end fallback for the 1 no-voiced train clip).
- Train augmentation in the frozen order: waveform-domain MUSAN SNR 5–25 dB
  (732-clip bank, loaded in RAM) → mel → tail-anchored time-warp 0.95–1.05
  → gain ±10 dB (±0.2 in (x+5)/5 units). OOS lanes: generated silence
  (exactly 0.0) + noise-only (no time-warp).
- Zero new dependencies: stdlib `wave` decode of HF arrow bytes (the venv
  has no soundfile/torchaudio/librosa).
- Verified on real data (`tools/feat_real_check.py`): the v1-bug holdout
  clip (d=5.0 s, t1=1.46 s, 3.54 s trailing silence) — anchored window
  max 1.566 (command present) vs file-end 0.406 (ambient-only room
  floor; v1 would have trained ambient noise → command). Train augment
  deterministic under a fixed seed; OOS lanes correct. Local + HPC
  self-test pass (shape (1,80,150) f32, silence=0, right-align, anchor,
  warp tail-anchored).
- Boss Q&A (instruction 13): (a) slot normalization is DECISIVE —
  ALARM's manifest form is `6:00 AM` vs canonical `6 AM`; without
  casefold + `:00`-strip all 1,440 ALARM clips (999/423/18 across
  train/test/holdout) lose slot supervision → untrained head
  (tools/slot_forms.py: exactly 3 raw forms per intent, 0 unmatched).
  (b) TAIL_S amended 0.5 → 1.0 s pre-run-1: the shipping V2 demo is
  silence_ms=1000, not the stale-documented 600 (V2 raised it because
  600 cut mid-phrase pauses) — training now matches the live geometry
  exactly (features.py + DECISIONS + BENCHMARK amended). (c) Trimming
  needs only t1 (speech end); no transcription — full short/long-clip
  protocol in INSTRUCTIONS 13.
- Boss follow-up (instruction 14): Case A has no +1.0 s tail because
  end=min(d, t1+1.0) is capped by file length (deficit is front-padded);
  the live capture is bimodal (1.0 s silence-stop vs ~0 s cap) so the
  training mix {0–0.3, 1.0} deliberately brackets it. Case C: validity
  is upstream (label + Whisper QA — windowing only slices, never
  re-labels); slot-in-end is the schema (all 18 templates end in
  {slot}; 18 values disjoint across intents); the slot head reads only
  the last 1.33 s. Measured (tools/long_span_check.py): front cuts hit
  span>3.0 s clips — train 713 (299 slotted safe / 305 fixed / 109 OOS),
  test 146, holdout 7 — the fixed group is what the pre-registered
  >3 s diagnostic watches.
- Phrase inventory (instruction 15, tools/phrase_inventory.py →
  data/phrase_inventory.csv): 1,622 unique (command, transcript) pairs —
  all 93 phrasebook phrases present (case/punct variants) + ~1.3k
  free-form surface transcripts (real_voice + source datasets);
  off-book surfaces are harmless (QA sits on the command/slot_value
  layer; out-of-vocab slots hit the command-only gate). OOS = 265 unique
  rejection utterances incl. near-miss commands; numerals split = 32
  unique digit words, 66,390 clips, all OOS (report-only).
- Slot read region 1.3 s → 2.0 s (instruction 16, pre-run-1 amendment):
  boss asked whether the ~0.33 s of slot content left inside the slot
  head's read region (after the live 1.0 s silent tail) is enough. It is
  NOT: probes (tools/slot_tail_check.py, tools/slot_len_check.py) show
  (a) holdout slotted clips are 42.6% live-geometry (tail ≥0.95 s) and
  live capture is ALWAYS 1.0 s tail → only 0.28 s of slot content in the
  old 1.28 s bcresnet read region; (b) V3 slot values are multi-word
  phrases (0.3–1.2 s; measured from TTS spans: "blue" ≈0.35 s, "9 PM"
  ≈0.5 s, "20 percent" ≈0.85 s, "one hundred percent"/"twenty-two
  degrees" up to ≈1.2 s). Under the old region, the digit+word heads
  (TIMER 10/30 s, TEMPERATURE 18/22/26 °, BRIGHTNESS 20/60/100 %) had
  only the shared noun ("seconds"/"degrees"/"percent") in region — the
  distinguishing digits fell OUTSIDE it. Fix: read region widened to
  2.0 s = 1.0 s live tail + 1.0 s (full slot value): bcresnet
  BC_SLOT_TAIL_CELLS 64→100, v2cnn SLOT_TAIL_CELLS 8→12. Rejected boss's
  alternative (1.5 s read + 0.8 s training tail): it would break the
  just-frozen live match (demo silence_ms=1000) and still leaves
  20-vs-60 percent ambiguous. Zero parameter change (30,134/98,022
  unchanged, pinned budget passes), no demo/data change. The smoke run
  exposed a latent V2 port bug: bcresnet's slot pooling maxed over (C, F)
  per time cell -> (B, T_slice), which only fed Linear(32s) by the
  64==64 coincidence (scale=2, 64 cells) and would have crashed at 100
  cells; fixed to per-channel max over (F, tail) -> (B, 32s), same
  semantics as v2cnn (V2 file untouched — its models are frozen/deployed;
  V3 slot heads train from scratch). Smoke: PASS.
- Next: `src/dataset.py` (loader + label normalization + supervision +
  frozen class weights) → `tools/train.py` → HPC smoke test (100–500
  clips) → run 1 under screen.

## 2026-10-02 (later) — Live benchmark SOP published (instruction 17)
- Boss released `github.com/airimonda/vcm-benchmark` — the class's unified
  unattended live benchmark (one guided laptop script + stdlib-only Pi agent).
  Reviewed end-to-end; clone kept at `third_party/vcm-benchmark-main/`.
- It pulls OUR frozen 196-clip holdout from HF (same repo/split), patches 3 takes
  of our recorded "hey boots" in front of each command (default gap 0.8 s), adds
  10 no-wake false-wake trials, and scores 19 intents / 93 (intent, slot) commands
  / slot distances / Pi telemetry into the 6-section report.md the boss listed.
- Alignment clean: our 19 class names == harness INTENTS verbatim (no aliases /
  id_order); our 20th class = its REJECT; our canonical slot strings parse to its
  canonical values (exact-match safe across the "6 AM"/"6:00 AM" surface delta).
- V3 Pi-package gaps (post-ONNX): log writer to
  `~/vcm_benchmark/<id>_<ts>.log` = `{"intent","slot","infer_ms","audio_ms"}`
  (sound check BLOCKS without both — V2's e2e_ms is not enough), wake log line,
  audio_ms = 3000, beep-only during the run, measured `--wake-gap` (our
  wake→listen latency > default 0.8 s).
- SOP day: laptop venv + `pip install -r requirements.txt`; Pi password-less SSH;
  `python benchmark.py --wake-word "hey boots" --model pi:~/me2/me2_vcm_v3.onnx`
  → guided setup → approve → unattended ~1 h → report.md.

## 2026-10-02 (later) — HPC shared dataset: newer HF revision (instruction 18)
- Boss: use `/mnt/jfs_hpc/data/ai231` (class-shared JFS). It is the HF dataset
  `airimonda/ai231-me2-voice-commands` cached by a classmate (el) — `default`
  config + 2 extra configs. "Use load.py, not gold_dataset.py" = `default`
  config only (the extras are out-of-freeze); load.py doubles as the health
  check script.
- It is a NEWER revision than our pull (HF re-push 10-02 10:53 PHT): holdout
  196→202, train 10,682→10,733, test 4,418→4,443, numerals identical; OOS now
  270/76/16 via new `group_synthetic_oos` + SLURP/SNIPS/FSC replacements; ~59
  holdout clips re-taken (+11 real group recordings from 2026-09-30).
- Our manifests are full exports of the OLD revision; 517 train / 226 test / 59
  holdout files won't resolve against the new arrow. Arrow files are IPC STREAM
  format (`pa.ipc.open_stream`, not `open_file`).
- Next: re-export manifests, re-run VAD audit on the ~802 changed files,
  re-freeze holdout 196→202 (one-shot clean), update BENCHMARK.md; dataset.py →
  /data/ai231, default config, revision-pinned.
- Housekeeping: installed pyarrow in the vcm env (was missing; needed for arrow
  streaming).

## 2026-10-02 (later) — Re-freeze executed + dataset/train harness (boss "yes go")
- Re-freeze to JFS rev 6947f130 executed: 4 manifests re-exported (tools/
  export_manifests.py; verified 10,733 / 4,443 / 202 / 66,390, OOS 270/76/16);
  FULL VAD re-audit on all three splits (trailing silence ≥3.0 s: train 0.6% /
  test 1.7% / holdout 23.3% — old 17.9% stale; old CSVs kept in
  data/vad_audit/oldrev_25111444/).
- Frozen class weights recomputed (w = nmax/n_c, nmax=1048): OOS 5.214→3.881
  (270 OOS train), CALL 3.215 (326), MESSAGE 3.255 (322); rest unchanged. The
  158 slot-NaN train clips persist (command-only — matches pre-registered note).
- BENCHMARK.md updated (re-frozen corpus line, VAD stats, weights table, OOS
  false-fire ≥68/76, holdout OOS ≥13/16 = 80% floor carried from 8/10, demo
  bench 16 OOS, gen-OOS lanes 135+135 pre-registered).
- src/dataset.py implemented + HPC smoke PASS: holdout exactly 202 (186 + 16
  OOS), train 10,733 + gen lanes, 0 vad-missing / 1 no-voiced, augmentation
  deterministic under fixed seed, frozen weights vector OK.
- tools/train.py implemented + HPC smoke PASS (bcresnet scale=2, 30,134 p,
  A100, conda vcm torch 2.14+cu130): (B,1,80,150)→(B,20) cmd + 6×(B,3) slot
  heads, slot-loss masking, OOS labels, no NaN losses, GUARD-FAILED fallback
  path exercised; per-epoch TEST metrics incl. SNR 20/10/5 (seeded deterministic
  MUSAN, cached), >3.0 s diagnostic, OOS softmax stats; holdout ONE-SHOT flag
  file; numerals report-only (file-end t1 fallback — no numerals VAD audit).
- MUSAN bank filtered to ≥3.0 s clips (data/noise16k_3s, 649 of 732): the
  frozen waveform-domain SNR mix needs full-length noise (V2 truncated in the
  MEL domain — different pipeline).
- Committed on HPC 97fc4ba (manifests + BENCHMARK.md + dataset.py + train.py
  + vad_audit patch + export_manifests.py + .gitignore).
- Post-hoc stat on the 1.33→2.0 s slot read region (re-frozen corpus, slot
  clips 5,994 / 2,538 / 108 train/test/holdout): OLD region = median 0% of
  the slot value visible (61-65% of slot clips had NO slot audio in the read
  region; V3 median voiced span is 1.5 s, so the old 1.67-3.0 s region mostly
  read silence); NEW region = median 75-77% visible, ~36% fully inside, ~10%
  still invisible (sub-1 s commands), 52% partial (first sliver of the first
  slot word clipped). NEW FLAG from the same pass: ~6% of command clips
  (624 train / 241 test / 9 holdout; 88% of the train ones = group_synthetic
  source) have implausibly short VAD spans (e.g. a 6-word phrase "voiced"
  for 0.03 s) — their 3.0 s windows start at the wrong offset (misaligned
  training samples, command + slot heads both affected). Recommend a quick
  QA pass (spot-listen / targeted VAD re-check on just those clips) before
  run 1.
- Next: full training run under boss GO (`~/.conda/envs/vcm/bin/python
  tools/train.py --run v3r1`, 30 epochs, under screen); then confidence-
  threshold pre-registration from the logged OOS softmax stats; then ONNX
  export + Pi package.

## Phase 3 — Submission packaging + epoch budget (2026-10-02 cont.)
- Boss asked: (1) run 100 (or 500) epochs to check train/TEST divergence for
  model selection? (2) push the repo to GitHub (public, replicable); (3) log the
  needed details + build the slides (two template images provided).
- Epochs: **100, not 500.** The frozen selection rule is already TEST-based
  (max slot acc among TEST cmd acc >= 0.990), so 100 just gives the selector more
  candidates + a visible divergence curve; 500 is disproportionate for a 30 k-param
  model on 10.7 k clips (~5x wall-clock; cosine baked to total steps). DECISIONS.
  `scripts/train.sh` + repro default = 100.
- Repo made push-ready for GitHub: LICENSE (MIT), repro README, requirements x3,
  scripts (fetch/train/smoke/build_slides), tools/manifest_stats.py, tools/
  export_onnx.py, .gitignore hardened, SUBMISSION.md tracker. Local folder is a
  fresh `git init` + commit (the HPC repo is separate; remote TBD until org/repo
  is confirmed — default jblagana/ai231-me2-v3).
- Slides built: slides_data.json (all fields + status) + scripts/build_slides.py
  -> slides/me2_v3_submission.pptx (3 slides: overview / To Be Submitted /
  appendix). Known values filled; pending (hours/speakers, acc, latency, weights
  URL) are amber + logged with their exact source in SUBMISSION.md section 6.
- Architecture note for the deck: fixed-window BC-ResNet KWS (2 ONNX), NOT
  streaming / KV-cache — the template's "causal encoder" box is replaced.
- HF dataset confirmed: `default` 81.8k rows (train 10.7k / test 4.44k /
  holdout 202 / numerals 66.4k), per-source licenses (Xela VCM DOI
  10.48804/IEKKVZ), 4.38 GB / 102,994 rows across 4 configs.

## Now / next
- [x] Re-export manifests from new JFS revision (6947f130) + re-run VAD audit
      (full; done 10-02)
- [x] Re-freeze holdout 196→202 + update BENCHMARK.md (one-shot clean; done
      10-02, HPC 97fc4ba)
- [x] Live benchmark SOP reviewed + logged (instruction 17; harness cloned to
      third_party/vcm-benchmark-main/)
- [ ] V3 Pi package: benchmark log writer (~/vcm_benchmark/, infer_ms + audio_ms)
- [x] Pull master Gold Dataset to HPC (HF; verified 10-02)
- [x] Verify manifest splits: speaker-disjoint, zero train/eval overlap
- [x] Ratify the 20th-class call → OUT_OF_SCOPE as 20th command class
- [x] Pre-register V3 targets in BENCHMARK.md (holdout one-shot rule)
- [x] Duration audit + truncation policy — refined to speech-end anchor
      (VAD audit: 17.9% of holdout had ≥ 3.0 s trailing silence; policy v2
      frozen, t1 cached in data/vad_audit/)
- [ ] Ratify the clean-slate V3 deltas (DECISIONS.md)
- [x] Capture policy: reuse V2's wake bcresnet (Pi = 2 ONNX, shared mel)
- [x] Feature grid: KEEP locked 80-mel / 3.0 s / 150f / 20 ms / log10
      (paper's 40-mel grid rejected — DECISIONS.md)
- [ ] Fetch Drive master manifest (`ai231-me2-gold-dataset`)
- [x] Feature extraction script (src/features.py — mel port, policy-v2
      windowing, frozen augmentation, OOS lanes; verified 2026-10-02)
- [x] Dataset loader (src/dataset.py — normalization + supervision + weights;
      HPC smoke PASS 2026-10-02)
- [x] Training script (tools/train.py — bcresnet, 20c, 6 slot heads; HPC
      smoke PASS 2026-10-02)
- [ ] Launch training (HPC now; DGX when path confirmed)
