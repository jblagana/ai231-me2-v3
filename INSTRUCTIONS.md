# INSTRUCTIONS.md — AI 231 ME2 V3

Verbatim instruction log, newest first. Rule: log before any work;
Interpretation is agent-owned — a user edit is the new instruction (stop,
log it, follow it). Re-read before each log update. `done` = committed +
pushed (remote repo TBD).

## 2026-10-02 (cont.8) — GitHub push (public/replicable) + slides + epoch budget
Status: done locally (repo push-ready + slides built + epoch budget set to 100); push + post-training fill-ins remain
Progress: 100% of local prep
### Instruction 19 (verbatim)
> during training, can we set epoch to 100 and check for divergence between
> train and test set for model selection? is that sound? should we go 500 epoch
> -i need this project pushed in github for submission such that it can be
> replicated by anyone,
> -the images attached are the things required for presentation, make sure u log
> the needed details and build the slides after (or while waiting for training to finish)
### Findings
- Epochs: 100 (not 500) — selection is already TEST-based; 500 disproportionate
  (DECISIONS.md). scripts/train.sh + repro default = 100.
- Local folder was NOT a git repo (only HPC ~/vcm_v3). Fresh `git init` here.
- HF dataset facts pulled (default 81.8k; Xela VCM DOI 10.48804/IEKKVZ; per-
  source licenses; 4.38 GB / 102,994 rows / 4 configs).
- python-pptx 1.0.2 available locally -> built the deck locally.
### Actions
- Added: LICENSE, requirements.txt/-train/-pi, scripts/{fetch_data,train,smoke,
  build_slides}, tools/manifest_stats.py, tools/export_onnx.py, SUBMISSION.md,
  slides_data.json, slides/me2_v3_submission.pptx; rewrote README; hardened
  .gitignore. Committed locally.
- Deck: 3 slides (overview / To Be Submitted / appendix); known values filled,
  pending amber + logged (SUBMISSION.md section 6).
- Open: (a) confirm GitHub org/repo (default jblagana/ai231-me2-v3) + push;
  (b) run tools/manifest_stats.py on HPC -> fill Hours/Speakers; (c) post-
  training: results.json -> acc/loss, Pi benchmark -> latency/RTF, export_onnx
  -> weights URL; rebuild deck; (d) confidence threshold pre-reg from OOS stats.

## 2026-10-02 (cont.7) — HPC shared dataset /data/ai231 (JFS): use this, not our home cache
Status: done — re-freeze executed + committed on HPC (97fc4ba); dataset.py + train.py implemented + HPC smoke PASS; awaiting boss GO for the full training run
Progress: 90%
### Instruction 18 (verbatim)
> check this dataset in hpc /mnt/jfs_hpc/data/ai231/, we'll have to use that (is
> that the same as the huggingface one?) they say only use the load.py and not the
> golden_dataset file (why)
### Findings (verified on HPC 2026-10-02)
1. It IS the HF dataset, cached on the class-shared JFS: `/data` is a symlink to
   `/mnt/jfs_hpc/data`; `airimonda___ai231-me2-voice-commands/` is a `datasets`
   builder cache of `airimonda/ai231-me2-voice-commands` (cached by
   el.veena.grace.alfonso.rosero, Oct 2 16:34 PHT). 3 configs on disk: `default`
   (3.4G), `supplemental_synth` (303M), `synthetic_negatives` (96M). The arrow
   files are IPC STREAM format (read with `pa.ipc.open_stream`, NOT `open_file`).
2. NOT the same revision as our home cache — it is a newer re-push of the same
   repo:
   - OURS (pulled Oct 2 03:56 PHT; HF rev 25111444, push 10-01 23:29 PHT):
     train 10,682 / test 4,418 / holdout 196 / numerals 66,390; OOS 201/47/10.
   - JFS (HF rev 6947f130, push 10-02 10:53 PHT): train 10,733 / test 4,443 /
     holdout 202 / numerals 66,390 (identical); OOS 270/76/16.
   - Changes: new `group_synthetic_oos` source (+69/+29/+6 = the OOS delta);
     SLURP/SNIPS OOS clips replaced in bulk; ~59 holdout clips are different
     takes/files (t1↔t2 swaps + FluentSpeechCommands replacements); +11
     real_voice holdout (group recording 2026-09-30); group_synthetic command
     clips up in train/test.
   - HF HEAD is now further ahead (19:32 PHT: new config `supplemental_fil`,
     14,120 clips — NOT in the JFS cache; default splits unchanged at HEAD).
3. Our manifests (data/manifests/) are FULL-split exports of the OLD revision
   (10,682/4,418/196/66,390). Vs the JFS arrow: 517 train / 226 test / 59 holdout
   manifest files do not resolve; 0 missing from our old arrow.
4. Why load.py and not gold_dataset.py (boss said "golden_dataset"): load.py =
   `default` config only + health check (prints rows/schema/first example/backing
   files). gold_dataset.py ALSO loads `supplemental_synth` (5,856 "unused
   synthetic") + `synthetic_negatives` (1,250 generated OOS) — neither is in our
   freeze/manifests; a third config (supplemental_fil) now exists that neither
   touches. V3 trains on `default` only; gold_dataset.py is also just a scratch
   snippet (module-level, no error handling).
5. Benchmark impact: the harness has no revision pin → it will pull the NEW
   202-clip holdout. No V3 model has seen either holdout → re-freeze 196→202 is
   one-shot clean. BENCHMARK.md (196, OOS 10) is stale.
### Actions
1. dataset.py will point at `cache_dir=/data/ai231` (pin revision 6947f130...
   until re-freeze is confirmed), `default` config only, `pa.ipc.open_stream`.
2. Re-export manifests from the new revision; re-run VAD audit for the ~802
   changed files (517+226+59) or full re-audit; update BENCHMARK.md 196→202
   (+OOS 16).
3. Housekeeping: installed `pyarrow` into ~/.conda/envs/vcm (was missing;
   required for arrow streaming). `datasets` lib is NOT in vcm — not needed if
   we stream the raw arrow.

## 2026-10-02 (cont.6) — Live benchmark SOP (github.com/airimonda/vcm-benchmark)
Status: done — reviewed; Pi-package work items logged (post-ONNX)
Progress: 60%
### Instruction 17 (verbatim)
> https://github.com/airimonda/vcm-benchmark
> i created a benchmarking code para iisa tayo ng SOP. so, bale, isesetup nyo yung pi
> nyo, then ipupull nito ung holdout natin, then it will ask you to record your
> wakeword thrice, then will patch wakeword + command. para pwede nyo na syang iwan
> mag isa nagsasalita. tas ang laman ng report.md natin ay:
> 1. Header: student, date, wake word, number of commands, connection type, holdout source.
> 2. Classification table, one column per 19 intent and per 93 command: accuracy +
>    95% CI; balanced accuracy; precision, recall, F1, F2; false accept rate (OOS
>    clips where the Pi fired, with range and count e.g. 1/10); false reject rate
>    (real commands the Pi rejected or ignored); misfire rate (real commands where
>    the Pi fired the wrong command). Below: pipeline (response rate, no-response
>    count, extra fires, wake detection rate); unknown intent names (scored wrong);
>    real vs synthetic accuracy.
> 3. Slot values: one row per slotted intent + "all" row, over trials where the
>    intent was right: exact-match rate; mean absolute error in the slot's unit
>    (seconds, minutes, degrees, percent); mean relative error; phonetic distance;
>    character distance.
> 4. Raspberry Pi: specs (model, hostname, cores, CPU, top clock, RAM, OS, kernel,
>    Python, ML/audio packages); mean/p95/max of response latency (+p50/p99),
>    inference time + RTF (only if infer_ms/audio_ms are printed), CPU temp/clock/
>    load/throttling, CPU + RAM of the Pi and of the assistant process, CPU-seconds
>    per second of speech, assistant CPU share, test duration; with an ONNX model
>    given: params, size, FLOPs, effective GFLOP/s.
> 5. Top-10 "expected → got" confusions at intent and command level.
> 6. Per-intent (+REJECT) clips, precision, recall, F1, F2.
### Actions taken
1. Reviewed the harness end-to-end (clone kept at
   `third_party/vcm-benchmark-main/`): `benchmark.py` (laptop, guided 7 steps),
   `pi_agent.py` (Pi, stdlib-only: specs, 1 Hz CPU/temp/RAM/throttle samples, log
   tail, clock-sync pings), `vcmbench/` (schema, dataset, audio, Pi links, scoring,
   report).
2. What it is: the class's UNIFIED live-benchmark SOP — one script, same procedure,
   same data, unattended after a short guided setup.
   - Holdout: downloads OUR frozen 196 from HF
     (`airimonda/ai231-me2-voice-commands`, split `holdout`) + 10 false-wake control
     trials (in-scope commands played WITHOUT the wake word) = 206 full / 113 quick,
     shuffled by seed.
   - Wake: asks for the wake word (we type "hey boots"; default prompt "hey pi"),
     records 3 takes on the laptop mic (loudest burst, trimmed, -20 dBFS); trial =
     [0.3 s][wake take, rotated mod 3][gap, default 0.8 s][command, -20 dBFS][0.5 s],
     16 kHz mono.
   - Link: SSH (auto-discovered, password-less key; agent copied to the Pi, tails
     the NEWEST `~/vcm_benchmark/<id>_<date-time>.log`), or `--mode http` (Pi
     pushes), or manual.
   - Sound check: 2 un-scored warm-ups; BLOCKS until every command line has
     `infer_ms` + `audio_ms`; flags unknown intent names + class-number ordering.
   - Run: 10–15 s between trials, auto-reconnect + replay, `--resume`, ~1 h; outputs
     `report.md` (+ metrics.json, trials.csv, pi_metrics.csv);
     `--model [pi:]path.onnx` adds params/FLOPs/GFLOP/s.
3. Scoring (verified in `vcmbench/report.py`, `slots.py`, `schema.py`): 19-intent +
   93-command level; a 93-level prediction = (intent, slot) pair; REJECT = OOS or
   silence; FAR/FRR/misfire per column; false-wake rate from the 10 no-wake trials
   (not in the 19/93 score); wake detection rate from wake lines; extra fires =
   >1 line per trial. Slot strings are PARSED before comparing (TIMER→s,
   ALARM→min on a 24 h circle, °, %, color word, text): exact = equality after
   parse, abs = L1 in unit, rel = abs/span (50/900/8/80), phonetic = norm-
   Levenshtein of simplified-Metaphone of spelled values, char = norm-Levenshtein
   of spelled text.
4. Alignment with V3 (clean): our 19 class names == harness INTENTS verbatim (no
   aliases, no `--id-order` — we print names, not numbers); our 20th class
   OUT_OF_SCOPE == its REJECT; our `src/slots.py` strings parse to its canonical
   values, so "6 AM" vs schema "6:00 AM" and "red" vs "Red" score EXACT.
5. V3 Pi-package work items (after ONNX export):
   - Log writer (REQUIRED): append one line per decision, line-buffered, to
     `~/vcm_benchmark/<id>_<date-time>.log` (new file per start):
     `{"intent": "TIMER", "slot": "30 seconds", "infer_ms": 8.4, "audio_ms": 3000}`;
     OOS → `{"intent": "OUT_OF_SCOPE", "slot": ""}`. V2's demo prints to stdout with
     `cmd`/`e2e_ms` only — no log file, no `audio_ms` → sound check would block.
   - Wake line on gate fire (matches `wake word|wake detected|woke|hotword|wake=1|
     listening`) → wake detection rate.
   - `--wake-gap`: default 0.8 s < our wake-end→capture-start (≤1 s stride + ack
     beep + 200 ms warm-up); measure on the Pi and pass the right value.
   - Beep-only (mock) during the run; restart the assistant fresh before the run
     (harness reads the newest log).
6. Benchmark-day runbook: laptop clone + venv + `pip install -r requirements.txt`
   (numpy, sounddevice, soundfile, pandas, pyarrow; onnx optional); Pi password-less
   SSH (README has the exact key commands) + assistant running;
   `python benchmark.py --wake-word "hey boots" --student <name> --model
   pi:~/me2/me2_vcm_v3.onnx` → guided 4 steps → approve → unattended → report.md.

## 2026-10-02 (cont.5) — Slot read region 1.3 s → 2.0 s (live tail + slot value)
Status: done — committed on HPC
Progress: 55%
### Instruction 16 (verbatim)
> -can u show the actual phrases for the commands and then separate table
> for the slots for each parametric command (which commands are parametric
> btw?)
> -"The architecture assumes the same thing on purpose: the slot head reads
> only the last ~1.33 s of the window (max-pool over the last 8 cells). It
> was designed on the premise that the slot lives in the tail — which is
> exactly what the front-cut preserves." - with the 1s silence at the end
> of the clip, is 0.33 window left actually enough to read the slot? should
> we increase it to 1.5 and reduce the end of speech silence to 800ms? so
> theres actually 700ms useful clip to read the slot
### Actions taken
- Phrase + slot tables delivered (13 fixed × 3 variations; 6 parametric —
  TIMER, ALARM, TEMPERATURE, BRIGHTNESS, COLOR, CREATE_REMINDER — × 3
  templates × 3 slot values; all 18 templates end in {slot}).
- **Boss is right — the old read region was too short.** Probes
  (`tools/slot_tail_check.py`, `tools/slot_len_check.py`): (a) live capture
  ALWAYS has the 1.0 s silent tail and holdout slotted clips are 42.6%
  live-geometry → the old 1.28 s bcresnet region held only **0.28 s** of
  slot content; (b) V3 slot values are multi-word (0.3–1.2 s, measured
  from TTS spans). Digit+word heads (TIMER 10/30 s, TEMPERATURE 18/22/26,
  BRIGHTNESS 20/60/100 %) had only the shared noun in region — the
  distinguishing digits were outside it.
- **Amendment (pre-run-1): read region 1.3 s → 2.0 s** = 1.0 s live tail +
  full slot value. `BC_SLOT_TAIL_CELLS` 64→100, `SLOT_TAIL_CELLS` 8→12.
  Rejected the 1.5 s read + 0.8 s tail alternative: it breaks the frozen
  live match (demo silence_ms=1000) and still leaves 20-vs-60 percent
  ambiguous. Zero parameter change (30,134 @20c / 98,022 @20c / pinned
  96,861 @11c all verified unchanged), no demo/data change.
- Smoke exposed a latent V2 port bug: bcresnet's slot pooling maxed over
  (C, F) per time cell → (B, T_slice), which only matched Linear(32s) by
  the 64==64 coincidence (scale=2 × 64 cells); it would have crashed at
  100 cells. Fixed to per-channel max over (F, tail) → (B, 32s), same
  semantics as v2cnn (V2 file untouched; V3 slot heads train from scratch).
- Updated: src/models.py, DECISIONS.md, docs/architecture.html,
  tools/duration_check.py, JOURNAL.md. models smoke: **PASS**.

## 2026-10-02 (cont.4) — Phrase inventory: dataset phrases vs the 93-phrasebook
Status: done — committed on HPC
Progress: 55%
### Instruction 15 (verbatim)
> can u tabulate the unique phrases from the dataset clips?
### Actions taken
- `tools/phrase_inventory.py` → `data/phrase_inventory.csv`: **1,622
  unique (command, transcript) pairs**. The 93-phrasebook phrases are all
  present (as case/punctuation variants); the rest are free-form surface
  transcripts from real_voice + source datasets ("Dim lights to 50%",
  "Call Jane", "will it rain tomorrow"), incl. a few garbled Whisper
  strings ("Satan alarm at 6am.", "...80% 1.3 585 585").
- **No training impact:** the model consumes (features → command);
  phrase surface is the label's surface, and its diversity IS the
  robustness signal (same as V2's training data). Out-of-vocab slot
  surfaces ("7 AM", "50%") hit the frozen supervision gate →
  command-only. The 93-phrasebook remains the TTS-generation +
  holdout-variation template; the holdout additionally carries surface
  forms of its 3 variations (e.g. "Set the brightness to 20 percent") —
  benchmark is fixed, scored as-is.
- **OOS:** 265 unique rejection utterances (backchannels + near-miss
  commands like "turn the light down in the bedroom" — the hard-reject
  material). **Numerals split:** 32 unique digit words, 66,390 clips,
  ALL command=OUT_OF_SCOPE (report-only per pre-registration).

## 2026-10-02 (cont.3) — Case A/C follow-ups: tail silence + long-span check
Status: done — committed on HPC
Progress: 55%
### Instruction 14 (verbatim)
> for case a, why isnt there +1 silence? and for case c, how do we know
> that its a valid command whose slot is in the end?
### Actions taken
- **Case A (short clip) has no +1.0 s tail by construction of
  `end = min(d, t1 + 1.0)`:** the file ends before t1+1.0, so there is no
  clip silence after the speech to include; right-alignment pins the
  clip's last sample to the window's last sample, so the deficit is
  front-padded. Effective trailing silence = min(d − t1, 1.0 s) (TTS
  clips: their natural ~0.1–0.3 s).
  We do NOT force 1.0 s on every clip deliberately: the live capture is
  BIMODAL — 1.0 s (silence-stop) vs ~0 s (3.5 s cap, continuous
  speech). The training mix {short ≈ 0–0.3 s, long = 1.0 s} brackets
  the live distribution → the model classifies by content, not by tail
  amount.
- **Case C validity — three layers.** (1) Upstream: every clip carries a
  command + transcript that passed Whisper QA (whisper_check column);
  windowing only SLICES an already-labeled, QA'd clip — it never
  re-labels. (2) Slot-in-end = schema, not assumption: all 18 variable
  templates in src/commands.py end in `{slot}`, and the 18 values are
  disjoint across intents, so the tail alone identifies slot AND
  intent. (3) Architecture: the slot head reads only the last ~1.33 s
  (max-pool over the last 8 cells) — it assumes the tail by design.
- **New measurement (tools/long_span_check.py) — front cuts that
  actually happen (span > 3.0 s):** train 713/10,682 (6.7%): 299
  slotted (safe by construction) + 305 fixed + 109 OOS (OOS unaffected
  — any slice is still OOS); test 146 (3.3%): 32 slotted + 95 fixed +
  19 OOS; holdout 7: 3 slotted + 1 fixed (VOLUME_UP) + 3 OOS. Cut
  amount = span − 3.0 s. The fixed-intent watch group is exactly what
  the pre-registered TEST ">3.0 s vs ≤3.0 s" diagnostic measures each
  run; run-2 lever if it hurts.

## 2026-10-02 (cont.2) — Design Q&A: slot normalization + tail silence
Status: done — committed on HPC
Progress: 55%
### Instruction 13 (verbatim)
> -case-insensitive, `6:00 AM` → `6 AM` (how is this important?)
> -speech-end windowing (policy v2, `t1+0.5 s`) (why arent we using the
> 600ms used in actual demo vad?)
> -how are we trimming the long clips so it ends correctly? should we
> transcribe them so u can trim them right and clean our dataset? (whats
> our current protocol for short clips and long clips on how theyre
> handled and converted to mel?
### Actions taken
- **Q1 slots — decisive.** `tools/slot_forms.py` on all manifests: each
  intent has EXACTLY 3 raw forms, 0 unmatched; the only non-canonical
  form is ALARM's `6:00 AM/8:00 AM/9:00 PM` (schema: `6 AM/8 AM/9 PM`).
  Without casefold + `:00`-strip the ENTIRE ALARM slot head gets zero
  slot supervision — 1,440 clips (train 999 / test 423 / holdout 18) —
  an untrained head, per-head gate fails. Normalization is a
  SUPERVISION GATE: the 6 heads predict 18 discrete canonical classes,
  so the manifest string must map onto the class for that clip's slot
  loss to be valid; it also keeps the vocab locked at 18 (V1's 119-value
  open slot was a clean-slate flag).
- **Q2 tail — corrected, TAIL_S 0.5 → 1.0 s (pre-run-1 amend).** The
  shipping V2 demo is NOT 600 ms: `pi_demo.py capture_command
  (silence_ms=1000, max_ms=3500, min_rms=0.004)` — V2 raised it from
  600 because "600 cut real pauses" (mid-phrase); the 600 ms docstring
  line is stale. Live trailing silence = 1.0 s (silence-stop, the common
  case) or ~0 s (3.5 s cap, continuous speech). TAIL_S=1.0 matches the
  dominant live case exactly; speech-to-EOF clips (TTS) keep their
  natural short tails, matching the cap case. Amended: features.py,
  DECISIONS, BENCHMARK. t1 CSVs unaffected (window end computed at
  feature time).
- **Q3 trimming protocol — no transcription needed.** Window =
  [max(0, end−3.0 s), end], end = min(d, t1+1.0 s); segment right-
  aligned into 48000 samples (left-pad silence):
  - d ≤ 3.0 s (~86% train): whole speech + ≤1.0 s tail, front-padded
  - trailing-silence clips (17.9% holdout): [t1+1.0−3.0, t1+1.0] — the
    policy-v2 fix; speech fully inside
  - speech span > 3.0 s (10.3% train, max 9.0 s): keep the LAST 3.0 s —
    slot is the last word, so the decision-critical tail is always
    inside; lost leading context is bounded by the pre-registered
    >3 s vs ≤3 s TEST diagnostic
  - no-voiced (1 train clip): file-end fallback + manifest flag
  Trimming needs only the speech END (t1, ms-precision energy VAD,
  cached). Transcription/word timestamps would only affect the START
  cut of >3 s spans — already resolved by tail priority, and the slot
  word is provably inside (window end ≥ t1 ≥ end of last word). The
  corpus already carries Whisper QA (`whisper_check` column) for
  transcript validity.

## 2026-10-02 (cont.) — Status check + feature extractor implemented
Status: done — committed on HPC (251d42b)
Progress: 55%
### Instruction 12 (verbatim)
> what are we doing next again, are we cleaning the data?
### Actions taken
- Status answered: data cleaning/audit is COMPLETE (verification, duration
  audit, VAD anatomy audit + policy v2). Remaining data work = label
  normalization built into the loader — not a separate pass.
- Implemented `src/features.py` against the frozen spec:
  - `Mel` = verbatim port of V2's Pi-verified numpy mel; V2's SHIPPED
    buffers copied into `data/mel_buffers/` — one mel implementation
    serves HPC train/eval, Pi runtime, and the reused wake gate (parity
    by construction; verify_pi_v3 still re-checks on-device)
  - policy-v2 speech-end windowing: end = min(d, t1 + 0.5 s), left-pad
  - train augmentation, frozen order: waveform MUSAN SNR 5–25 dB → mel →
    tail-anchored time-warp 0.95–1.05 → gain ±10 dB (±0.2 normalized)
  - OOS lanes: `oos_silence` (exactly 0.0), `oos_noise_only` (no warp)
  - zero new deps (stdlib `wave` decode of HF arrow bytes; no
    soundfile/torchaudio in the venv)
- Verified: local + HPC self-test (shape (1,80,150) f32, silence=0,
  right-align, speech-end anchor, determinism, tail-anchored warp);
  `tools/feat_real_check.py` on the real corpus — the v1-bug clip
  (holdout, d=5.0 s, t1=1.46 s, tail 3.54 s): anchored window max 1.566
  vs file-end 0.406 (ambient-only floor — v1 would have labeled ambient
  noise as the command); train augment deterministic; OOS lanes OK.
- Next: `src/dataset.py` (loader + label normalization + supervision +
  frozen class weights) → `tools/train.py` → HPC smoke test (100–500
  clips) → full run 1 under screen.

## 2026-10-02 — Ratifications: primary arch + 20th class + duration audit
Status: done — committed on HPC (b29c6e3, de79bfd)
Progress: 45%
### Instruction 7 (verbatim)
> am i right that the slot head has another 6 heads? make bc resent our
> primary then ratify the 20class command. do both arch accept the same
> mel input? if yes, check if our dataset are all within 3s
### Actions taken
- Confirmed: one shared slot-feature tail → 6 per-intent 3-class Linear
  heads; each clip supervises only its intent's head. Confirmed: both
  arches consume the same locked (B,1,80,150) mel feature.
- Ratified: **bcresnet scale=2 = PRIMARY** (v2cnn → locked-budget
  fallback); **OUT_OF_SCOPE = 20th command class**.
- `src/commands.py` → 20 classes (93 phrases unchanged); @20c params
  verified: bcresnet 30,134 / v2cnn 98,022; local self-tests + smoke
  PASS.
- `BENCHMARK.md` pre-registered run-1 targets + frozen w=1/n_c weight
  vector (OOS 5.214); DECISIONS/README/JOURNAL/architecture.html all
  updated 19→20.
- Duration audit (`tools/duration_check.py`): NOT all clips ≤ 3.0 s —
  train 85.8% / test 88.1% / holdout 61.2% / numerals 99.7%.
  Truncation policy = next call (recommendation: right-align last 3.0 s).
### Instruction 8 (verbatim)
> well use wake word model already availbel in v2. how about feature
> proessing and engineering, any changes?
### Actions taken
- Clean-slate audit of the V1/V2 pipeline (full code: mel.py, train_v2.py,
  export_v2.py, train_wake.py, make_runb_weights.py) — 11 ranked flags +
  V3 deltas in DECISIONS.md "Clean-Slate Review"; deltas pre-registered in
  BENCHMARK.md (lexicographic TEST selection, OOS augmentation,
  waveform-domain noise, ±10 dB gain, SNR 20/10/5 reporting).
- Ratified: **reuse V2's wake bcresnet** — ckpt
  ~/ai231_me2_v2/runs/v2w/bcresnet/vcm_wake_bcresnet_best.pt (balanced
  0.9990, wake recall 0.94, no-wake FPR 0.0000, ONNX parity verified);
  no retraining needed (no_wake = any non-wake audio). Pi package = 2
  ONNXs sharing one numpy mel + buffers.
- Feature computation UNCHANGED (locked grid, V2-validated, shared by both
  ONNXs). Engineering changes: waveform-domain noise mix (mel on the fly,
  train base cache dropped), gain ±3→±10 dB, time-warp stays mel-domain
  after the noise, new OOS data path (MUSAN bank 732 clips + generated
  silence → OUT_OF_SCOPE). Truncation ratified: right-align / keep last
  3.0 s / left-pad (V2-validated rule).
### Instruction 9 (verbatim)
> -should we remake the data using the 40mel paper? and the complete
> methodology of the paper such as logmel instad of log10 etc
### Actions taken
- Answer: **no** — locked 80-mel / 3.0 s / 150f / 20 ms / log10 grid
  stays (full rationale: DECISIONS.md "Feature grid"). Nothing to remake
  (features computed, not recorded); wake reuse requires (1, 80, 150);
  1.0 s window kills the slot tail; 40-mel is lower-res for 20 close
  intents + 18 spoken numbers; ln vs log10 = constant 2.303× rescale.
- Pre-registered: grid change is NOT a run-2 lever (order: λ → v2cnn →
  augmentation/data); last-resort only if run-1 command gates fail.
- `docs/pipeline.html` converted to light-mode conference theme
  (white panels, print CSS, meta header) — committed.
### Instruction 10 (verbatim)
> whats the benfit of extenign the window from 3s
### Actions taken
- Answer: theoretical gains = fuller prefixes for the 14.2% of train
  (11.9% test, 38.8% holdout) clipped > 3.0 s + reduced train/holdout
  skew; but (1) ratified wake reuse is locked to (1,80,150) — any window
  change retires the wake gate, (2) Pi compute scales ~33%/extra second
  vs the ≤ 100 ms budget, (3) right-align only cuts leading filler
  (median 1.9 s; long clips = head padding, slot at tail preserved — V2
  proved 0.9990 on this scheme).
- BENCHMARK protocol: added per-run truncation diagnostic (test cmd acc
  > 3.0 s vs ≤ 3.0 s) so the window question is decided on run-1 data.
### Instruction 11 (verbatim)
> should we check and remove the outliers in the dataset? if we crop
> them at the tail, what if the 1-3s is the entire command while 3-10s
> is silence, hows that handled in training?
### Actions taken
- Checked: built tools/vad_audit.py (deterministic 10 ms RMS energy VAD;
  stdlib wave decode; reads audio inline from the HF arrow cache) and ran
  it on all splits (data/vad_audit/clip_vad_*.csv).
- Results: RISK (speech ends > 3.0 s before EOF → file-end right-align
  window = pure silence labeled command): train 0.6% (65), test 1.7% (77,
  all real_voice), holdout **17.9% (35, all real_voice — fixed 5.0 s
  group-recording capture)**. Boss's case confirmed real; v1 right-align
  never handled it (V2 corpus was uniform 3.0 s TTS).
- Decision: outliers NOT removed. Truncation policy v2 (FROZEN):
  right-align to VAD speech end — window ends at t1 + 0.5 s (matches the
  600 ms live-capture timeout), front-pad silence; file-end right-align
  only for no-voiced clips (1 in train, flagged).
- Docs updated: DECISIONS (policy v2 + audit table), BENCHMARK (feature
  bullet), pipeline.html (S2/B2), JOURNAL.

## 2026-10-02 — V3 kickoff: same project, new dataset
Status: in progress — workspace + taxonomy scaffolded
Progress: 10%
### Instruction 1 (verbatim)
> work here C:\Users\Jan\Desktop\AI 231 ME2 V3
> peek at C:\Users\Jan\Desktop\AI231 ME2 V2 workflow and summarize
### Instruction 2 (verbatim)
> well do the same project, but we will use the datasets here
> https://docs.google.com/spreadsheets/d/1VFm1-SAdNqtSwOeSHZct23tF6pPTntmj2940sugj61E/edit?gid=0#gid=0
> summarixze the dataset
### Instruction 3 (verbatim)
> look for the command labels in v2, and also tell me which how best to use
> these dataset (what to remove or keep or trim etc) based on the command labels
### Instruction 4 (verbatim)
> cant u find the commands in v2? theyre 10, ask time and ask whether are
> merged into ask question
### Instruction 5 (verbatim)
> ive also enumerated the slots to use in v2, as well as the model arch
### Instruction 6 (verbatim)
> 1. we use 10class
> 2. ignore
> 3. optB
>
> do we need all these dataset, i think theyre a lot
### Instruction 7 (verbatim)
> check this out for the final dataset and commands
> [file: c:\Users\Jan\Downloads\AI 231 MEx2 Meeting Summary.docx]
### Interpretation (agent)
- V3 = the same project as V2 (tiny on-device voice command model, RPi,
  ONNX, no ASR/LLM/cloud), rebuilt from scratch in THIS folder on the
  group's collated corpus.
- **SUPERSEDING UPDATE (group meeting summary, instruction 7 — final):**
  1. **Taxonomy = 19 intents** per Sir Mark's Dataset Schema (Option B):
     13 fixed + 6 variable (slotted). Supersedes the 10-class V2 port
     (instruction 4's "10class" was the V2-label mapping reference; the
     group's final schema governs). Schema sheet:
     docs.google.com/spreadsheets/d/1HZ7Q58S-konS5XWUE6FUGC-S-C8qAoSPaVbIP4NYSU0;
     machine-readable: github.com/markandrian30/AI231 MEX2/Data
     (labels.json / slots.json / README.md).
  2. **Corpus = master Gold Dataset** (Ma'am Ailene's collation of ALL
     collected + generated datasets — supersedes the 5-train/5-bench
     slimming: "Gagamitin ang lahat ng nalukom at na-generate na dataset
     para sa model training/validation/testing"):
     - HF: airimonda/ai231-me2-voice-commands (81,686 clips / 3.03 GB;
       train 10.7k / test 4.42k / holdout 196 / numerals 66.4k)
     - Drive: ai231-me2-gold-dataset (class + Prof. Atienza source of
       truth; master manifest) — DGX cluster path TBA (SHARCHPC email
       pending; SHARCHPC.ovpn in Downloads)
     - Sir Mark's set: 100 speakers (84 LibriSpeech + 16 SilencioPH
       Filipino-English), speaker-disjoint 80/10/10, clean + light-noise
       conditions, 18,600 → 17,851 files after 0.80-similarity filtering
     - Sources: SLURP, Google Speech Commands v2, Common Voice 19, Fluent
       Speech, SNIPS SLU, Timers and Such, MLEnd, Xela's VCM (Multi-Sensor
       Voice Command, CC BY 4.0, DOI 10.48804/IEKKVZ — the previously
       unknown Drive file is now IDENTIFIED and in scope), group synthetic
       (TTS from LibriSpeech + SilencioPH refs), group recordings
       (students + Xela S1–S5)
  3. **Fixed demo benchmark = the 3 phrase variations** per intent
     (Option B wording) — the holdout split (196 clips), one-shot.
  4. Model arch: V2 zoo unchanged (now 19 classes, 6 slot heads, 18
     values).
  5. Data policy Option B (instruction 6) remains consistent: mixed
     real+synthetic training, speaker-disjoint, zero overlap — the master
     manifest's splits ARE the ratified splits.
- src/commands.py + src/slots.py rebuilt on the 19-intent schema
  (93 phrases = 13×3 fixed + 6×3×3 variable; 18 locked slot values);
  tools/make_v3_commands.py deleted (10-class generator, superseded).
- Open: (a) slot vocab = schema's 18 vs manifest's ~93 slot-value classes
  (default: 18); (b) out_of_scope/no_command as a 20th class? (manifest
  has the flag + 11 noise clips — V2's silent-fire lesson); (c) DGX path.
- [ ] Pull master Gold Dataset to HPC (HF + Drive master manifest)
- [ ] Reconcile manifest splits vs Sir Mark's 80/10/10 speaker split
- [ ] Pre-register V3 targets (BENCHMARK.md), launch training
