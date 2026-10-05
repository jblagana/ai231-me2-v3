# ME2 V2 — Report To-Dos (from prof, 2026-10-02)

Most of this is **writing + packaging**, not code. Only A has real compute work.

## A. Code / compute work (the real gaps)
- [ ] **RPi validation run** — run inference on the actual Pi, record accuracy + latency *there* (not just the A100). This is the whole edge-deployment story and the biggest missing piece.
- [ ] **End-to-end latency benchmark** — time **end-of-speech → action fires** (not model-only). Capture per-run values, then report **mean + p95**. Warm + cold start.
- [ ] **Save checkpoints** — export `command_model` + `slot_model` to `.pt` (and `.onnx` for the Pi). Already exist in training output — just save + link them.
- [ ] **Reproducible runner** — one `run.sh` (or notebook): data → train → eval → latency benchmark, so a reviewer can clone + rerun.

## B. Writing / packaging (prose + one figure)
- [ ] **Dataset description** — # utterances, how recorded, who spoke, the 10 classes, slot distribution, train/val/test split, any augmentation.
- [ ] **Latency methodology** — *how* you timed it (timer placement), hardware, run count. The number is useless without this.
- [ ] **Model diagram** — one block diagram: mic → VAD → command model (frozen) → slot model → risk threshold → execute/clarify.
- [ ] **Slot 2nd-pass section** — describe the slot model as a separate inference pass, with its own latency + confidence.
- [ ] **GitHub repo** — make it public (or private + link), confirm checkpoints + runner are in it.

## C. Report body (assemble from the above)
- [ ] Dataset section
- [ ] Method + model diagram section
- [ ] Results: A100 **and** RPi (acc + latency tables)
- [ ] Reproducibility: repo link + checkpoint link + how-to-run

## Effort reality
- **A** = ~1–2 days of actual work (the Pi run is the long pole).
- **B** = a few hours of writing + one figure.
- **C** = assembly.
- Start today → **the RPi validation run is the only thing needing wall-clock time**; kick it off first, write everything else while it runs.

## RPi validation — what to measure
- **Direct ONNX inference is enough for the latency number.** Prof wants the *model* latency on the Pi, not a full audio-capture pipeline. Feed a pre-recorded audio file (or feature tensor) into the ONNX runtime on the Pi, time it, repeat N times → mean + p95.
- **Speaker/mic is NOT required** for the benchmark — that would add VAD + capture noise you don't control. Keep the measurement clean: input tensor in → action decision out.
- **Do include one full end-to-end pass** (mic/speaker → VAD → model → action) as a *separate, clearly-labeled* demo number, so the report shows both: (1) clean model latency, (2) real user-perceived latency. But the headline benchmark = direct ONNX inference.

## Evaluation set — what to run it on
- **Use your held-out test split** — the same one you used for the A100 numbers. It must be **disjoint from train/val** (no leakage) and **identical across A100 and RPi**, so the two accuracy/latency tables are apples-to-apples (same utterances, same ground truth).
- **Version it in the repo** — commit the exact file (e.g. `eval/test.json` + the audio, or a manifest pointing to it) so a reviewer reruns the *same* set. The number is only as good as the set behind it.
- **Report per-class, not just overall** — a 10-class confusion matrix + per-class accuracy/F1. Overall 0.95 can hide a class that's 0.6; the prof (and a reviewer) will look for that.
- **Slot model: eval on the slot values of the test-set utterances** (per predicted class), report top-1 accuracy + the margin distribution (that margin is what drives your clarify threshold — show its histogram so the threshold choice looks principled, not arbitrary).
- **Latency: average over the same test set** (or a fixed N-subset if the full set is slow on the Pi) — report mean + p95, warm + cold start.
