# AI 231 ME2 V3 — tiny on-device voice-command assistant (always-on keyword + intent)

A **no-cloud** voice assistant for the **Raspberry Pi 4**: an always-on
wake/keyword detector + a tiny command + slot classifier, both tiny
**BC-ResNet** models that share one fixed **80-mel / 3.0 s** feature and run
on **ONNX Runtime (CPU)**. No ASR, no LLM, no cloud round-trip.

- **20 command classes** = 19 intents (13 fixed + 6 variable/slotted) + `OUT_OF_SCOPE`
- **6 slot heads** × 18 whitelisted values (BRIGHTNESS / COLOR / TEMPERATURE /
  TIMER / ALARM / CREATE_REMINDER)
- Primary model: **BC-ResNet scale=2, ~30 k params, ~120 KB ONNX**
- Trained on the group **master Gold Dataset** (HuggingFace
  `airimonda/ai231-me2-voice-commands`, `default` config, 81.8 k clips)

> This is the V3 rebuild of the V2 project on the group's collated corpus.
> Group-meeting scope (19 intents, Option B) is in
> [`data/meeting_summary_2026-10-02.txt`](data/meeting_summary_2026-10-02.txt).

## Quick links (the working docs)
| File | What |
|---|---|
| [`INSTRUCTIONS.md`](INSTRUCTIONS.md) | verbatim instruction log (newest first) |
| [`DECISIONS.md`](DECISIONS.md) | ratified decisions + rationale |
| [`JOURNAL.md`](JOURNAL.md) | dated story |
| [`BENCHMARK.md`](BENCHMARK.md) | pre-registered targets + canonical run numbers |
| [`SUBMISSION.md`](SUBMISSION.md) | public/submission tracker (repo, DOI, weights, checklist) |
| [`slides_data.json`](slides_data.json) | single source of truth for the deck |

## Repo map
```
src/        commands.py (20 classes / 93 phrases) · slots.py (18 values / 6 heads)
            features.py (80-mel/3.0s + frozen aug + OOS lanes) · models.py (bcresnet/v2cnn)
            dataset.py (manifests + arrow + VAD t1 -> samples)
tools/      train.py (train+eval, frozen selection) · export_onnx.py · vad_audit.py
            export_manifests.py · manifest_stats.py · (audit/QA helpers)
scripts/    fetch_data.py · train.sh · smoke.sh · build_slides.py
data/       manifests/ (frozen split exports, metadata only) · mel_buffers · dataset xlsx
runs/       <run>/ (train.log, results.json, checkpoints, caches)  — gitignored
third_party/ vcm-benchmark (clone separately — see below)          — gitignored
```

## 1. Install
```bash
git clone https://github.com/jblagana/ai231-me2-v3 && cd ai231-me2-v3

# data tools (manifests / VAD / stats):
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# training (A100) — on the class HPC use the conda `vcm` env (torch + cu);
# elsewhere:  pip install -r requirements-train.txt

# Raspberry Pi 4B inference:  pip install -r requirements-pi.txt
```

## 2. Data (frozen)
V3 trains on the **`default` config only** of
[`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands):
train 10,733 · test 4,443 · holdout 202 · numerals 66,390 (re-frozen JFS rev
`6947f130`). `supplemental_*` / `synthetic_negatives` are **out of the freeze**.

- Class HPC: data is at the shared JFS path `/data/ai231` (already there).
- Anywhere else: materialize the HF cache once, then point `--arrow-dir` at it:
  ```bash
  pip install -U datasets
  python scripts/fetch_data.py      # -> arrow dir: .../default/0.0.0
  ```
- The frozen split exports (metadata, no audio) are committed at `data/manifests/`.
- Each audio source keeps its own license (CC BY 4.0 / CC0 / non-commercial
  academic / own); see [`SUBMISSION.md`](SUBMISSION.md) §2. Research + education use.

## 3. Train (one command)
The frozen protocol is in [`BENCHMARK.md`](BENCHMARK.md). On the A100 node:
```bash
screen -S vcm_v3          # run under screen; the job is ~25-40 h at 100 epochs
bash scripts/train.sh     # == tools/train.py --run v3r1 --arch bcresnet --epochs 100 ...
```
Selection picks the best **TEST** epoch (max slot-acc among TEST cmd-acc ≥
0.990), evaluates **holdout exactly once** (one-shot), and writes
`runs/v3r1/{train.log, results.json, checkpoint_best.pt, per-class + slot-confusion CSVs}`.

Fast pipeline smoke (no full training, ~1-2 min) on any box:
```bash
bash scripts/smoke.sh
```

## 4. Export ONNX + Pi package
After training (on the box with torch + onnxruntime):
```bash
~/.conda/envs/vcm/bin/python tools/export_onnx.py \
  --checkpoint runs/v3r1/checkpoint_best.pt \
  --out models/me2_vcm_v3.onnx --verify
```
→ opset-17 ONNX, input `mel (1,80,150)`, outputs `command_logits (20)` + 6
`slot_* (3)`. The Pi package = this ONNX + the (V2) wake ONNX + shared
`mel_window.npy` / `mel_fb.npy` + a metadata JSON + a parity script.

## 5. Live benchmark (vcm-benchmark)
The class live harness is a separate public repo (pinned):
```bash
cd third_party && git clone https://github.com/airimonda/vcm-benchmark.git
```
The Pi assistant must write one JSON line per command to
`~/vcm_benchmark/<id>_<date-time>.log` with `intent`, `slot`, `infer_ms`,
`audio_ms` (= 3000). See the harness README for the exact contract.

## 6. Results
Pre-registered targets + any run numbers live in
[`BENCHMARK.md`](BENCHMARK.md); the run writes `runs/v3r1/results.json`.

## 7. Reproduce the deck
```bash
pip install python-pptx
python scripts/build_slides.py      # -> slides/me2_v3_submission.pptx
```
Fill the `pending` fields in `slides_data.json` (see `SUBMISSION.md` §6), then rebuild.

## License
MIT — see [`LICENSE`](LICENSE). The dataset is separate and governed by its own
per-source licenses (research + education).

## Citation
```
J. Lagana et al., "AI 231 ME2 V3: tiny on-device voice-command model",
UP EEE AI231, 2026. Code: https://github.com/jblagana/ai231-me2-v3
Dataset: https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands
```

