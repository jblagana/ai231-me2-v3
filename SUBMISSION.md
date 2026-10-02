# SUBMISSION.md — AI 231 ME2 V3 (public + replicable)

The "To Be Submitted" checklist, one place, with the exact command or source
for every field. Machine-readable twin: `slides_data.json` (feeds
`scripts/build_slides.py`). Status legend: **done / pending / plan / in-progress**.

Rule (INSTRUCTIONS.md): log before work. `done` = committed + pushed.

## 1. GitHub repository — `jblagana/ai231-me2-v3` (public, MIT)
Status: **in progress** (repo is push-ready locally; needs org/repo confirm + push)
- [x] `LICENSE` (MIT)
- [x] Reproduction-first `README.md`
- [x] `requirements.txt` / `requirements-train.txt` / `requirements-pi.txt`
- [x] `scripts/` (fetch_data / train / smoke) — one-command reproduction
- [x] `.gitignore` (no 4 GB corpus, no checkpoints, no third-party drift)
- [x] Frozen split exports committed under `data/manifests/` (metadata only, no audio)
- [ ] Create the GitHub repo + push (see commands below). Confirm org (default
      `jblagana`) + repo name `ai231-me2-v3`.

Push (from this folder, after `git init` + commit are done):
```
git remote add origin https://github.com/jblagana/ai231-me2-v3.git
git branch -M main
git push -u origin main
```
(One-time: authenticate to GitHub on this machine first, e.g.
`git credential approve` via a token, or Git Credential Manager.)

## 2. Dataset location (licensed + citable DOI)
Status: **done (cite); verify Zenodo DOI on push**
- HF: `airimonda/ai231-me2-voice-commands` — `default` config (81.8 k rows;
  train 10.7k / test 4.44k / holdout 202 / numerals 66.4k). Zenodo DOI is
  auto-minted by HF (`10.57967/hf/...`) — read the exact DOI from the
  "Cite this dataset" block on the card.
- Drive master: `ai231-me2-gold-dataset` (master manifest; source of truth).
- Per-source licenses (each source keeps its own): SLURP CC BY 4.0 · Google SCv2
  CC BY 4.0 · Common Voice 19 CC0 · Fluent Speech Commands (non-commercial
  academic) · SNIPS SLU · Timers and Such · MLEnd · Multi-Sensor VCM (Xela)
  CC BY 4.0, **DOI 10.48804/IEKKVZ** · group synthetic (own; cloned from
  LibriSpeech CC BY 4.0 + SilencioPH refs) · group recordings (own; consent).
  "Use for research and education."
- Exact corpus stats (clips / hours / speakers / per-source) for the slide:
  ```
  .venv/bin/python tools/manifest_stats.py    # on HPC (has data/manifests/)
  ```

## 3. A100 cluster (node, GPUs, wall-clock, seeds)
Status: **pending (fills after run 1)**
- GPUs: **1× NVIDIA A100** (class HPC, conda `vcm` torch+cu).
- Node ID: `__fill from HPC (hostname / allocation) __`
- Wall-clock: `__fill from runs/v3r1/train.log final "wall Ns" line __`
- Seed: **20261002** (fixed). Epochs **100** (raised from 30 — DECISIONS.md).
- Reproduce: `bash scripts/train.sh` (or the `tools/train.py` command in README).

## 4. Model weights (release URL + licence)
Status: **pending (after training + export)**
- Export: `~/.conda/envs/vcm/bin/python tools/export_onnx.py --checkpoint runs/v3r1/checkpoint_best.pt --out models/me2_vcm_v3.onnx --verify`
- Options: (a) GitHub Release asset, or (b) HF repo. Pick one, paste URL here.
  The ONNX is ~120 KB (tiny) — committing it to the repo is also acceptable
  for one-command reproduction (currently gitignored under `models/`).
- Licence: **MIT** (this repo).

## 5. What the reviewer will look for
| # | Item | Status | Where |
|---|------|--------|-------|
| 1 | Repo public, one-command reproduction | in-progress | this repo + `scripts/train.sh` |
| 2 | Dataset licensed and citable (DOI) | **yes** | HF + per-source licenses; Xela 10.48804/IEKKVZ |
| 3 | Training logs + final checkpoint committed | pending | `runs/v3r1/` (results.json + checkpoint) |
| 4 | Pi 4 latency reproduced by the posted script | pending | Pi package + `third_party` vcm-benchmark |
| 5 | Held-out test set, unseen speakers | **yes** | holdout 202, speaker-disjoint, one-shot |
| 6 | Baseline of comparable size compared | plan | in-repo **v2cnn (98,022 p)** size baseline + V2 **v2a** (bcresnet 10c, 0.9990 EVAL-SYN) lineage |

## 6. Fill-in order (post-training)
1. `tools/manifest_stats.py` → dataset Hours/Speakers → `slides_data.json`.
2. `runs/v3r1/results.json` → Steps/loss, Keyword/intent acc, False-accept → `slides_data.json`.
3. Pi benchmark run → Latency p95 / RTF / Runtime threads → `slides_data.json`.
4. `tools/export_onnx.py --verify` → weights URL → `slides_data.json`.
5. `bash scripts/build_slides.py` → regenerate `slides/me2_v3_submission.pptx`.
6. Update this file's statuses; commit; push.
