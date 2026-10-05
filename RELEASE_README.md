# ME2 V3 — Separately-Released Artifacts

**Voice-Controlled Smart Device (AI 231 ME2 V3)** · J. Lagana + AI231 ME2 group (UP EEE)
2 October 2026 · Model · Dataset · Training on A100 · Validation on RPi 4

## What is in this archive
The public repo intentionally excludes weights and training outputs
(`.gitignore`: `runs/`, `*.pt`, `*.onnx`). This archive carries exactly those
"released separately" items:

| Path | Artifact | Size |
|---|---|---|
| `me2_vcm_v3.onnx` | **Final model** — BC-ResNet (scale 2), 30,134 params, opset 17; 20-class intent head (19 intents + OUT_OF_SCOPE) + 6 slot heads × 3 values; shared 80-mel / 3.0 s window | ~172 KB |
| `runs/v3r1/results.json` | **Training logs** — final metrics + model selection | KB |
| `runs/v3r1/train.log` | Full per-epoch training log | KB |
| `runs/v3r1/launch.out` | Raw launch stdout (host / GPU / command / wall-clock) | KB |
| `runs/v3r1/per_class_test.csv` | Per-intent P / R / F1 / F2 (test set) | KB |
| `runs/v3r1/slot_confusion_test.csv` | Slot-value confusion matrix (test) | KB |
| `runs/v3r1/checkpoint_best.pt` | Best PyTorch checkpoint (ONNX source) — optional | ~120 KB |

## Headline results (from `runs/v3r1/results.json`)
- Command accuracy: **0.9455** holdout (186 clips, one-shot, unseen speakers) / 0.9293 test
- Slot accuracy: **0.9722** holdout / 0.9748 test
- OUT_OF_SCOPE reject: 11/16 (0.688) holdout / 34/76 (0.447) test — known weak spot, gated in deployment by the wake word + a 0.50 confidence gate
- Selection: best-command **epoch 142** · final training loss **0.206** (cmd 0.179 + slot 0.026)
- On-device (Pi 4B, ONNX Runtime CPU): **92 ms** p95 per 3 s clip (RTF 0.030)

## How it was trained (provenance)
- Cluster: 1× **NVIDIA A100**, COE HPC node **n003**, conda `vcm` (torch + CUDA)
- Budget: **200 epochs × 336 steps = 67,200 steps**, ~47 min wall-clock, **seed 20261002**
- Objective: class-weighted CE (20 command classes) + 1.0·CE (6 slot heads); softmax (no CTC / attention)
- Optimiser: AdamW, lr 1e-3, wd 1e-4, cosine schedule (eta_min 1e-5)
- Reproduce: `bash scripts/train.sh` · Export + parity check: `tools/export_onnx.py --verify`

## Where everything else lives
- **Repo (public, MIT, one-command reproduction):** https://github.com/jblagana/ai231-me2-v3
- **Dataset (licensed + citable):** HF `airimonda/ai231-me2-voice-commands` (81,768 clips; Zenodo DOI from the card's "Cite this dataset") + Google Drive `ai231-me2-gold-dataset` (master) · per-source licenses · Xela VCM **DOI 10.48804/IEKKVZ**
- **Model licence:** MIT

**This archive + the repo URL are the complete submission.**
