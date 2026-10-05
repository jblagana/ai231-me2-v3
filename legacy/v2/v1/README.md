# AI231 ME2 — Voice Command Model (VCM)

Tiny on-device voice command classifier for the 10 most common smart-device
commands. **No ASR, no LLM, no cloud** — raw audio → log-mel → small CNN →
command class, real-time on RPi 4/5.

## Status
Day 1 (2026-09-25) — dataset pipeline + model v1. See `PLAN.md` for the
architecture, timeline, and benchmark proposal. Deadline: **Sat 2026-10-03**.

## Quickstart
```bash
.venv/Scripts/python.exe src/make_dataset.py        # generate TTS dataset -> data/raw
.venv/Scripts/python.exe src/model.py --smoke       # forward pass + param/latency check
.venv/Scripts/python.exe src/train.py --smoke       # 1-epoch pipeline check
.venv/Scripts/python.exe src/train.py --epochs 15   # real training -> runs/v1
```

## The 10 classes
`play_music · ask_question · control_lights · dim_lights · set_timer ·
set_alarm · set_temperature · media_control · set_reminder · make_call`
(single source of truth: `src/commands.py`)

## Notes
- Dataset: edge-tts synthesis, 40 voices split speaker-disjoint (20 train /
  20 eval), speed jitter + noise augmentation on the train side only.
- Model: 3-block CNN (32/64/128), target < 1M params, 80-bin log-mel, 1 s window.
- Export path: PyTorch → ONNX → ONNX Runtime on the RPi demo.
