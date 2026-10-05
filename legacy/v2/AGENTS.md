# AGENTS.md — AI231 ME2 V2 workspace

## Instruction log rule
Every user instruction is logged **verbatim, newest first** in `INSTRUCTIONS.md`,
before any work on it. The Interpretation section is agent-owned; a user edit is
the new instruction (stop, log it, follow it). Re-read before each log update
and before `git push`. `done` = **pushed to the remote repo**
(jblagana/AI231_ME2_V2).

## Project ground truth
- **First read: `DECISIONS.md`** (ratified decisions: architecture, data
  policy, slot taxonomy, demo design). `README.md` = status + layout.
- The V1 repo is read-only heritage, NOT extended:
  `C:\Users\Jan\OneDrive - University of the Philippines\MASTERS\GRAD\2s
  2025-2026\AI 231\AI 231 ME2\` (HANDOVER.md, BENCHMARK.md, V2_MAPPING.md,
  a2_notes.md, learned.md). V2 rebuilds from scratch in THIS repo.
- Raw spec: verbatim in `INSTRUCTIONS.md` (instruction 1).
- Deadline: **Sat 2026-10-03**
- Hard constraints: no ASR, no LLM, no cloud at inference, tiny (RPi 4B 8GB
  real-time), 11 command classes (spec's 10 items; ask_question split
  weather/time), wake word "hey boots" in scope.
- **Data policy (boss-ratified):** TRAIN = synthetic only (edge-tts,
  speaker-disjoint); AUTHENTIC VCM audio = isolated EVAL-REAL only, never
  training. Model selection on EVAL-SYN (pre-registered).
- Inference artifact: **ONNX** (ONNX Runtime CPU EP ARM64), ~107K params.
- HPC: `jan.rhey.lagana@n002.ai.internal`; shared JuiceFS `~` visible from
  n002+n003. HPC working copy: `~/ai231_me2_v2`. V1 assets: `~/vcm/`
  (VCM_BALANCED audio + whisper scour in `~/vcm/runs/v1i/`).
- Local python: `.venv` (CPU torch + edge-tts + soundfile), no GPU — TTS
  runs locally overnight, train/eval on HPC.

## Layout
- `src/commands.py` — 11 classes × phrases (the training phrasebook)
- `src/slots.py` — 98 slot values, 7 per-class slot heads
- `tools/` — mine_phrases.py (whisper transcripts → phrases.json),
  build_dashboard.py (phrases.json → site/index.html),
  prep_real_eval.py (HPC: reconcile VCM.zip, MASTER split, relabel 16→11)
- `site/` — static work dashboard (open `site/index.html`)
- `data/` — gitignored: `data/mine/*.csv` (scour artifacts), TTS output
- `runs/` — gitignored: training runs, logs, onnx exports

## Ops
- Commit only this repo's work; commit then push.
- HPC long jobs: screen, torch `-B 0`, log to file (V1 lesson).
- **HPC training interpreter: `~/.conda/envs/me2/bin/python`** (torch
  2.14.0+cu130, CUDA verified 10-01). `~/vcm/.venv` has CPU-ONLY torch —
  never launch GPU jobs with it (caught 10-01: `device: cpu` in log).
- Dashboard rebuild: `python tools/mine_phrases.py && python tools/build_dashboard.py`
