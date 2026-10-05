# AGENTS.md — AI231 ME2 workspace

## Instruction log rule
Every user instruction is logged **verbatim, newest first** in `INSTRUCTIONS.md`,
before any work on it. The Interpretation section is agent-owned; a user edit is
the new instruction (stop, log it, follow it). Re-read before each log update and
before `git push`. `done` = **pushed to the remote repo** (jblagana/AI231_ME2).

## Project ground truth
- **First read: `HANDOVER.md`** (state of the world: done/running/next,
  ratified decisions, gotchas, ops quick-ref). Then `PLAN.md` (architecture
  + timeline — re-read at session start; update as you go). `WHYS.md` =
  the "why" behind every architecture decision (writeup + slide source).
- Specs: `C:\Users\Jan\Muji\ai231_me2_specs.md` (boss-confirmed 2026-09-25)
- Deadline: **Sat 2026-10-03**
- Hard constraints: no ASR, no LLM, no cloud at inference, tiny (RPi real-time)
- Hardware: boss's RPi **4B 8GB** + 64GB SanDisk Ultra SD + case with fan
  (SD card is the bottleneck — tmpfs ring buffer, minimal logging)
- Wake word: **in scope** (2-class gate in front of the VCM; TBD vs Porcupine)
- Inference artifact: **ONNX** (ONNX Runtime CPU EP ARM64). TensorRT: no
  (no ARM64 build, irrelevant at 94K params)
- HPC: boss has access for the final pooled-dataset train — ask him for
  endpoint/creds when needed
- Shared memory: `C:\Users\Jan\Muji\REPOS.md` (repo map),
  `C:\Users\Jan\Muji\BACKLOG.md` (parked work), muji's `learned.md` /
  `tool_notes.md` at `C:\Users\Jan\.cline\data\workspaces\chat\muji\`

## Ops
- Python: `.venv` (CPU torch + edge-tts + soundfile). No GPU on this machine.
- Data lives in `data/` (gitignored); code in `src/`; notebooks in `notebooks/`.
- Commit only this repo's work; commit then push.
