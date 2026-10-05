# HANDOVER — AI231 ME2 · Voice Command Model (VCM)

**From:** muji session (Agendas) · **To:** AI231 ME2 session · **2026-09-25**
**Deadline: Sat 2026-10-03** · Remote: `jblagana/AI231_ME2` (public) · Local: this folder

Read this first, then `PLAN.md` (architecture + timeline), then
`INSTRUCTIONS.md` (instruction log — the top entry is the handover + boss
clarifications). Specs live at `C:\Users\Jan\Muji\ai231_me2_specs.md`.
**After v1f lands:** read `DATASETS.md` (2026-09-29) — the verified map of
every real-human + synthetic dataset the class group has (SLURP, Fluent,
Snips SLU, Timers-and-Such, Common Voice, Mark's Option B, Ayla's pool) for
the robustness / real-human phase.

## State of the world (verified 2026-09-25 ~16:30)

### Done
- Repo + layout: `src/` (commands, make_dataset, model, train), `notebooks/`,
  `data/` (gitignored), `runs/` (gitignored checkpoints). Commits `aeef261`,
  `6bde4e4` pushed; this handover is the next commit.
- **`src/commands.py`** — 10 classes × 69 phrases, single source of truth.
- **`src/make_dataset.py`** — edge-tts TTS: 40 voices across 10 English accents,
  **speaker-disjoint** (20 train voices / 20 eval voices, split BEFORE
  generation so the model can't memorize speakers), 5 speeds, retries +
  concurrency.
- **`src/model.py`** — 3-block CNN, **94,410 params** (target was <1M),
  log-mel(80), 16 kHz, ~1 s window. ~1 ms/clip on this CPU machine.
- **`src/train.py`** — full pipeline: augmentation (speed jitter, gain, noise
  floor), smoke mode, speaker-disjoint eval, ONNX export hook.
- **`BENCHMARK.md`** — collective benchmark proposal (5 offline metrics +
  live task-completion + log format + "eval voices must not be in training"
  rule). **NEEDS TO BE PASTED into the ai231-me2 Telegram group** — the
  boss's 2-min unblock (he's been silent there 2 weeks; the 09-24 group
  protocol is waiting on input).

### Running right now
- **TTS dataset generation, background pid 10692** (`python src/make_dataset.py`),
  restarted 2026-09-25 ~19:05 after the first run (pid 15868) died on a DNS
  failure at 18:15 (7,576/13,800 clips). Progress in `tts_full2.log`
  (manifest is the truth: `data/raw/manifest.jsonl`). Check:
  `Get-Content tts_full2.log -Tail 3`.
  If the process is dead on resume, re-run `python src/make_dataset.py` —
  it skips existing files, so resume is safe.
- **Task #10** on the boss's local task list: train v1 when data lands,
  toast tomorrow 9AM.

### Next (Day 2 = 09-26)
1. TTS gen finishes → sanity-check `data/raw/` (per-class counts, listen to
   a few clips).
2. `python src/train.py --smoke` first (fast path), then full train v1.
3. Sanity metrics on the speaker-disjoint eval set.
4. Post BENCHMARK.md to the group (boss does the paste; muji can't send TG).

## Ratified decisions (don't re-litigate without boss)
- **10 classes as per spec**; parametric slots (dim to X%, timer X min) =
  **class detection only** — a 94K-param model isn't parsing numbers.
- **Wake word: IN SCOPE** (boss: "wakeword is included for the smart device").
  Separate 2-class gate (wake/no-wake) in front of the VCM — matches the
  09-22 group protocol (wake required while music plays, volume drops to 5%).
  Implementation lean: **tiny 2-class CNN** reusing the TTS pipeline (fully
  on-device, no vendor dep) vs Porcupine (free personal license, custom
  wake word, runs on Pi). Boss to veto. Benchmark inclusion TBD — the 09-24
  group protocol doesn't mention it; flag as an optional item in BENCHMARK.md.
- **TensorRT: NO** — no ARM64/RPi build (x86 + Jetson only), and irrelevant
  at 94K params (~1ms/clip). Single inference artifact = **ONNX**
  (ONNX Runtime CPU EP, ARM64 wheels).
- **Hardware (boss's own):** RPi **4B 8GB** + 64GB SanDisk Ultra SD + case
  with fan. RAM is a non-issue; the **SD card is the bottleneck** — use tmpfs
  for the mic ring buffer, minimal logging, watch SD wear.
- **HPC:** boss has access, **credentials not yet shared** — ask him when the
  Day-5 final train on the pooled collective dataset needs it. Local CPU
  torch is fine for 94K params.
- **Demo (task 5) = API-UI mock device** (boss ratified): one local web app,
  mic → wake gate → VCM → `POST /device/command` → mock state + status page.
  - dim_lights → mock light brightness slider (boss's pick, "free")
  - set_timer/alarm/reminder → local timer/reminder engine (due-time + toast;
    muji's own taskstore pattern is the reference)
  - set_temperature → mock thermostat
  - play_music / media_control → local audio player (pause/stop/next/volume)
  - make_call → mock dialer (shows number, no real call)
  - ask_question → weather/time via public API (writeup note: "no cloud"
    constrains VCM inference, not the action side)

## Gotchas (learned the hard way, don't relearn)
- **edge-tts rejects bare `"0%"` rate** — must be `"+0%"`. Already fixed in
  make_dataset.py; if you touch the rate strings, keep the sign.
- **`data/` is gitignored** — the dataset never goes to GitHub (it's TTS,
  regenerable; keep the repo light). `runs/` checkpoints too.
- **`.venv` is CPU-only torch** (deliberate — no GPU on this machine).
- **`tts_full.log` is tracked** (shows as modified during gen) — commit it at
  milestones, it's the progress record.
- **Speaker-disjoint split is sacred** — the whole point of the eval set.
  Never let an eval voice leak into training data (same rule is in
  BENCHMARK.md for the collective comparison).
- **Instruction log rule** (workspace root `AGENTS.md`): log every boss
  instruction verbatim in INSTRUCTIONS.md BEFORE doing work on it.
  `done` = pushed to the remote.
- **TG read-only:** muji can search/read the boss's Telegram but not send —
  group posts are the boss's manual step.

## Group context (ai231-me2 Telegram, searchable via muji's tg_search)
- 09-24 protocol: N non-owner evaluators, each command × N, output logs
  required; different command sets OK (2–3 variations); Sir's ruling: train
  from scratch, model must respond to ANY person.
- Benchmarks under discussion: latency, inference time, recall,
  task-completion rate, WER — our BENCHMARK.md covers all of these.
- Pooled data pipeline exists (ayla011/ai231-me2 repo + GDrive recording
  pool, student IDs as speaker labels) — Day-5 final train may use it.

## Ops quick-ref
```
python src/make_dataset.py            # TTS gen (resumable — fixed 2026-09-25:
                                      # manifest append + skip-done; was NOT resumable
                                      # before that fix; first run died 18:15 on a
                                      # DNS failure, restarted as pid 10692 → tts_full2.log)
python src/train.py --smoke           # fast training path check
python src/train.py                   # full train v1
Get-Content tts_full.log -Tail 3      # dataset gen progress
git -C <this folder> status           # only commit this repo's work
```
Python: `.venv\Scripts\python.exe` (PowerShell).
