# legacy/v2 — V2 archive (frozen 2026-10-05)

V2 (own-recorded-data, 10-class wake + 6-slot BCResNet, the 10/02 Pi system)
archived here so it is available anytime. V3 (this repo) is the live line;
the V2 GitHub repo (jblagana/AI231_ME2_V2) is frozen as the historical record.

- src/      V2 training/eval/export/wake code (10-class, 6 slots)
- tools/    V2 tooling (TTS, monitors, slot matrix, music-player test)
- hpc/      HPC deploy/launch/monitor scripts (n002/n003 era)
- v1/       V1 notes (pre-V2 docs)
- data/     V2 dataset + artifacts:
  - mine/            boss-recorded eval clips + vocab/quirk scans
  - pi_music/        the 5 locked songs (48 kHz mono WAVs, ~104 MB)
  - pi_takes/        119 Pi calibration takes (16 kHz WAVs)
  - pi_pkg_v2e_ep800/ the shipped v2e ep800 Pi package (model + meta)
  - *.json/*.log     eval matrices + calibration logs
- *.md      V2 working docs (BENCHMARK/JOURNAL/DECISIONS/DEFENSE/...)

HPC n002 drift (stale fork worktree ~/ai231_me2_v2) is preserved in the V2
repo under hpc_drift_20261005/ (commit a4d8aee) — only deltas vs local:
older pi/pi_demo.py + me2_ui.py, a --record rec flag in run_demo.sh,
tools/mon_summary.json, BENCHMARK/JOURNAL drift.
