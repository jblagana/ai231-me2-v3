# V2 MAPPING — VCM_BALANCED → the 10-class spec (2026-09-29)

**Audited by:** muji · from `C:\Users\Jan\Downloads\VCM.zip` (2.83 GB, 51,970 entries,
CRC spot-check 3/3 OK) · class×source×speaker counts parsed from
`VCM_BALANCED/manifests/audio_provenance.csv` + `train.csv`, split counts from
`VCM_MASTER/manifests/{train,val,test}.csv`.

## The 10-class mapping (VCM_BALANCED, 15,268 clips)

| Spec class (commands.py) | VCM source class(es) | Clips | Speakers | Source dataset(s) |
|---|---|---|---|---|
| `play_music` | PLAY_MUSIC | 600 | 70 | FluentSpeechCommands (495 orig + 105 aug) |
| `ask_question` | WEATHER + TIME (merge) | 1,400 + 600 = **2,000** | 94 / 63 | SLURP (weather_query, datetime_query, qa_*) |
| `control_lights` | LIGHT_ON + LIGHT_OFF (merge) | 1,200 + 1,200 = **2,400** | 76 | FluentSpeechCommands |
| `dim_lights` | LIGHT_DIM | 522 | 65 | SLURP (261 orig + 261 aug) |
| `set_timer` | SET_TIMER | 600 | 71 | TimersAndSuch (545 orig + 55 aug) |
| `set_alarm` | SET_ALARM | 1,000 | 142 | SLURP 727 + TimersAndSuch 273 |
| `set_temperature` | SET_TEMPERATURE | 600 | 33 | **120 real** (SET_TEMPERATURE_REAL) + 480 Piper-TTS synth |
| `media_control` | VOLUME_UP + VOLUME_DOWN + MEDIA_PAUSE + MEDIA_STOP + MEDIA_NEXT (merge) | 1,200+1,200+484+574+956 = **4,414** | 72/73/67/68/16 | FSC (all except NEXT) + Multi-Sensor (NEXT) |
| `set_reminder` | — **NO REAL CORPUS** — | 0 | — | gap → Ayla's pool / Common Voice / real-voice TTS |
| `make_call` | — **NO REAL CORPUS** — | 0 | — | gap → Ayla's pool / Common Voice / real-voice TTS |

**Real-voice coverage: 8/10 classes, 12,136 clips.** The 2 gap classes are
exactly the ones flagged in the SLURP audit — VCM confirms, not changes, the
gap set. `set_temperature` is 80% synthetic (Piper) — the weakest "real" class;
its 120 real clips are the only true recordings.

## OOD / reject classes VCM adds (not in the 10-class spec)

| VCM class | Clips | Note |
|---|---|---|
| UNKNOWN | 2,500 | out-of-domain reject (SLURP) — 16.4% of the balanced set |
| SILENCE | 632 | GSC background noise — only 2 "speakers" (it's ambient) |

Spec has no reject class. **Recommendation:** keep them out of the v2 10-class
train for now (spec compliance), but stage them for a v2.1 12-class run —
a tiny VCM that can say "I didn't catch that" is worth a lot on RPi, and the
clips are already there.

## ⚠ Split issue — the one real blocker

**VCM_BALANCED is train-only.** `manifests/` contains `train.csv` (15,269 rows)
+ `audio_provenance.csv` — **no val.csv, no test.csv**. VCM_MASTER has proper
splits (train 25,628 / val 4,735 / test 4,759, all 16 classes) but the balanced
subset was never re-split.

- `train.csv` has a `group_id` column (13,084 distinct groups) — the intended
  leak-prevention key (group = speaker+source cluster).
- **Fix (next step):** derive val/test from **VCM_MASTER's existing splits**
  (already speaker-disjoint per the QC reports) by intersecting with the
  balanced selection manifest — do NOT carve a new split from the balanced
  train.csv (that would leak the augmentation twins: 1,467 augmented clips
  share `group_id` with their originals).
- MASTER splits (verified row counts): train **27,130** / val **4,735** /
  test **4,759** → val+test ≈ 25.9% of MASTER. Applied to the 15,268
  balanced clips: expected ≈ **11.3k train / 1.0k val / 1.0k test**
  (exact numbers come from the manifest intersection, not the ratio).

## Format check

VCM_BALANCED audio: **16 kHz mono 16-bit WAV** (verified from provenance
columns: `sample_rate=16000, channels=1, bit_depth=16`, 15,268 files,
1.27 GB uncompressed). Matches the v1 pipeline's `wav_to_logmel` input spec
exactly — no resampling needed. (The zip says "flac tier" in the report
header but the actual files are WAV — the report is stale on this point;
the audio is fine.)

## v2 decision update (supersedes the SLURP-backbone decision)

The "reuse SLURP as backbone" decision from earlier today is **superseded**:
VCM_BALANCED is a strict superset (SLURP + FSC + TimersAndSuch + Multi-Sensor,
pre-balanced, pre-QC'd, speaker-leakage-checked). v2 backbone = **VCM_BALANCED**
with the MASTER-derived split. The 2 gap classes (`set_reminder`, `make_call`)
still need sourcing — the 1-clip Chatterbox de-risk test remains the gate for
whether real-voice TTS is a legitimate filler there.

## Status

- [x] Audit (this file)
- [x] VCM.zip → n003 scp (launched 2026-09-29 19:51, ETA ~20:30)
- [ ] Verify remote VCM.zip (size + `unzip -t` sample)
- [ ] Extract VCM_BALANCED → `~/vcm/data/VCM_BALANCED/` on n003
- [ ] Derive val/test from MASTER splits → `VCM_BALANCED/manifests/{val,test}.csv`
- [ ] Relabel 16→10 classes (merge map above) → `data/v2_raw/`
- [ ] 1-clip Chatterbox de-risk (gate for reminder/call fill)
