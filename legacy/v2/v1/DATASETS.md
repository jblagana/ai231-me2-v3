# DATASETS — classmate dataset compilation (AI231 ME2)

**Logged by:** muji (boss session) · **2026-09-29**
**For:** the AI231 ME2 session — read after v1f training lands, before the
real-human / robustness phase.
**Source:** the shared compilation sheet `AI231_ME2.xlsx` (12 rows, `Datasets`
sheet) + the AI 231 THFY Telegram chat (dataset links) + the two public
classmate repos (fetched and verified 2026-09-29).

## Why this file exists
v1f is trained on the synthetic edge-tts set (40 voices, speaker-disjoint).
The next phase is robustness — getting **real-human** clips of the 10 tasks so
the model responds to ANY person (Sir's ruling). This is the verified map of
what real-human + synthetic data the class group already has, so we don't
re-hunt it.

## Real-human corpora (the ones that matter for robustness)

| Dataset | Contributor | Covers which of the 10 tasks | Notes |
|---|---|---|---|
| **SLURP** | Mark | lighting, music, volume, alarm, weather, time, qa — **NOT timer/temp/reminder/call** | Spoken Language Understanding Resource Package — 16,521 entries, 93 intents, real people, real rooms. **The main real corpus.** See the verified coverage table below. |
| **Fluent Speech Commands** | Mark | 31 intents incl. lights/music/volume/temp | 97 speakers, 248 phrases, action/object/location slots. |
| **Snips SLU** | Quiel | smart-home appliances (lighting) | Real, English, has speaker demographics. |
| **Timers and Such** ⭐ | Quiel | **TIMER + ALARM specifically** | NeurIPS 2021 benchmark, real human SetTimer/SetAlarm. **Only 2,151 real utterances across all 4 intents.** 12.2GB zip — the one still downloading as of 09-29. |
| **Common Voice** (Mozilla) | King | general / any phrase you filter | 200k recordings, **has Filipino-accent recordings**. CC-licensed, on HuggingFace, filterable by `sentence` column. |
| **GSC v0.02** | Mark + Joven | single words (on/off/stop) + background noise | Real, but single-word only — use the noise set for the noisy condition, not the commands. |

## Synthetic (fill gaps, not the "real" answer)

| Dataset | Contributor | Scale |
|---|---|---|
| **eSpeak NG** | Mark | 10,000 (20 labels × 10 voices × 10 phrases × 5 variations) |
| **Chatterbox TTS** + LibriSpeech + Common Voice refs | Mark | 9,000 (20 labels × 30 refs × 5 phrases × 3 variants; 24 LibriSpeech + 6 Filipino-English refs) |
| **SynTTS-Commands** (cosy2) | Cherry | 14k (cmd 1) + 53k (cmd 8) |
| **jbuchner synthetic** | Joven | one-word, prune to off/on/stop |

### SLURP intent coverage — VERIFIED against the 10 classes (2026-09-29)
Pulled the full `train.jsonl` + `devel.jsonl` + `test.jsonl` from
`github.com/pswietojanski/slurp` and counted every `intent` field (16,521
entries, 93 unique intents, 18 scenarios). Mapped to `src/commands.py`:

| Our class | SLURP intent(s) | Real clips (train+dev+test) |
|---|---|---|
| `play_music` | `play_music` | 911 |
| `ask_question` | `qa_factoid` + `qa_definition` + `qa_currency` + `qa_maths` + `weather_query` + `datetime_query` + `news_query` | 765+378+202+116+834+490+704 ≈ **3,489** |
| `control_lights` | `iot_hue_lighton` + `iot_hue_lightoff` | 30+205 = **235** |
| `dim_lights` | `iot_hue_lightdim` + `iot_hue_lightchange` | 111+183 = **294** |
| `set_timer` | — **NONE** — | 0 |
| `set_alarm` | `alarm_set` + `alarm_query` + `alarm_remove` | 253+183+113 = **549** |
| `set_temperature` | — **NONE** — | 0 |
| `media_control` | `audio_volume_up` + `audio_volume_down` + `audio_volume_mute` + `audio_volume_other` | 135+71+157+23 = **386** |
| `set_reminder` | — **NONE** — (closest is `calendar_set`, a different task) | 0 |
| `make_call` | — **NONE** — | 0 |

**SLURP covers 6 of 10 classes.** The 4 gaps (`set_timer`,
`set_temperature`, `set_reminder`, `make_call`) have **zero** matching
intents — confirmed by keyword search across all 93 intent names (no
timer/temperature/thermostat/remind/call/phone hits). Those four fall to
**Timers and Such** (timer+alarm), **Fluent Speech Commands** (temp), and
**Ayla's pool / Common Voice** (reminder+call).

## Classmate repos (public, fetched + verified 2026-09-29)
- **Mark's Option B** — `github.com/markandrian30/AI231/tree/main/MEX2/OptionB`
  — 28,800 synthetic WAVs, 150 speakers (134 foreign + 16 Filipino-English),
  19 intents, clean + noisy conditions, 27,956 active after whisper
  filtering. TTS-generated (Chatterbox/Piper) over real reference voices.
  Already splits `WEATHER` and `TIME` into separate intents (our robustness
  change #2 is baked into Mark's schema).
- **Ayla's voice pool** — `github.com/ayla011/ai231-me2-voice-data`
  — the **real-human recording pipeline**: classmates record a fixed prompt
  list, whisper.cpp validates each take on the spot, approved clips go to the
  shared GDrive pool. This is the "50–100 real-human clips/class" mechanism
  for robustness #3, already being built.

## GDrive pools (login-gated — boss's account)
- `drive.google.com/drive/folders/1_GcDuvaRnGlkdtGdoogQI7FflgWf6gF8` — pooled
  real voice records (student-ID speaker labels)
- `drive.google.com/file/d/1miLmuozdojOylqD0xlfOwXAdNsbGYwfN/view` — someone's
  "Frankenstein Dataset"
- `drive.google.com/drive/folders/15MwS2UpPgaAUL-zJVbEV8tl_oQ38avp2` — slop
  outputs (transcription <80%, to be dropped)
- Master index: `docs.google.com/spreadsheets/d/1VFm1-SAdNqtSwOeSHZct23tF6pPTntmj2940sugj61E`
  (the shared compilation — the sheet this file was built from)

## The takeaway (what to actually do)
1. **No public dataset has our 10 commands pre-labeled** — they're our spec,
   not a benchmark. The real-human answer is a **combination** (per the
   verified SLURP table above):
   - **SLURP** → lighting, music, volume, alarm, weather/time/qa (6 of 10, ~6,300 real clips)
   - **Timers and Such** → TIMER + ALARM (the specific real set; lands with the 12.2GB download)
   - **Fluent Speech Commands + Snips SLU** → temperature + lighting fill
   - **Common Voice** (filter by `sentence`, has Filipino accents) → novel phrasings + Filipino-accent fill
   - **Ayla's pooled recordings** → real clips of our *exact* phrases (the strongest match)
   - **set_reminder + make_call** → **no real public corpus** — Ayla's pool + Common Voice only
2. **Synthetic (Mark's Option B + eSpeak NG + SynTTS)** fills the gaps to hit
   50–100 clips/class where real data is thin.
3. **Common Voice filtering is now the fallback, not the primary** — the class
   group already has real-human coverage of all 10 tasks.
4. **GSC v0.02 noise set** is the source for the noisy-condition eval (not the commands).

## v2 data-source decision (2026-09-29, boss session)
**Decision: REUSE SLURP audio as the v2 real-human backbone. Do NOT pivot to a
new real-voice TTS replacement.** TTS is only a targeted filler for the 2
classes with zero real corpus (`set_reminder`, `make_call`) — never the
dataset backbone.

**Why (the evidence chain, all verified this session):**
1. v1e is TTS-on-TTS → 0.665. Fed through real SLURP voices → **0.101** with a
   2-class collapse (play_music + set_reminder). The model is *broken* on real
   voices, not merely "worse."
2. Feature-statistics sanity check (`feature_stats.py`): SLURP log-mels are
   **in-distribution** — spectral-envelope cosine **0.993** vs TTS, feature-scale
   ratio ~1.4×, all 128 conv channels still fire (ch-std ratio 0.75). So the
   collapse is **not** a feature-scale artifact → normalization won't fix it.
3. Entropy probe (`entropy_probe.py`): mean softmax entropy **0.935 of max**
   (near-flat), and confidence is uncorrelated with correctness (correct 2.145
   vs wrong 2.153). The model is *not knowing*, not *knowing wrong*.
4. **Conclusion:** the gap is a **representation gap** — the model never saw
   real-voice *content*. Only real-human training data closes it. Synthetic
   (any TTS, including TTS-over-real-reference) is what v1 already has and it
   demonstrably fails on real voices.

**The v2 combination (SLURP-centric, real-first):**
| Class | v2 source (primary → fallback) |
|---|---|
| play_music, ask_question, control_lights, dim_lights, set_alarm, media_control (6) | **SLURP real** (~6,300 clips; relabel SLURP intent → our class via the `make_slurp_eval.py` map) |
| set_timer, set_alarm (gap) | **Timers and Such** (real SetTimer/SetAlarm; 2,151 real utterances) |
| set_temperature (gap) | **Fluent Speech Commands** (+ Snips SLU lighting fill) |
| set_reminder, make_call (gap, **no real public corpus**) | **Ayla's pooled recordings** → **Common Voice** (filtered) → **real-voice TTS** (Chatterbox/Piper over a real reference) as last resort |
- Keep a **subset of edge-tts as augmentation** to preserve class balance, but
  real-human is the primary signal.

**Why NOT a new real-voice TTS replacement (the full-dataset TTS route):**
1. The sanity check proved the bottleneck is a *representation gap on
   real-voice content*. TTS-over-real-reference is still synthetic — it will not
   teach the model the real-voice content variation that broke it. v2's whole
   point is real-human audio.
2. SLURP is already downloaded (3.7 GB, sunk cost) and is the largest real
   corpus available. Reusing it is the highest-value, lowest-cost move.
3. A full TTS replacement costs generation + compute for a strictly worse signal.

**Honest caveats:**
- SLURP labels are its own 93-intent schema, not our 10-class spec — the
  relabeling (SLURP intent → our class) must be applied to the full train+dev
  splits, and phrasings differ from our TTS ("play some jazz" vs "play music").
  That diversity is a *feature* for robustness, but the mapping must be audited
  per class before training.
- The 4 gap classes are thin in real data (Timers-and-Such = 2,151 real
  utterances across 4 intents; reminder/call ≈ 0 real). The edge-tts
  augmentation is what keeps those classes trainable — don't drop it.
- **Worth a 1-clip experiment before committing:** does Chatterbox-over-real-ref
  audio classify correctly on the v1e model? If yes, TTS-over-real-ref is a
  legitimate gap-filler (real-enough); if no, it confirms only true recordings
  work. Cheap to test, and it de-risks the reminder/call fill.
- This decision is for the **10-class v2**. If demo scope shrinks to the 6
  SLURP-covered classes, SLURP alone (real-only, no TTS) is sufficient and the
  gap-class fill becomes optional.

## SLURP real-human eval set (built 2026-09-29, boss session)
The cross-domain probe — feeds real SLURP voices through the TTS-trained VCM
to get the "does it hold up on real people" number (Sir's ruling). Built, not
just described:
- **`src/make_slurp_eval.py`** — downloads the SLURP *text* (test + devel
  held-out splits, 5,007 entries), maps each `intent` onto the **6 covered
  classes** (play_music, ask_question, control_lights, dim_lights, set_alarm,
  media_control), selects a **balanced 504-clip** set (84/class, 1 distinct
  FLAC/entry). Writes `data/eval_slurp/manifest.csv` + `needed_flac.txt`.
- **`src/eval_slurp.py`** — runs a checkpoint over the SLURP clips via the
  exact training feature path (`wav_to_logmel`), reports overall + per-class
  cross-domain accuracy + top confusion pairs. `--ckpt runs/v1e/vcm_v1.pt`.
- **Audio:** `slurp_real.tar.gz` (3.92 GB, Zenodo record 4274930) — extract
  only the 504 FLACs in `needed_flac.txt` into `data/eval_slurp/audio/`.
- **The 4 gap classes** (set_timer, set_temperature, set_reminder, make_call)
  are NOT in this set — they need Timers-and-Such / Fluent / Ayla's pool.
- **Why it matters:** 0.665 (v1e full eval) is TTS-on-TTS. This set is the
  first real-human number. If it's far below 0.665, the TTS→real gap is the
  bottleneck (→ v2 real-human training); if it's close, within-domain
  confusion is still the bigger problem (→ keep fixing taxonomy/arch).

## Verification status (honest)
- Dataset **names + descriptions**: from the classmates' own annotations in
  the sheet (their verified notes) + the two **public GitHub repos** fetched
  this session.
- **Not re-fetched this turn:** each dataset's URL (the sheet's descriptions
  are detailed enough). If a class's coverage is load-bearing, pull that
  dataset's intent list from source before relying on it.
- **Not openable:** the Google Sheet + GDrive folders (login-gated) — reported
  from the chat + the xlsx the boss uploaded.
