# A2 build notes — 3.0s window + two-head (command + slot)

Goal: build, train, evaluate A2. Deadline Sat 2026-10-03.

## Ground truth (verified from raw_v1i manifest, 22,800 clips, 11 classes)
Class counts: media_control 3000, make_call 2800, control_lights 2400,
dim_lights 2000, play_music 2000, set_alarm 2000, set_reminder 2000,
set_temperature 2000, set_timer 2000, ask_weather 1600, ask_time 1000.

Parametric (slot-bearing) classes + slot vocab (derived from actual phrases):
- dim_lights (5): twenty, thirty, fifty, seventy, eighty  [%]
- set_timer (8): one, two, five, ten, fifteen, twenty, thirty, forty five [min]
- set_alarm (7): five am, five thirty am, six am, seven am, eight am, nine am, six pm
- set_temperature (7): eighteen, twenty, twenty two, twenty four, twenty five, twenty six, twenty eight
- set_reminder (10): buy groceries, call mom, drink water, pay the bills,
  take out the trash, water the plants, the meeting, five pm, tomorrow, next week
- make_call (6): mom, dad, brother, sister, friend, the doctor
Total slot values = 43. Non-parametric (no slot): ask_time, ask_weather,
control_lights, media_control, play_music.

## Architecture (tool-verified forward pass before commit)
Input (B,1,80,150) -> conv stack (32/64/128, 3x3 pad1, maxpool2 x3)
  -> (B,128,10,18)  [10 freq x 18 time cells]
Head 1 command: AdaptiveMaxPool2d(1) over all 18 cells -> FC(128,11)
Head 2 slot (per parametric class c): slice time cells SLOT_CELLS (default 3..18),
  AdaptiveMaxPool2d(1) -> FC(128, N_c)
Params: conv 93,120 + cmd FC 1,419 + slot FC 5,547 = 100,086.

## Loss
CE_cmd (all clips) + 0.5 * CE_slot (only slot-bearing clips, per-class).

## Recipe
v1g (v1k failed): jitter 0.95-1.05, gain +-6 dB, SNR 5-25 dB white noise.
Full retrain from scratch (no warm-start; 50->150 shape breaks conv weights).
50 epochs.

## Files (self-contained, do NOT touch n002's dirty model.py/train.py)
- src/slots.py       (new) taxonomy + phrase->slot extraction + self-test
- src/model_a2.py    (new) VCMTwoHead, self-contained conv stack, smoke
- src/train_a2.py    (new) 150-frame pipeline + dual-loss train, self-contained data utils
- src/eval_a2.py     (new) cmd acc + slot acc + confusion, vs v1i baseline 0.8618

## Train env
n002.ai.internal (A100). Dataset: data/raw_v1i (11-class, .raw s16le + manifest.jsonl).
v1i baseline: 0.8618 cmd acc. Fallback: if A2 < v1i, ship v1i command-only.

## Status
- [x] verify n002 data (.raw + durations + v1i timing)
  - 22,800 .raw files (no .mp3), 11 classes, median 2.35s, p90 2.95s, 92% <= 3.0s
  - v1i: 30 ep, ~3.2s/ep, best 0.862 (ep 27)
- [x] slots.py + self-test (63/63)
- [x] model_a2.py + smoke (100,086 params, shapes verified)
- [x] train_a2.py (dual loss, right-aligned 150-frame, v1g recipe)
- [x] eval_a2.py (cmd + slot acc + confusions)
- [x] commit b4f534b + push
- [x] sync to n002, smoke (cuda, 96 clips, 1 ep OK)
- [x] train (50 ep, GPU 6, ~8 s/ep, ~7 min)
- [x] eval + report

## RESULT (2026-09-30, runs/v1a2/vcm_a2_best.pt, eval n=11400)
- **command acc 0.889** (v1i baseline 0.8618) → **+2.7 pp**. The 3.0 s window
  fix works — A2 is the new champion on command.
- **slot acc 0.741** overall. By slot type:
  - categorical slots STRONG: set_reminder 0.932, make_call 0.932, set_alarm 0.846
  - number-word slots WEAK: dim_lights 0.326, set_temperature 0.539, set_timer 0.588
  - (number words "twenty/eighteen/fifty/seventy" are acoustically near-identical
    at 1.6 kHz mel; a number decoder / mini-ASR is the production path — the CTC
    lever already in Future Work)
- per-class cmd: set_temperature 0.999, set_timer 0.962, make_call 0.946,
  ask_time 0.928, set_reminder 0.925, set_alarm 0.924, control_lights 0.868,
  play_music 0.861, ask_weather 0.850, dim_lights 0.824, media_control 0.754
  (media_control is the weakest — "pause/stop/skip" are short + shared with
  play_music; the top confusions are media_control<->play_music<->make_call)
- top cmd confusions: media_control->play_music 167, media_control->make_call
  160, control_lights->media_control 99, play_music->media_control 88
- **Decision: A2 ships as the new command model (0.889 > 0.8618).** The slot
  head is a bonus (categorical slots demo-ready; number slots need a decoder).
  Fallback to v1i is NOT needed — A2 beats it on command.

## Key build decisions (verified, not guessed)
- **Right-align the 3.0 s window** (pad LEFT), not center: the slot word is the
  LAST word, so it must sit at the tail where the slot head reads. A center
  alignment put short clips' slot word ("call mom") in the middle, outside the
  tail slice. Command head uses global max (alignment-invariant), so right-align
  costs it nothing.
- **Slot head reads TAIL cells** (last 8 of 18), not a front slice: make_call's
  slot word is at cell ~1.2 (short clip), so a front slice [K:] would drop it.
  A back slice [-8:] captures the slot word for ALL clip lengths (verified
  _a2_cellprobe.py: last-word-start cell ranges 1.2 (call) to 11 (timer)).
- **GPU precompute of base log-mels**: the 3.0 s mel is ~8x the 1.0 s work
  (~354 ms/item on CPU -> ~135 min for 22.8k; ~2-5 ms/item on A100 -> ~2 min).
  Precompute on GPU into a CPU RAM cache; per-item training path stays a lookup.
- **Slot label folded into the items tuple** (not a parallel list) so the
  single shuffle keeps (clip, slot) aligned — a parallel list desyncs after
  shuffle (caught in smoke: eval slot coverage 0%).
