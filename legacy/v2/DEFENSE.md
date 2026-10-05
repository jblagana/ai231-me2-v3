# Project Defense Manual — AI231 ME2 V2

> Read this once top-to-bottom (~20 min). The Numbers Cheat Sheet at the top
> is for the last 5 minutes before you present. Sections 4–6 are your
> ammunition: every design call, every number, and the hard questions with
> the crisp answers. Section 8 is the honest limitations list — knowing it
> before they find it is what separates "defended" from "dug in".

## 0. Numbers cheat sheet (memorize)
| Thing | Value |
|---|---|
| Command classes | **11** (10-class ablation merges ask_weather+ask_time → ask_question) |
| Phrases | **236** (119 of them carry a slot value) |
| Slot values | **119** across **7** parametric classes (dim 12, timer 24, alarm 48, temp 13, reminder 10, call 6, song 6) |
| Training audio | **47,200 clips** = 236 phrases × 40 voices × 5 speeds |
| Voices | **40** (10 English accents × 4), **speaker-disjoint**: 20 train / 20 eval, split *before* generation |
| Window / feature | **3.0 s**, 16 kHz, **80-mel**, **150 frames** (20 ms hop, n_fft 1024) |
| Primary model | **v2cnn — 109,890 params** (3-block CNN 32/64/128, two heads) |
| Experiment model | **bcresnet — 36,114 params** (BC-ResNet, arXiv 2106.04140, scale 2) |
| Baseline to beat | v1i: **EVAL-SYN 0.8618**, **EVAL-REAL 0.2289** |
| Stretch | 0.889 EVAL-SYN (V1's A2) |
| On-device | **ONNX + numpy mel + onnxruntime CPU** — no torch, no cloud, ~433 KB (v2cnn) / ~173 KB (bcresnet) payload |
| Pi latency | ~1 ms/clip on a server CPU → single-digit ms on Pi 5, tens of ms on Pi 4 (measured on-device) |
| Hardware | Raspberry Pi 4/5 + USB mic, ~$55–110 total |

## 1. The 30-second pitch
"We built a voice-command smart device that runs **entirely on a
Raspberry Pi** — no cloud, no vendor API, no subscription. A ~110K-parameter
model (0.1% of a typical speech LLM) hears 11 command classes with 119
parameter values — *'set the timer for twenty five minutes'*, *'dim the
lights to forty percent'*, *'set an alarm for seven thirty am'* — and fires
the right command **plus the exact slot value**, on-device, in a few
milliseconds. The whole pipeline — dataset protocol, training, leak-free
evaluation, ONNX export, and a verified feature-parity gate so the Pi
reproduces training bit-for-bit — is built and tested; the Pi OS image is
already flashed."

## 2. The problem, and why it's actually hard
- **The task:** free-form short commands (3–9 words) → one of 11 intents +
  an exact parameter (a number, a time, a song title, a contact).
- **Why naive approaches fail on-device:**
  - ASR→NLU cascade (Whisper + LLM): 0.5–5 GB models, seconds of latency,
    needs a phone/server, and the number/slot step is where cascades lose
    accuracy. We don't need words — we need the *intent+slot*.
  - Keyword-spotting (Porcupine-style): exact keywords only; no open
    phrasings, no slots, vendor lock-in (free for personal use, but the
    model is a black box and can't express *'dim to forty percent'*).
- **What we do instead:** an end-to-end classifier on 3 s of log-mel
  spectrogram. The architecture is small by construction (pinned budget),
  the slot is read from a **validated tail of the spectrogram** (the last
  spoken word — probed empirically in V1), and everything — including the
  mel extraction — runs in numpy/ONNX on the Pi.

## 3. System architecture (draw this if asked)
```
[USB mic] → 3.0 s @16 kHz → numpy log-mel (80×150, shipped window+filterbank)
  → ONNX (conv/pool/BN/linear, opset 17) → head 1: cmd logits (11)
                                         → head 2: slot logits (per-class vocab)
  → JSON {cmd, conf, slot} → actuation: GPIO LED/beep + LAN dashboard + muji UI
```
- **Trains on HPC** (4× GPU, torch); **runs on Pi** (onnxruntime CPU).
- The mel is *not* in the ONNX graph (FFT isn't exportable to legacy ONNX)
  — it's computed in numpy on the Pi from the exact training window
  (1024,) + filterbank (513, 80) shipped as .npy files, and **verified
  equal to the training mel (<1e-3) on real clips before anything ships**.

## 4. Key design decisions — and the one-line defense of each
1. **Locked 3 s window / 80-mel / 150 frames.** Every phrase is 3–9 words
   (< 3 s at TTS speed); 3 s leaves margin and makes the input a *fixed
   shape* — no padding logic, no variable-length graphs, tiny model.
   *Defense: "fixed input shape is what lets a 36K-param network run on a
   Pi; variable length is the thing that kills on-device latency."*
2. **Two heads: command + per-class slot.** The slot value is the last
   spoken word, so each slot head max-pools only the **last 8 time cells**
   (~1.3 s) of the feature map — validated in V1 (`_a2_cellprobe`): the
   slot word lives in the tail for every class, and the verb (cells 0–3)
   is excluded so heads don't latch onto shared "set/call/play" boilerplate.
   *Defense: "it's a targeted ablation of *where* the answer is in the
   spectrogram, not a guess."*
3. **v2cnn = V1's A2 architecture, pinned at 109,890 params.** V1's A2 was
   the best model we've ever measured; V2 inherits its conv stack exactly
   (a pinned budget in the smoke test keeps it honest).
4. **bcresnet = BC-ResNet (arXiv 2106.04140), faithful port, scale 2,
   36,114 params.** The paper's SOTA keyword-spotting architecture: 1D
   *temporal* convs with dilations 1/2/4/8 over the full time axis, freq
   compressed to sub-bands with a broadcasted residual connection. We test
   whether a **3× smaller** model matches the CNN. *Defense: "we're not
   shopping for architectures — we're running one pre-registered fair
   experiment: does a 36K-param specialist beat an 110K-param CNN on the
   same locked feature?"*
5. **10-class ablation (--merge-10).** Merging ask_weather+ask_time into
   ask_question isolates the cost of fine-grained non-parametric classes.
   Four models total = 2 architectures × 2 taxonomies.
6. **Speaker-disjoint protocol (the integrity core of the project).** 40
   voices are split 20/20 *before* any audio is generated. The eval set is
   audio from voices the model never heard in any form. *Defense: "a voice
   model that memorizes speakers scores well and means nothing; this
   protocol makes the metric measure what we actually care about —
   generalization to new voices."*
7. **EVAL-SYN selects, EVAL-REAL reports — never the reverse.** Model
   selection happens **only** on EVAL-SYN (synthetic, speaker-disjoint,
   pre-registered, ≤1 targeted follow-up run allowed). EVAL-REAL (authentic
   VCM recordings, 9,492 clips, val/test speakers disjoint from everything)
   is isolated: we look at it **once, after selection**. *Defense: "this is
   the difference between a benchmark and a leak — the selection signal and
   the honesty signal are physically separated."*
8. **Real-noise augmentation (MUSAN) + gain jitter + speed jitter.**
   Training augments with real background noise at random SNR so the model
   meets the real room; 5 TTS speeds cover speaking-rate variance.
9. **ONNX + numpy mel, verified not trusted.** No torch on the Pi (no 2 GB
   dependency, no GPU driver, no license surface). The Pi's mel is computed
   in numpy from shipped buffers and a parity gate (`verify_pi_v2.py`)
   re-checks numpy-mel == training-mel and ONNX == torch **on real clips,
   <1e-3, before shipping** — because a silent feature mismatch is the #1
   way on-device accuracy dies without anyone noticing.
10. **Class weights + real noise floor in training** — the V1 recipe that
    got 0.8618; V2 ports it unchanged so the delta is the data, not the
    optimizer.

## 5. Methodology, in the order a reviewer will attack it
1. **Dataset:** 236 locked phrases (no locations, numbers as words,
   Celsius only, no open vocabulary — boss-ratified), each TTS'd on 40
   voices × 5 speeds = 47,200 clips. V1's raw audio was deleted, so V2
   regenerates the *entire* corpus with V1's exact protocol — which is what
   keeps V2's EVAL-SYN directly comparable to v1i's 0.8618.
2. **Split:** speaker-disjoint train/eval (20/20 voices), decided before
   generation.
3. **Selection:** EVAL-SYN only, pre-registered; 4-way Run A (v2cnn/bcresnet
   × 11c/10c), 30 epochs, batch 32, class weights, MUSAN noise.
4. **Honesty check:** isolated EVAL-REAL on authentic VCM audio (val∩test
   speakers = 0, benchmark ∩ val/test = 0).
5. **Deployment:** ONNX export + feature-parity gate + Pi demo app;
   wake word (2-class gate) is planned, not yet in the demo build.
6. **Known gap, stated up front:** v1i's EVAL-REAL was 0.2289 vs 0.8618
   SYN — a ~63-point domain gap from synthetic training data. That gap is
   *the* research question of the next phase (EVAL-REAL is the target:
   beat 0.2289). We don't hide it; we have it measured, isolated, and on
   the roadmap.

## 6. The hard questions — and the answers
**Q: Why not just use Whisper + an LLM? Everyone has that now.**
A: That's a 0.5–5 GB cascade with second-scale latency, a cloud or a phone
behind it, and the slot step ("twenty five" → 25 → timer) is exactly where
cascades fumble. We need intent+slot, not transcript+reasoning. End-to-end
skips the lossy middle, runs in a few ms on a $60 board, and is fully
private. If the question is "does it scale to 10,000 commands?" — no, and
we never claimed it does; this is a *device* with a *locked* vocabulary.

**Q: 0.86 on the benchmark — is that actually good?**
A: Judge it against the constraints: 86.2% on *unheard voices* (speaker-
disjoint) with a 110K-param model at 3 s fixed window and 5 TTS speeds.
The honest number is the real-audio one (22.9%, v1i) — and that's the
measured domain gap we're attacking next, not a hidden failure.

**Q: Why is real-audio accuracy so much lower?**
A: Train is synthetic TTS (40 clean voices); EVAL-REAL is authentic VCM
recordings — different rooms, accents, background, phrasing drift. That
gap is the standard synthetic-to-real problem in speech, measured and
isolated in this project precisely so we can close it (real-audio
augmentation, more authentic data) instead of discovering it at demo time.

**Q: How do you know the Pi runs *the same* model as training?**
A: We don't trust it, we verify it. The mel is computed in numpy on the Pi
from the exact training window/filterbank shipped as .npy; a parity gate
recomputes numpy-mel vs torch-mel and ONNX vs torch on real clips and
requires <1e-3 on all three before anything ships. A feature mismatch is
the #1 silent killer of on-device accuracy — so it's gated, not assumed.

**Q: Why ONNX and not TFLite or raw C?**
A: ONNX covers our whole op set (conv/pool/BN/linear — even the
SubSpectralNorm, which is just reshape+BN+reshape) at opset 17,
onnxruntime runs on every Pi CPU, and the export is a 10-line wrapper.
TFLite would force a second model definition; the payoff isn't there at
this op complexity.

**Q: Why not TensorRT / a NPU?**
A: No ARM/TensorRT path for this model class, and we don't need it:
110K params × 3 s clip is single-digit ms on a Pi 5 CPU. Adding an NPU
toolchain would multiply the deploy surface for ~0 ms of latency.

**Q: What's the wake word? The 09-24 protocol requires one (music plays →
wake → volume drops to 5%).**
A: In scope, separate stage: a tiny 2-class wake gate in front of the VCM
loop, trained in both architectures (v2cnn + bcresnet, winner by the same
protocol), same TTS pipeline, fully on-device (no vendor dep — Porcupine was
the alternative and vetoed for the black box). VAD is not a model either:
after wake, record until ~600 ms of RMS silence, right-pad to 3 s. The
current demo build is RMS-gated (fixed 3 s windows); the wake gate + silence
endpointing is the next demo increment.

**Q: Where does the slot value actually come from in the audio?**
A: The last spoken word. We probed it empirically in V1 (`_a2_cellprobe`):
for every class the slot word's onset falls in the last ~1.3 s of the
window, while the verb sits in cells 0–3. So each slot head reads only the
tail — which also stops it latching onto the shared "set/call/play"
boilerplate.

**Q: What happens when confidence is low / the phrase is unseen?**
A: The app gets `cmd + conf`; the actuator applies a confidence threshold
and a confirmation path (say it again). Unknown speech lands in the
closest class — the threshold + cooldown in the demo loop is the guard,
and the wake gate (next stage) removes most non-command audio before it's
scored.

**Q: 40 voices — isn't that too few?**
A: For the claim we make (generalization across *accent* and *speaker*,
measured speaker-disjoint), 40 across 10 accents is the budget that fits a
single HPC TTS pass (~12 h at current edge-tts throughput). The protocol
scales linearly — doubling voices is a config change, not a redesign.
What we're validating is the *method* (disjoint protocol, slot design,
parity gate), which is voice-count independent.

**Q: Why regenerate the whole V1 dataset? Wasted work?**
A: V1's raw audio was deleted; keeping V2's SYN eval comparable to v1i's
0.8618 *requires* the identical protocol (40 voices, 20/20 split, 5
speeds). Regenerating with the same recipe is what makes "0.8618 → X" a
honest comparison instead of an artifact of a different dataset.

**Q: What would you change with 10× the data / compute?**
A: Real-audio first (it's the measured bottleneck), then more voices, then
streaming (sliding windows) for lower latency. The architecture budget
stays — the constraint is the Pi, not the HPC.

**Q: How is this different from "just ask Google/Alexa"?**
A: Three words: latency, privacy, ownership. Cloud assistants: 300 ms–2 s
round trip, your audio leaves the house, and you're rented. This: a few ms
on the device, audio never leaves the room, and every model + dataset +
script is ours — reproducible from this repo.

**Q: Walk me through a false positive. What's the worst case?**
A: Worst case: background speech hits a high-confidence false command (a
TV says "play music"). Mitigations in order: wake gate (planned) →
confidence threshold → confirmation for high-consequence actions (calls) →
the app logs every fire with confidence so false positives are *measured*,
not anecdotal.

**Q: Why 11 classes and not fewer?**
A: The 10-class merge is the ablation — Run A trains both taxonomies so
the cost of splitting weather/time is a measured number, not a debate.
The taxonomy is locked to what the demo device must actually do.

## 7. Demo script (60 seconds — rehearse it)
1. **Power the Pi** (USB mic plugged in). Open the Pi dashboard
   (http://me2pi.local:8080) on a phone on the same WiFi — one tile per
   command, all idle.
2. Say: **"set the timer for twenty five minutes"** → timer tile animates
   to 25 min; the terminal shows `set_timer / "twenty five" / conf 0.9x`.
3. Say: **"dim the lights to forty percent"** → slider slides to 40%.
4. Say: **"play jetlag"** → player tile: *jetlag* — the song slot is exact,
   not "any".
5. Show the terminal: every fire is one JSON line with confidence and
   inference time (tens of ms on Pi 4, single-digit on Pi 5).
6. Close with the one-liner: **"None of that audio left this room."**

## 8. Honest limitations (volunteer these yourself)
- Real-audio accuracy is the weak point (v1i: 22.9%) — the synthetic→real
  domain gap; it's the explicit next-phase target.
- No wake word in the current demo build (RMS gate); the 2-class gate is
  designed, not yet trained.
- Fixed 3 s window: the phrase must finish inside the window; right-
  alignment (left-pad silence) protects short phrases, long pauses at the
  start are safe, but a phrase starting 3 s before you stop is truncated.
- Closed vocabulary by design (236 phrases) — open requests are out of
  scope, say so plainly if asked.
- 10 accents / 40 voices; unusual accents score lower — which is exactly
  what the speaker-disjoint eval measures.
- One near-field mic (0.3–3 m), no beamforming / echo cancellation in the
  demo path.

## 9. Glossary (30-second definitions)
- **log-mel (80×150):** the standard audio "image" — 80 frequency bands
  over 150 time frames (20 ms each) of a 3 s clip, in log power.
- **ONNX:** open neural-network exchange format; torch → ONNX so a 433 KB
  file (not a 2 GB torch install) runs on the Pi.
- **Opset 17:** the ONNX operator-version level we export at.
- **Speaker-disjoint:** train and eval audio come from different people;
  the model can't pass by memorizing voices.
- **Slot head:** one small linear layer per parametric class predicting
  the parameter value (the last word) from the spectrogram tail.
- **BC-ResNet:** "Broadcasted Residual" keyword-spotting network — 1D
  temporal convolutions over frequency sub-bands (arXiv 2106.04140).
- **SubSpectralNorm:** batch norm over grouped frequency sub-bands (part
  of BC-ResNet; exports to ONNX as reshape→BN→reshape).
- **MUSAN:** real-world noise corpus (babble, music, ambient) used to
  augment training at random SNR.
- **EVAL-SYN / EVAL-REAL:** synthetic speaker-disjoint eval (selection)
  vs authentic VCM eval (isolated honesty check).
- **VAD:** voice-activity detection — decides when to score (RMS in the
  demo build; the wake gate is the upgrade).
- **Parity gate:** pre-ship check that the Pi's numpy mel and ONNX outputs
  match training torch within 1e-3 on real clips.
- **Class weights:** up-weighting rarer classes in the loss so 119 slot
  values / 11 classes don't bias toward the frequent ones.
- **Right-aligned window:** audio sits at the right of the 3 s window;
  silence pads the left — so the last word (the slot) is always inside.
