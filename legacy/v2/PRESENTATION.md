# PRESENTATION — two slides

Two slides max. ~2 min talking per slide. `DEFENSE.md` is the Q&A backstop.

## Slide 1 — "The system" (what it is, why not ASR)

One-liner (title area):
**A 109,890-parameter voice-command model that runs 100% on a Raspberry Pi —
no cloud, no LLM, no ASR.**

Diagram (the center of the slide):

```
USB mic ─► 3.0 s window ─► numpy log-mel (80×150) ─► ONNX CNN (CPU)
   ─► 11 commands + 119 slot values ─► JSON ─► Pi app (lights/timer/music/…)
```

Margin numbers (small, four items):
- < 0.13 M params (pinned to the A2 budget; BC-ResNet variant 36,114 p)
- inference per command on Pi: ____ ms (fill after first Pi boot)
- training data: 47,200 TTS clips, 9 edge-tts voices; eval: 9,494 real
  clips, speaker-disjoint
- 11 command classes from real usage logs (259,164-command Alexa/GHome study)

Why-not-ASR one-liner:
"ASR adds a second model and a text bottleneck; VCM maps audio→command
directly — that's the whole point of the task spec."

## Slide 2 — "The results" (evidence + live demo)

Main table (fill after Run A finishes):

| model | params | EVAL-SYN ↑ | EVAL-REAL ↑ |
|---|---|---|---|
| v1i (our baseline) | — | 0.8618 | 0.2289 |
| A2 (paper) | 110k | 0.889 | — |
| **v2cnn-11c** | 109,890 | ___ | ___ |
| **bcresnet-11c** | 36,114 | ___ | ___ |

- Bar: success = beat v1i on BOTH axes (EVAL-SYN > 0.8618, EVAL-REAL > 0.2289).
- One line of taxonomy: 11 classes / 119 slot values — play_music,
  media_control, control_lights, dim_lights, set_timer, set_alarm,
  set_temperature, set_reminder, ask_time, ask_weather, make_call.
- Bottom strip (honest limitations, small): no wake word (RMS gate),
  actuation is simulated in the demo build, slot values spoken in
  word-form ("twenty five", not digits).

Ends with: **live demo** — Pi on the bench, phone showing the tiles; run the
90-second choreography from `ACTUATION.md`.

## Talk track

- Slide 1 (2 min): what → how → the four numbers → why-not-ASR.
- Slide 2 (2 min): the table → success bar → limitations honestly → "let me
  show you" → demo.
- Any question → `DEFENSE.md` (Q&A section, 30 pre-answered).
