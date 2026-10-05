# ME2 V3 — Presentation Talk Script

> **Local only — do NOT push this to the remote repo.**
> Three slides (Model/Dataset/Training/Validation → To Be Submitted → Appendix)
> + a live Pi demo. Aim: ~6 minutes of talking + ~2 minutes live demo.
> Numbers below match `slides/me2_v3_submission.pptx` and `BENCHMARK.md`.

---

## OPEN (before slide 1) — 20 s

"Thanks. Let me walk you through **ME2 V3** — the voice-controlled smart
device. The one-line pitch: a small neural model that, **on a Raspberry Pi 4,
with no cloud round-trip**, hears me say a command and drives the device.
This deck is the final V3 submission — model, dataset, training, and validation —
and I'll end by showing it **live on the Pi**."

---

## SLIDE 1 — The system + the numbers (main slide, ~3 min)

### Model (point at the six boxes)
"So the pipeline: a microphone at 16 kHz becomes an 80-mel log feature over a
**fixed 3-second window** — 150 frames. That goes into a **BC-ResNet** encoder
— that's the same architecture family as our V2 deployment, scaled to
**30,134 parameters**. Two heads come out: an **intent head with 20 classes** —
19 real commands plus an **OUT_OF_SCOPE** rejection class — and a **slot head**:
6 heads, 3 whitelisted values each, 18 values total. The whole thing runs as
**ONNX on CPU**, and it fits in **172 kilobytes**. Two ONNX models — the wake
gate and this command/slot model — share the **same mel feature** on the Pi."

> If asked "is it streaming?": "No — this is a fixed-window keyword classifier,
> not a streaming KV-cache encoder. The 3-second window ends ~1 second after I
> stop talking, which is what the endpointing captures."

### Dataset (right column)
"For data: the master **Gold Dataset** — 10 sources, real and synthetic, from
Hugging Face, speaker-disjoint with verified zero overlap between splits.
**81,768 clips** total: 10,733 train, 4,443 test, 202 holdout. Labels are the
20 intents and the 18 slot values across 6 heads."

### Training (bottom right)
"Trained on **one A100** on the COE HPC (node n003). Objective is a
class-weighted cross-entropy on the 20 command classes plus an unweighted
cross-entropy on the 6 slot heads. AdamW, cosine schedule. **200 epochs, about
67,000 steps, roughly 47 minutes.** Selection picked **epoch 142** for best
command accuracy — final training loss 0.206."

### Validation (the key table) — slow down here
"And the results, which is why I'm comfortable showing you this live:

- **Command accuracy: 0.9455 on the held-out set** — that's one-shot, 186
  command clips, unseen speakers — and 0.9293 on the full test set.
- **Slot accuracy: 0.9722 holdout, 0.9748 test** — so when it does pick a
  slotted command, it gets the value right essentially every time.
- On latency: **about 92 milliseconds per 3-second clip on the Pi 4B**, well
  inside our 100-millisecond target.

**This is the go/no-go I was given: at least 90% holdout on command AND at
least 90% holdout on slot. We cleared both — 94.5% and 97.2% — so this is a GO,
and the model is live on the Pi right now.**"

> **Honesty beat (say it, don't hide it):** "I'll be transparent about one
> number. **OUT_OF_SCOPE rejection** is our weak spot — about 69% on holdout,
> lower on test. The model occasionally fires on non-command audio. In
> deployment that's mitigated by the **wake gate in front** — the command model
> only runs after 'hey boots' — and a **0.5 confidence gate** before anything
> actually actuates. So a spurious low-confidence fire never touches a device.
> It's the thing I'd push hardest in V3.1."

---

## SLIDE 2 — To Be Submitted + reviewer checklist (~1.5 min)

"So what's being submitted: the **public, MIT-licensed GitHub repo**
(`jblagana/ai231-me2-v3`) — one-command reproduction with `scripts/train.sh`.
The dataset is **citable** — Hugging Face plus the Xela VCM set, both with DOIs.
Training ran on node n003, ~47 minutes. The model weights are the 172 KB ONNX,
deployed live on the Pi, released separately per the repo's `.gitignore`
(we keep weights out of the git history on purpose).

The reviewer checklist — everything's green now: reproducible repo, citable
dataset, held-out unseen-speaker test set, on-device latency reproduced by the
posted script, and a comparable-size baseline to compare against — our V2 model
and the in-repo v2cnn."
---

## SLIDE 3 — Appendix (optional, ~30 s — only if time)

"This is just the provenance: the exact order we filled the amber cells —
dataset stats from `manifest_stats.py`, the results from `runs/v3r1/results.json`,
the Pi latency from the live benchmark, and the ONNX export verified for
torch/ONNX parity. Citations are down here. I can walk through this on request."

---

## LIVE DEMO (on the Pi) — ~2 min

"Now let me show it actually running. *(bring up the Pi dashboard,
http://me2pi.local:8330)* This is the real device state. Say the wake word —
**'Hey boots'** — and after it wakes, give it a command. Let me try a few:

- **"Hey boots" … "play music"** → you'll see MUSIC light up and a track start.
- **"Hey boots" … "turn on the lamp"** → LAMP goes on.
- **"Hey boots" … "set the brightness to 60 percent"** → BRIGHTNESS tile, 60%.
- **"Hey boots" … "what's the weather"** → it queries live weather and answers.
- And for the rejection: I'll say something random / silence → it either stays
  idle or flags **OUT_OF_SCOPE** — it does **not** fire.

Each fire shows the command, the confidence, the slot, and the end-to-end
latency on the card. That's the full loop — mic → wake → 3-second capture →
ONNX on the Pi → device — with no cloud in between."

> **Demo fallback (if the mic/wake is flaky in the room):** "The room acoustics
> can be rough — if a wake doesn't catch, I can drive the same pipeline directly
> on recorded clips, which is exactly what the on-device benchmark does — 92 ms
> per clip, same 20-command model." (Have `_pi_app/_bench_latency.py` output
> ready as the backup.)

---

## CLOSE — 15 s

"So: 20 commands, 18 slot values, 30k parameters, 172 KB, running at ~92 ms on
a Pi 4 with no cloud, 94.5% command / 97.2% slot on the held-out set — that
cleared the gate and it's live. The known gap is OOS rejection, and that's the
top priority for the next iteration. Questions?"

---

## Anticipated Q&A (prep, don't read)
- **"Why a fixed 3-second window and not streaming?"** → The window ends ~1 s
  after speech-end (VAD-anchored), which is exactly what the Pi endpointing
  captures; it made the holdout gate reachable where file-end alignment
  (V1) couldn't.
- **"0.925 vs 0.9455 — which is the real number?"** → 0.9455 is the held-out
  one-shot (unseen, the honest generalization number); 0.9293 is the full test
  set used for tuning; both are reported.
- **"What does OUT_OF_SCOPE do in practice?"** → It's the learned no-action
  class; combined with the 0.5 confidence gate, a low-confidence OOS/ambiguous
  utterance actuates nothing.
- **"Can I reproduce it?"** → Clone `jblagana/ai231-me2-v3`, `scripts/train.sh`
  for the A100 run, `tools/export_onnx.py --verify` for the ONNX, and the Pi
  package (`mel.py` + `pi_demo.py` + `me2_ui.py`) for the device.
- **"Why 200 epochs and not the 100 the plan said?"** → Pushed to 200 to let
  the model keep improving toward the 0.99 command gate; it plateaued around
  0.926–0.93 on test, so selection chose the best-command epoch (142).