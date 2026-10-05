# ACTUATION — from JSON to things

The model only emits JSON (`pi_demo.py`). Everything after the JSON is
actuation: app logic on the Pi. Design goal — **one process, zero cloud,
phone is just a browser tab**.

## Architecture

```
USB mic ──► pi_app.py (single process on the Pi)
              │  1. capture 3.0 s → numpy mel → ONNX → {cmd, slot, conf}
              │  2. guard: conf ≥ 0.50, 2 s debounce on same (cmd, slot)
              │  3. dispatch (table below) → device state
              └─► stdlib http.server :8090 → tiles UI + SSE event stream
                     ▲
             phone/laptop browser at http://me2pi.local:8090
```

- `pi_app.py` = `pi_demo.py`'s capture+infer as functions + dispatch + the
  tiny web server (stdlib `http.server` + one static page; no pip deps beyond
  the demo's).
- `pi_demo.py --post http://127.0.0.1:8090/event` stays as the headless test
  path — the app accepts the same JSON line over HTTP, so the demo works
  with or without the UI running.
- Network is observation-only. The model, features, and all state are on the
  Pi; cutting the Pi's network changes nothing about what it can do.

## Dispatch table (11 classes, 119 slot values)

| cmd | slot vocab (n) | actuator (demo build) | UI tile |
|---|---|---|---|
| `play_music` | 5 songs + `any` (6) | mpv plays the 5 local rips; `any` → random; same-song repeat → no-op | now-playing + cover |
| `media_control` | pause/stop/next/prev/vol_up/vol_down… | mpv IPC (unix socket) — pause/stop/seek/volume on the player | transport bar |
| `control_lights` | on / off | software state machine (simulated lamp). Optional real upgrade: GPIO relay on the Pi 5 (noted, not built) | lamp on/off |
| `dim_lights` | up/down + 10 explicit % (12) | explicit % → set; up/down → ±30, clamp [10,100] | brightness bar |
| `set_timer` | 24 minute values | countdown thread; fires a bell wav (`aplay`) + flash | countdown chip |
| `set_alarm` | 48 clock values (h m am/pm) | 1 Hz scheduler check; bell at fire time | alarm list |
| `set_temperature` | 13 setpoints | simulated HVAC — stores target, shows delta to "current" | thermostat dial |
| `set_reminder` | 10 values (times + phrases) | schedule list (max 20); UI + bell at fire time | reminder list |
| `ask_time` | — | Pi RTC → text answer chip + short beep | clock chip |
| `ask_weather` | — | **no network by design** → honest fallback chip: "standalone device — no weather feed" (the VCM recognized the command; the data source doesn't exist on-device) | info chip |
| `make_call` | 6 contacts | no dialer on the Pi → "dialing mom…" chip + beep. Real bridge (SIP / phone-app intent) is out of scope; the command+contact were correctly recognized and routed | call chip |

Design stance: every class produces a *visible* effect (a tile, a sound, or
an honest "I can't, and why" chip). Nothing silently no-ops — the defense
story is "the model's output is exactly what you see".

## Guardrails (app side)

- **Confidence gate:** `conf < 0.50` → ignored, logged, shown as a faint
  "heard something (0.31)" chip. Matches V1's two-stage threshold practice.
- **Debounce:** identical `(cmd, slot)` within 2 s → dropped (echo/
  double-trigger protection).
- **One-shot semantics:** timers/alarms/reminders fire once; lists cap at 20.
- **No wake word** (known limitation, see `pi/README.md`): the loop is
  RMS-gated. Plan if demoed long: a tiny 2-class wake gate in front of the
  same pipeline.

## Demo choreography (90 s, 5 commands)

1. "turn on the lights" → lamp tile lights up (GPIO relay optional)
2. "set a timer for fifteen minutes" → live countdown on the phone
3. "play jetlag" → music from the Pi's speaker, now-playing tile
4. "what time is it" → clock chip
5. "call mom" → call chip

UI footer shows `infer_ms` on every event — the "100% on-device" claim,
live, on every command.
