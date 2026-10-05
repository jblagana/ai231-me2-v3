# DEMO_UI_PLAN.md — Mock Device UI + RPi Control Architecture

**Status: DRAFT (2026-10-01)** · **Demo deadline: 2026-10-03**

## Goal

Show the VCM's output in a way a grader can *see* the command being executed.
The RPi is the brain; the browser is the "device." No cloud, no external
service — pure LAN over WiFi.

## Architecture (LAN-only, WiFi)

```
┌─────────────────────────────────────────────────────────────────┐
│  RPi 4B 8GB (pi-me2.local, 192.168.x.x)                        │
│                                                                 │
│  EMEET M1A ──USB──► ALSA capture                                │
│       │                                                         │
│       ▼                                                         │
│  Wake gate (2-class CNN, ONNX) ──► VCM (94K params, ONNX)      │
│       │                          │                              │
│       │ (no-wake: sleep)         ▼                              │
│       │                    (command, slot)                      │
│       │                          │                              │
│       │                          ▼                              │
│       │              ┌──────────────────────┐                   │
│       │              │  Device state dict   │                   │
│       │              │  {lights, temp,      │                   │
│       │              │   timer, alarm,      │                   │
│       │              │   music, call, ...}  │                   │
│       │              └──────────┬───────────┘                   │
│       │                         │                               │
│       │                         ▼                               │
│       │              FastAPI daemon (port 8080)                 │
│       │              ├─ GET /          → mock_device.html       │
│       │              ├─ GET /state     → JSON snapshot          │
│       │              └─ WS  /ws        → push state on change   │
│       │                                                         │
│       └── USB ──► EMEET M1A speaker (TTS confirmation clips)    │
└─────────────────────────────────────────────────────────────────┘
         │
         │  WiFi (same subnet, mDNS or static IP)
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Browser (phone / laptop / tablet on same WiFi)                 │
│                                                                 │
│  http://pi-me2.local:8080                                       │
│  ws://pi-me2.local:8080/ws                                      │
│                                                                 │
│  Renders the mock device UI, updates in real-time on WS push.   │
│  No auth, no cloud, no external dependency.                     │
└─────────────────────────────────────────────────────────────────┘
```

### Why WiFi (not wired / BLE / serial)

| option | verdict |
|---|---|
| **WiFi (LAN)** | ✅ the pick — RPi has GbE + WiFi; browser connects over the same subnet; mDNS (`pi-me2.local`) makes discovery trivial; no extra hardware |
| USB serial | ❌ browser can't talk serial without a web-serial polyfill + user gesture; fragile |
| BLE | ❌ RPi 4 has BLE 4.2 but the browser's Web Bluetooth API is Chrome-only and flaky on Android; overkill for a demo |
| HDMI + monitor | ❌ defeats the "phone shows the device" story; also the borrowed Pi has no monitor |

**WiFi is the only option that lets a phone/laptop on the same network
render the device state with zero extra hardware.** The EMEET is USB to the
RPi (mic + speaker); the RPi serves the page over its GbE/WiFi port.

### Connection details

- **RPi hostname:** `pi-me2` (set at flash time in Raspberry Pi Imager)
- **mDNS:** `pi-me2.local` — works on any OS (macOS, iOS, Windows 10+, Linux)
  without DNS config. If mDNS is flaky on the venue's WiFi, fall back to the
  RPi's static IP (set via `dhcpcd.conf`).
- **Port:** `8080` (FastAPI daemon)
- **WebSocket:** `ws://pi-me2.local:8080/ws` — server pushes full state JSON
  on every mutation; client just renders whatever arrives
- **No auth:** demo is on a closed LAN; adding auth is scope creep

## Commands → UI mapping

All 12 classes (10 original + `ask_weather` / `ask_time` split) need a UI
card. The split is **interactive** (user can click to change state, useful
for demo rehearsal) vs **display-only** (state changes via voice, user
watches the animation).

| # | class | slot values | UI card | interactive? | animation on command |
|---|---|---|---|---|---|
| 1 | `control_lights` | — | Toggle switch | ✅ click to toggle | switch flips, card glows |
| 2 | `dim_lights` | 20/30/50/70/80% | Brightness slider | ✅ drag slider | slider animates to new % |
| 3 | `set_temperature` | 18–28°C | Thermostat readout | ❌ voice-only | number ticks to new value |
| 4 | `set_timer` | 1–45 min | Countdown ring | ❌ voice-only | ring fills, countdown runs |
| 5 | `set_alarm` | 5am–9am, 6pm | Alarm time display | ❌ voice-only | time appears, card pulses |
| 6 | `set_reminder` | 10 values | Reminder list | ❌ voice-only | new item slides in |
| 7 | `make_call` | 6 contacts | Call status | ❌ voice-only | avatar + "Calling…" + green pulse |
| 8 | `play_music` | — | Music player (track, play/pause, progress) | ✅ play/pause/next/prev | track name appears, progress bar runs |
| 9 | `media_control` | pause/next/prev/vol±/mute | Volume bar + transport | ✅ vol slider | volume bar animates, track changes |
| 10 | `ask_weather` | — | Weather card (icon, temp, desc) | ❌ voice-only | icon + temp fade in |
| 11 | `ask_time` | — | Clock (live) | ❌ always-on | card highlights on query |
| 12 | `play_music` (title) | — | (same as #8) | — | — |

### Slot → UI value mapping (the "action layer")

The VCM outputs `(command, slot_label)`. The daemon maps slot_label →
concrete UI value:

```python
# daemon.py (RPi) — the action layer
SLOT_TO_UI = {
    "dim_lights": {
        "twenty": 20, "thirty": 30, "fifty": 50,
        "seventy": 70, "eighty": 80,
    },
    "set_temperature": {
        "eighteen": 18, "twenty": 20, "twenty two": 22,
        "twenty four": 24, "twenty five": 25,
        "twenty six": 26, "twenty eight": 28,
    },
    "set_timer": {  # minutes → seconds
        "one": 60, "two": 120, "five": 300, "ten": 600,
        "fifteen": 900, "twenty": 1200, "thirty": 1800,
        "forty five": 2700,
    },
    "set_alarm": {  # display strings
        "five am": "5:00 AM", "five thirty am": "5:30 AM",
        "six am": "6:00 AM", "seven am": "7:00 AM",
        "eight am": "8:00 AM", "nine am": "9:00 AM",
        "six pm": "6:00 PM",
    },
    "set_reminder": {  # display strings (as-is)
        "buy groceries": "Buy groceries",
        "call mom": "Call mom",
        "drink water": "Drink water",
        "pay the bills": "Pay the bills",
        "take out the trash": "Take out the trash",
        "water the plants": "Water the plants",
        "the meeting": "The meeting",
        "five pm": "5:00 PM",
        "tomorrow": "Tomorrow",
        "next week": "Next week",
    },
    "make_call": {  # display strings (as-is)
        "mom": "Mom", "dad": "Dad", "brother": "Brother",
        "sister": "Sister", "friend": "Friend", "the doctor": "The Doctor",
    },
}
```

Non-parametric classes (`control_lights`, `play_music`, `ask_weather`,
`ask_time`, `media_control`) have no slot — the action is determined by
the class alone (toggle, play, query, query, sub-command from phrase).

### `media_control` sub-command routing

`media_control` has no slot head — the sub-command (pause / next / vol up /
etc.) comes from the **phrase text**, not the model. For the demo, the
daemon keeps a small phrase→action map:

```python
MEDIA_ACTIONS = {
    "pause": "pause", "stop": "pause",
    "next song": "next", "skip": "next",
    "previous song": "prev",
    "volume up": "vol_up", "louder": "vol_up", "turn up the volume": "vol_up",
    "volume down": "vol_down", "quieter": "vol_down", "turn down the volume": "vol_down",
    "mute": "mute", "unmute": "unmute",
}
```

This is fine for the demo (the VCM already classified it as `media_control`;
the phrase text disambiguates the sub-action). In production you'd add a
sub-classifier or a slot head for media actions.

## RPi daemon (FastAPI, ~150 lines)

```python
# daemon.py — runs on the RPi, port 8080
# pip install fastapi uvicorn onnxruntime numpy sounddevice

import asyncio, json, time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np

app = FastAPI()

# ── device state ──
state = {
    "lights":  {"on": False, "brightness": 50},
    "temp":    {"value": 24, "mode": "Auto · Cooling"},
    "timer":   {"active": False, "remaining": 0, "total": 0},
    "alarm":   {"set": False, "time": None},
    "music":   {"playing": False, "name": "Nothing playing",
                "artist": "—", "progress": 0, "volume": 70},
    "call":    {"active": False, "contact": None},
    "weather": {"temp": 31, "desc": "Sunny · Cebu City", "icon": "☀️"},
    "time":    {"value": "—", "sub": "—"},
    "reminders": [],
}

# ── WebSocket clients ──
clients: list[WebSocket] = []

async def broadcast():
    payload = json.dumps(state)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.remove(ws)

# ── routes ──
@app.get("/")
async def index():
    return FileResponse("site/mock_device.html")

@app.get("/state")
async def get_state():
    return state

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    await ws.send_text(json.dumps(state))  # initial snapshot
    try:
        while True:
            await ws.receive_text()  # keepalive (client sends pings)
    except WebSocketDisconnect:
        clients.remove(ws)

# ── VCM inference (called by the audio loop) ──
def on_vcm_result(command: str, slot: str | None):
    """Mutate state + broadcast. Called from the audio thread."""
    # ... (mapping logic from SLOT_TO_UI above) ...
    # e.g.:
    # if command == "control_lights":
    #     state["lights"]["on"] = not state["lights"]["on"]
    # elif command == "dim_lights" and slot in SLOT_TO_UI["dim_lights"]:
    #     state["lights"]["brightness"] = SLOT_TO_UI["dim_lights"][slot]
    #     state["lights"]["on"] = True
    # ...
    asyncio.run_coroutine_threadsafe(broadcast(), loop)

# ── audio loop (separate thread) ──
# 1. read 3.0 s window from EMEET (sounddevice, 16 kHz, mono)
# 2. run wake gate → if no-wake, continue
# 3. run VCM → (command, slot)
# 4. call on_vcm_result(command, slot)
# 5. play TTS confirmation clip via EMEET speaker
```

### Audio loop details

- **Capture:** `sounddevice.InputStream(samplerate=16000, channels=1,
  blocksize=480)` — 480 samples = 30 ms; ring buffer of 100 blocks = 3.0 s
- **VAD / endpointing:** simple energy threshold (RMS > 0.01 for > 300 ms
  = speech start; RMS < 0.005 for > 500 ms = speech end). No external VAD
  library — keep deps minimal.
- **Wake gate:** runs on every 3.0 s window (or every 500 ms if you want
  lower latency — the 2-class CNN is ~0.5 ms on the Pi).
- **VCM:** runs once per detected utterance (after endpointing). ~1 ms on
  the Pi for 94K params.
- **TTS confirmation:** pre-recorded clips (one per command class, ~1 s
  each). No TTS engine on the Pi — just `sounddevice.play()` with the
  pre-recorded WAV.

### Dependencies on the RPi

```
fastapi
uvicorn[standard]
onnxruntime        # CPU EP, ARM64
numpy
sounddevice
```

That's it. No PyTorch, no TensorFlow, no cloud SDK. Total install ~80 MB.

## Browser client (the mock device page)

Already built: `site/mock_device.html` (self-contained, no external deps).

### How it connects

```javascript
// In mock_device.html — the WS client (add this to the <script> block)
const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (e) => {
    const newState = JSON.parse(e.data);
    // merge into local state + re-render all cards
    Object.assign(state, newState);
    renderAll();
};
ws.onclose = () => {
    // show "disconnected" banner, retry every 3 s
    setTimeout(() => location.reload(), 3000);
};
// ping every 5 s to keep the connection alive
setInterval(() => ws.readyState === 1 && ws.send("ping"), 5000);
```

### What the browser does NOT do

- No audio capture (the RPi handles all audio)
- No model inference (the RPi handles all ML)
- No cloud calls (weather/time are mocked or fetched by the RPi daemon)
- No auth (closed LAN)

The browser is a **pure renderer** — it displays whatever the RPi pushes.

## Weather / Time (the "API call" question)

The PLAN.md already notes: "no cloud constrains the VCM inference, not the
action side." For the demo:

- **`ask_time`:** the RPi daemon reads its own clock (`datetime.now()`).
  No API call. The browser's clock is also live, but the daemon's response
  is what triggers the card highlight.
- **`ask_weather`:** two options:
  1. **Mock (recommended for demo):** the daemon returns a fixed
     "31°C, Sunny, Cebu City" — no API key, no network dependency,
     no failure mode. The grader sees the command → UI response loop.
  2. **Real API (if time permits):** the daemon calls
     `wttr.in/Cebu?format=j1` (free, no API key, HTTP GET). Adds a
     network dependency + a failure mode (venue WiFi might block it).

**Recommendation: mock the weather for the demo.** The point is the VCM →
UI loop, not the weather API. One line in the writeup: "weather response
is mocked for the demo; a production deployment would call a weather API
on the RPi (outside the VCM boundary)."

## Demo flow (what the grader sees)

1. **Setup (before grader arrives):**
   - RPi boots, daemon starts, EMEET connected
   - Phone/laptop opens `http://pi-me2.local:8080`
   - Mock device page loads, shows all cards in default state
   - Command log is empty

2. **Live demo (grader speaks to the EMEET):**
   - "Turn on the lights" → lights card flips to ON, glows blue
   - "Dim the lights to seventy percent" → brightness slider animates to 70%
   - "Set the temperature to twenty two degrees" → thermostat ticks to 22°C
   - "Set a timer for five minutes" → timer ring starts counting down
   - "Play music" → track name appears, progress bar runs, play button active
   - "Volume up" → volume bar animates up
   - "Call mom" → call card shows "Mom · Calling…" with green pulse
   - "What's the weather" → weather card highlights, shows 31°C Sunny
   - "What time is it" → time card highlights

3. **Command log** (bottom of the page) records every command + slot in
   real-time — the grader can see the exact (command, slot) pairs the VCM
   produced.

## What's NOT in scope (demo)

- Real hardware control (GPIO relays for lights, MQTT for thermostat) —
  the mock device page IS the device for the demo
- Multi-user / auth — single grader, closed LAN
- TTS on the RPi — pre-recorded confirmation clips only
- Cloud weather API — mocked
- Real phone calling — the call card is a visual mock (no actual call placed)

## Files to build

| file | location | purpose |
|---|---|---|
| `daemon.py` | repo root (or `src/daemon.py`) | FastAPI daemon: state dict, WS broadcast, VCM hook, audio loop |
| `mock_device.html` | `site/` (✅ already built) | Browser client: renders state, WS client, simulate buttons |
| `confirmations/` | `data/` (gitignored) | Pre-recorded TTS WAV clips (one per command class) |
| `requirements_pi.txt` | repo root | `fastapi uvicorn[standard] onnxruntime numpy sounddevice` |
| `start_demo.sh` | repo root | One-liner: `uvicorn daemon:app --host 0.0.0.0 --port 8080` |

### `daemon.py` build checklist

- [ ] FastAPI app + state dict + WS endpoint
- [ ] `on_vcm_result()` with full SLOT_TO_UI mapping
- [ ] Audio loop: sounddevice InputStream → ring buffer → VAD → wake gate → VCM
- [ ] TTS confirmation playback (sounddevice.play)
- [ ] Weather mock (fixed response)
- [ ] Time (datetime.now)
- [ ] Timer countdown thread (decrement state["timer"]["remaining"] every 1 s, broadcast)
- [ ] `start_demo.sh` + `requirements_pi.txt`
- [ ] Test on the RPi: `ssh pi@pi-me2.local`, `python daemon.py`, open `http://pi-me2.local:8080`

## Timeline (2 days to deadline)

| day | task |
|---|---|
| 10-01 (today) | ✅ Mock device page built + browser-verified. Write this plan. |
| 10-02 | Build `daemon.py` (FastAPI + WS + state + VCM hook). Test on RPi. Record TTS confirmation clips. |
| 10-03 (deadline) | Full end-to-end test: mic → wake → VCM → state → WS → browser. Rehearse demo flow. Buffer for venue WiFi issues. |
