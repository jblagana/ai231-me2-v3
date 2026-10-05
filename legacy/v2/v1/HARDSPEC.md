# HARDSPEC.md — target hardware (canonical)

Single source of truth for the RPi demo hardware. Other docs (PLAN, HANDOVER,
BENCHMARK) point here; if they disagree, this file wins.

## Target board — Raspberry Pi 4 Model B, 8 GB
- **Ownership: BORROWED** (2026-09-27) — not the boss's. Return it after the
  demo; keep the SD image + this repo as the durable artifacts.
- CPU: quad-core Cortex-A72 @ 1.5 GHz (ARM64)
- RAM: 8 GB LPDDR4
- GPU: VideoCore VI (irrelevant — CPU-only inference)
- Storage: microSD (the demo bottleneck — SD wear from continuous mic writes;
  use tmpfs for ring buffer, minimal logging)
- Power input: USB-C, **5 V / 3 A (15 W) required**
- Connectors: 2× USB 3.0, 2× USB 2.0, 2× micro-HDMI, 3.5 mm jack, GbE, GPIO 40-pin
- Inference runtime: **ONNX Runtime (CPU, ARM64)** — no TensorRT (no ARM64/RPi
  build; see WHYS.md)

## SD card
- SanDisk Ultra 64 GB, microSD — **bought 2026-09-27 (boss's)**
- 64 GB ≫ needed: OS Lite (~4 GB) + model (94K params, <1 MB ONNX) + dataset
  copy for eval — plenty of headroom

## Power supply
- **iPad 10th-gen 20 W USB-C adapter** (boss's, genuine Apple) + USB-C→C cable
- Verified 2026-09-27: Apple tech-specs page lists "20W USB-C Power Adapter"
  in the iPad 10 box (support.apple.com/en-us/111840); 20 W ≥ the 15 W the
  4B 8GB wants. Watch for brownout LED (yellow flash) if anything's off.

## Power bank (final demo)
- **Anker 737, 20,000 mAh / 74 Wh** (boss's; USB-C out 5V/3A = 15W, USB-A 18W)
- Viability: **viable** — 12–17 h at ~3.5 W, 9–13 h at ~5 W, 6–8 h at ~7.5 W
  (tool-computed 2026-09-27, 80–90% conversion efficiency)
- Airline-legal carry-on (74 Wh < 100 Wh limit)
- Charge to 100% morning of demo; iPad 20W adapter stays as wall backup
- Watch for: Pi 4 boot-peak draws can briefly exceed steady load — a
  mid-demo reboot = under-powered symptom → wall adapter, not bigger bank

## Audio (demo mic + speaker)
- **EMEET OfficeCore M1A USB conference speakerphone** (₱1,994 after voucher;
  Shopee, ordered 2026-09-28, arrival 28–29 Sept)
- Role: **both mic and speaker** for the demo — voice in for the VCM, response
  out as pre-recorded confirmation clips (no TTS/LLM on device).
- Connection: **USB, plug-and-play** (UAC class device — shows up as a plain
  ALSA capture/playback device, no drivers, no pairing).
- In the box: M1A, **USB-C→USB-A adapter**, USB-C→C cable, manual. The Pi 4B
  has 2× USB 3.0 + 2× USB 2.0 — the included C→A adapter plugs straight in;
  bus-powered, well within the 600 mA USB 2.0 budget.
- Mics: 2× omnidirectional, 360° pickup, full-duplex, built-in noise/echo
  cancellation — echo cancellation matters here (mic + speaker in one device,
  response playback while listening for the next command).
- Setup on the Pi: `sudo apt install libasound2-dev` → `arecord -l` to find the
  device → record a 3 s clip → confirm the capture loop in the demo app points
  at it. Test the night before demo day.

## Dev host (headless workflow)
- **Acer Nitro 5 (AN515-55)** (boss's Windows machine) — trains locally,
  flashes the SD card (Raspberry Pi Imager), SSHes into the Pi. No
  monitor/keyboard needed at the Pi: OS Lite + headless + SSH + Wi-Fi baked in
  at flash time.
  - CPU: Intel Core i5-10300H (4C/8T, 2.5 GHz) — plenty for CPU training of a
    94k-param CNN on ~27.6k clips
  - GPU: NVIDIA GTX 1650 Ti 4 GB (CUDA-capable; optional — model is small
    enough that CPU training is fine)
  - RAM: 16 GB
  - Storage: 256 GB WDC SN530 NVMe
  - (Read from the live machine, 2026-09-27.)

## HPC access (UP COE HPC — optional, for fast retrain / benchmarks)
- **Access:** `ssh jan.rhey.lagana@n00x.ai.internal`, x = **2** or **3**
  (key-based auth works from the Nitro 5; no password prompt).
- **Nodes (probed live 2026-09-27):**
  - Both: 8× A100 40 GB, 256 cores / 1 TB RAM, Ubuntu 24.04.
  - **Pick by live utilization, not habit** (boss rule 2026-09-27 —
    utilization fluctuates, re-check before assuming). One-line probe:
    `ssh jan.rhey.lagana@n00X.ai.internal 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'`
    → take the node with more idle A100s. (2026-09-27: n003 all 8 idle,
    n002 all 8 busy → n003.)
  - Local `/` is nearly full on both (n002: 333 GB free / 81%, n003:
    133 GB free / 93%) — **never store big things on `/`, keep everything
    in `~`**.
- **Storage:** `~` is on shared **JuiceFS** (1.0 PB total, ~907 TB free,
  shared across nodes) — the laptop's 256 GB SSD stops being the
  dataset/checkpoint bottleneck.
- **No Slurm — direct access only** (boss-confirmed 2026-09-27). There is no
  queue to submit to; you just run on the node you ssh into.
- **Role in ME2:** detour, not the deliverable — the story is "trains on a
  laptop, runs on a Pi." Use for fast retraining experiments and
  compute-side benchmarks only.

## Setup status (2026-09-27)
- [ ] OS flashed (Raspberry Pi OS Lite 64-bit, hostname `pi-me2`, SSH + Wi-Fi in Imager)
- [ ] First boot + `ssh pi@pi-me2.local` from Nitro 5
- [ ] ONNX Runtime smoke test on the Pi
