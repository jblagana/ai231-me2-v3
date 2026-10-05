# ME2 V2 overnight — local state tracker (recovery doc)
# Updated by muji. HPC-side state lives in the SHARED home (jfs) — same on n002+n003.

## DONE (2026-10-01 12:44 PST) — winner v2a_bcresnet_10c
- EVAL-SYN: v2cnn 11c 0.9201 / v2cnn 10c 0.9333 / bcresnet 11c 0.9446 /
  **bcresnet 10c 0.9514** (baseline 0.8618 -> +8.96pt). Winner picked by
  pre-registered rule (max cmd acc): v2a_bcresnet_10c, slot acc 0.5512.
- EXPORT PASS + PI-PATH VERIFY: PASS (worst onnx-vs-torch diff 2.19e-05, tol 0.001)
- On laptop: C:\Users\Jan\hpc_overnight\downloaded\
  - pi_pkg_v2.tar.gz (163K) -> pi_pkg_v2/v2a_bcresnet_10c/{model.onnx,
    mel.py, mel_fb.npy, mel_window.npy, meta.json, pi_demo.py}
  - OVERNIGHT_STATUS.md + eval_v2a_*.log (all 4)
- HPC leftovers (fine to keep): ~/vcm/overnight/ (logs, FINALIZE_LOCK,
  STATE files), ~/vcm/runs/v2a_* (4 checkpoints), ~/vcm/data/pi_pkg_v2/
- Pipeline instances: all exited (n002 finalizer STATE=done, n003 killed 12:01)

## Data (done, verified)
- TTS: 47,200 ok / 0 failed (04:41->07:27). Decode: 47,200 raw / 0 fails, train 23,600 / eval 23,600
- Integrity spot-check passed (16 kHz sizes, split counts exact)
- Bug fixed pre-launch: train_v2.py expected extract_slot's old bare-value API;
  slots.py returns (label, index) -> KeyError ('any', 0) killed all 4 first launches.
  Fix: unpack tuple, use sidx. Verified: selftest 179 phrases/119 values OK,
  dataset init on real data 23,600/23,600 (17,700 slot items). Crash logs: *.log.crash1

## Next: Pi handoff (boss does VPN off + hotspot on)
1. laptop: tar -xzf downloaded\pi_pkg_v2.tar.gz
2. scp -r pi_pkg_v2/v2a_bcresnet_10c jan@me2pi.local:~/me2/  (hotspot: 192.168.137.9)
3. Pi: python3 ~/me2/pi_demo.py --list-devices  (EMeet M1A index)
4. Pi: python3 ~/me2/pi_demo.py --dir ~/me2 --once --device <N>  -> first live fire
   (needs EMEET M1A plugged in — not done yet as of 10-01 morning)

## Next: HPC — wake gate A/B (boss-ratified 10-01: v2cnn AND bcresnet)
- Audit V1-era "hey boots" synthetic data (coverage/voices/negatives)
- make wake TTS if thin (same pipeline) -> train both archs 2-class ->
  eval -> export 2nd tiny ONNX -> loop: wake -> 600 ms RMS silence
  endpointing ("VAD") -> right-pad 3 s -> VCM infer
- Then: fill PRESENTATION.md results table

## Gotchas learned (this session)
- PS->ssh marshaling EATS double quotes in remote commands: use single quotes only
  (doubled '' inside PS single-quoted strings). Verified safe pattern in poll scripts.
- pgrep -f matches the ssh shell itself — use ps -ef | grep X | grep -v 'bash -c'
- pkill -f <pattern> also matches the wrapping bash -c (its cmdline contains the
  pattern text) -> kills the wrapper, no output. Use the bracket trick: pkill -f '[o]vernight_pipeline2.sh'
- bash syntax errors in the remote command (e.g. unquoted `echo (not started)`)
  abort the WHOLE command -> looks like "empty output" when 2>$null hides stderr.
  Test remote snippets; keep 2>$null OFF while debugging.
- Long/complex ssh commands in run_commands get mangled — always use local .ps1 files
- Inter-node ssh (n002->n003) does NOT work for interactive shells; but the HOME
  DIR is a shared jfs mount — deploy files once, and atomic mkdir locks work across nodes
- Parameter layer JSON-decodes backslashes: \r in text becomes CR; avoid backslashes
  in file content (base64 transfer for critical files)
- NEVER re-run a launch script that started pipeline instances (10:52 double-launch:
  harmless thanks to pidfile+lock guards, but don't). Name launchers uniquely + don't re-run.

