#!/usr/bin/env bash
# me2_watchdog.sh — supervisor for the wake loop. The loop prints a
# heartbeat ('hb') every 15 s while alive; a wedged sounddevice read
# emits nothing forever, so demo.log's mtime freezes. If the log goes
# silent for SILENCE_S, kill the loop and let systemd Restart=always
# bring it back (fresh ALSA stream, wedge cleared).
set -u
LOG=/home/jan/me2/demo.log
PROC='pi_demo[.]py'
SILENCE_S=60
CHECK_S=20
while true; do
  sleep "$CHECK_S"
  # mtime of the log, as epoch seconds (0 if the file is gone).
  mt=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
  now=$(date +%s)
  age=$(( now - mt ))
  if [ "$age" -ge "$SILENCE_S" ]; then
    echo "[watchdog] $(date '+%H:%M:%S') demo.log silent ${age}s -> killing pi_demo for restart" >> /home/jan/me2/watchdog.log
    pkill -f "$PROC" 2>/dev/null
  fi
done
