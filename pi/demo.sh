#!/bin/bash
# ME2 Pi demo — start / stop / watch (run from anywhere)
#   ~/me2/demo.sh start    wake loop + UI + recording (rec/)
#   ~/me2/demo.sh stop
#   ~/me2/demo.sh status
#   ~/me2/demo.sh log      last 20 log lines (fires + wake confs)
#   ~/me2/demo.sh rec      list recorded WAVs (newest first)
cd ~/me2 || exit 1
case "$1" in
  start)
    pkill -f pi_demo.py 2>/dev/null; pkill -f me2_ui.py 2>/dev/null; sleep 1
    source .venv/bin/activate
    nohup ./run_demo.sh 1 > demo.log 2>&1 &
    disown
    sleep 3
    tail -4 demo.log
    echo "UI:      http://me2pi.local:8330"
    echo "stop:    ~/me2/demo.sh stop"
    ;;
  stop)
    pkill -f pi_demo.py 2>/dev/null; pkill -f me2_ui.py 2>/dev/null; sleep 1
    pgrep -af "pi_demo|me2_ui" | grep -v pgrep || echo "stopped"
    ;;
  status)
    pgrep -af "pi_demo|me2_ui" | grep -v pgrep || echo "not running"
    echo ---
    tail -3 demo.log 2>/dev/null
    ;;
  log)
    tail -20 demo.log 2>/dev/null
    ;;
  rec)
    ls -lt rec/ 2>/dev/null | head -15 || echo "no recordings yet"
    ;;
  *)
    echo "usage: ~/me2/demo.sh {start|stop|status|log|rec}"
    ;;
esac