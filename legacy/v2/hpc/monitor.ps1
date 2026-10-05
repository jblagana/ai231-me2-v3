# FAST status (no sleep)
$ErrorActionPreference = 'Continue'
$sa3 = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR', 'jan.rhey.lagana@n003.ai.internal')
$c = 'date +%T; echo ---STATE-n003: $(cat ~/vcm/overnight/STATE_ai-n003 2>/dev/null); echo ---STATE-n002: $(cat ~/vcm/overnight/STATE_ai-n002 2>/dev/null); echo ---STATE-ALL: $(cat ~/vcm/overnight/STATE_ALL 2>/dev/null || echo none); echo ---DONE-FILES: ; ls ~/vcm/overnight/train_v2a_*.done 2>/dev/null || echo none; echo ---LOCK: ; ls -d ~/vcm/overnight/FINALIZE_LOCK 2>/dev/null || echo none; echo ---TAIL-n003; tail -n 3 ~/vcm/overnight/overnight_ai-n003.log; echo ---TAIL-n002; tail -n 3 ~/vcm/overnight/overnight_ai-n002.log; echo ---TRAIN-TAILS; for n in v2a_v2cnn_11c v2a_v2cnn_10c v2a_bcresnet_11c v2a_bcresnet_10c; do echo "$n: $(tail -n 1 ~/vcm/overnight/train_$n.log 2>/dev/null || echo missing)"; done; echo ---EVALS; ls ~/vcm/overnight/eval_v2a_*.log 2>/dev/null || echo none'
& ssh @sa3 $c
