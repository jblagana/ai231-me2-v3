# Overnight HPC pipeline — status poller (n002 TTS + repo layout)
$ErrorActionPreference = 'Continue'
$sa = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n002.ai.internal')
$cmd = 'echo === TTS-PROC ===; ps -o pid,etime,%cpu,%mem,cmd -p 3967973 2>/dev/null || echo PROC-GONE; echo === TTS-LOG-TAIL ===; tail -n 15 ~/vcm/tts_v2.log; echo === CLIP-COUNT ===; find ~/vcm/data/raw_v2 -name "*.mp3" 2>/dev/null | wc -l; echo === REPO ===; ls ~/ai231_me2_v2/; echo === TOOLS ===; ls ~/ai231_me2_v2/tools/ 2>/dev/null; echo === VCM-DIR ===; ls ~/vcm/ | head -30'
& ssh @sa $cmd 2>&1 | Select-String -NotMatch '^#|^\s*#|^\s*$'

