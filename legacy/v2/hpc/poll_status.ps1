# Poll overnight pipeline status on n003
param([int]$WaitSec = 900)
$ErrorActionPreference = 'Continue'
if ($WaitSec -gt 0) { Start-Sleep -Seconds $WaitSec }
$sa3 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n003.ai.internal')
$c = 'date; echo ---STATE---; cat ~/vcm/overnight/STATE 2>/dev/null; echo ---PIPELINE-LOG---; tail -n 4 ~/vcm/overnight/overnight.log 2>/dev/null; echo ---TTS---; grep -o ''cum ok=[0-9]*'' ~/vcm/tts_v2.log | tail -1; tail -n 1 ~/vcm/tts_v2.log; echo ---PROCS---; ps -ef | grep overnight_pipeline.sh | grep -v -e grep -e ''bash -c'' | wc -l; ps -ef | grep train_v2.py | grep -v -e grep -e ''bash -c'' | wc -l'
& ssh @sa3 $c 2>$null | Select-Object
