# Redeploy pipeline (fixed stall threshold) + restart on n003
$ErrorActionPreference = 'Continue'
$t = [IO.File]::ReadAllText('C:\Users\Jan\hpc_overnight\pipeline.sh')
$t = $t -replace "`r`n", "`n"
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($t))
if ($b64 -match '[^A-Za-z0-9+/=]') { throw 'BAD-B64' }
$sa3 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n003.ai.internal')
$c = 'echo ' + $b64 + ' | base64 -d > ~/vcm/overnight_pipeline.sh; bash -n ~/vcm/overnight_pipeline.sh && echo SYNTAX-OK; kill 3372986 2>/dev/null; sleep 1; nohup bash ~/vcm/overnight_pipeline.sh > ~/vcm/overnight_launch.log 2>&1 < /dev/null & echo RELAUNCH-PID=$!; sleep 4; echo STATE=$(cat ~/vcm/overnight/STATE 2>/dev/null); tail -n 3 ~/vcm/overnight/overnight.log'
& ssh @sa3 $c 2>$null | Select-Object

