# Deploy pipeline2 to n002+n003, kill old v1 (n003), launch with per-node RUNS
$ErrorActionPreference = 'Continue'
$t = [IO.File]::ReadAllText('C:\Users\Jan\hpc_overnight\pipeline2.sh')
$t = $t -replace "`r`n", "`n"
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($t))
if ($b64 -match '[^A-Za-z0-9+/=]') { throw 'BAD-B64' }
$sa2 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n002.ai.internal')
$sa3 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n003.ai.internal')
$deploy = 'echo ' + $b64 + ' | base64 -d > ~/vcm/overnight_pipeline2.sh; bash -n ~/vcm/overnight_pipeline2.sh && echo SYNTAX-OK'
Write-Output '===== deploy n002 ====='
& ssh @sa2 $deploy 2>$null | Select-Object
Write-Output '===== kill v1 + deploy n003 ====='
& ssh @sa3 ('kill 3396484 2>/dev/null; ' + $deploy) 2>$null | Select-Object
Write-Output '===== launch n003 (v2cnn pair) ====='
$c3 = 'RUNS=''v2a_v2cnn_11c:v2cnn:11: v2a_v2cnn_10c:v2cnn:10:--merge-10'' nohup bash ~/vcm/overnight_pipeline2.sh > ~/vcm/overnight2_launch_n003.log 2>&1 < /dev/null & echo N003-PID=$!; sleep 45; echo STATE=$(cat ~/vcm/overnight/STATE_ai-n003 2>/dev/null); ls ~/vcm/overnight/ 2>/dev/null | grep -e pid -e assigned; tail -n 4 ~/vcm/overnight/overnight_ai-n003.log'
& ssh @sa3 $c3 2>$null | Select-Object
Write-Output '===== launch n002 (bcresnet pair) ====='
$c2 = 'RUNS=''v2a_bcresnet_11c:bcresnet:11: v2a_bcresnet_10c:bcresnet:10:--merge-10'' nohup bash ~/vcm/overnight_pipeline2.sh > ~/vcm/overnight2_launch_n002.log 2>&1 < /dev/null & echo N002-PID=$!; sleep 45; echo STATE=$(cat ~/vcm/overnight/STATE_ai-n002 2>/dev/null); tail -n 4 ~/vcm/overnight/overnight_ai-n002.log'
& ssh @sa2 $c2 2>$null | Select-Object
