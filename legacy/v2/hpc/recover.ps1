# RECOVERY: kill instances, clear stale assigned files, deploy fixed pipeline2,
# relaunch one instance per node. 10c auto-launches on n003's first loop pass.
$ErrorActionPreference = 'Continue'
$sa2 = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR', 'jan.rhey.lagana@n002.ai.internal')
$sa3 = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR', 'jan.rhey.lagana@n003.ai.internal')

Write-Output '===== 0. pre-state ====='
& ssh @sa3 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8; ps -ef | grep [o]vernight_pipeline2.sh | wc -l'
& ssh @sa2 'ps -ef | grep [o]vernight_pipeline2.sh | wc -l'

Write-Output '===== 1. kill instances (both nodes) ====='
& ssh @sa3 'pkill -f ''[o]vernight_pipeline2.sh''; true'
& ssh @sa2 'pkill -f ''[o]vernight_pipeline2.sh''; true'
Start-Sleep -Seconds 3
& ssh @sa3 'echo n003-left: $(ps -ef | grep [o]vernight_pipeline2.sh | wc -l)'
& ssh @sa2 'echo n002-left: $(ps -ef | grep [o]vernight_pipeline2.sh | wc -l)'

Write-Output '===== 2. clear stale assigned files + deploy fixed script ====='
$t = [IO.File]::ReadAllText('C:\Users\Jan\hpc_overnight\pipeline2.sh')
$b2 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($t -replace "`r`n","`n")))
if ($b2 -match '[^A-Za-z0-9+/=]') { throw 'BAD-B64' }
$dep = 'rm -f ~/vcm/overnight/assigned_ai-n00*.txt; echo ' + $b2 + ' | base64 -d > ~/vcm/overnight_pipeline2.sh; bash -n ~/vcm/overnight_pipeline2.sh && echo SYNTAX-OK; grep -c "" ~/vcm/overnight_pipeline2.sh'
& ssh @sa3 $dep

Write-Output '===== 3. relaunch n003 (v2cnn pair; 10c auto-launches) ====='
$c3 = 'RUNS=''v2a_v2cnn_11c:v2cnn:11: v2a_v2cnn_10c:v2cnn:10:--merge-10'' nohup bash ~/vcm/overnight_pipeline2.sh > ~/vcm/overnight2_relaunch_n003.log 2>&1 < /dev/null & echo N003-PID=$!; sleep 20; tail -n 2 ~/vcm/overnight/overnight_ai-n003.log; ls ~/vcm/overnight/train_v2a_v2cnn_10c.pid 2>/dev/null && cat ~/vcm/overnight/train_v2a_v2cnn_10c.pid'
& ssh @sa3 $c3

Write-Output '===== 4. relaunch n002 (bcresnet pair; runs already done -> straight to finalize) ====='
$c2 = 'RUNS=''v2a_bcresnet_11c:bcresnet:11: v2a_bcresnet_10c:bcresnet:10:--merge-10'' nohup bash ~/vcm/overnight_pipeline2.sh > ~/vcm/overnight2_relaunch_n002.log 2>&1 < /dev/null & echo N002-PID=$!; sleep 20; tail -n 4 ~/vcm/overnight/overnight_ai-n002.log; ls ~/vcm/overnight/train_v2a_*.done 2>/dev/null; ls -d ~/vcm/overnight/FINALIZE_LOCK 2>/dev/null'
& ssh @sa2 $c2
