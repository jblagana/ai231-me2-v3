# RECOVER2 (one-shot, DO NOT RE-RUN): kill all instances, clear lock+assigned,
# deploy fixed pipeline2, launch ONE instance on n002 with all 4 RUNS.
# The fixed gpu_free will launch v2cnn-10c on n002 as soon as a GPU is free.
$ErrorActionPreference = 'Continue'
$sa2 = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR', 'jan.rhey.lagana@n002.ai.internal')
$sa3 = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR', 'jan.rhey.lagana@n003.ai.internal')

Write-Output '===== 1. kill instances (both nodes) ====='
& ssh @sa3 'pkill -f ''[o]vernight_pipeline2.sh''; true'
& ssh @sa2 'pkill -f ''[o]vernight_pipeline2.sh''; true'
Start-Sleep -Seconds 3
& ssh @sa3 'echo n003-left: $(ps -ef | grep [o]vernight_pipeline2.sh | wc -l)'
& ssh @sa2 'echo n002-left: $(ps -ef | grep [o]vernight_pipeline2.sh | wc -l)'

Write-Output '===== 2. clear lock + assigned, deploy fixed script ====='
$t = [IO.File]::ReadAllText('C:\Users\Jan\hpc_overnight\pipeline2.sh')
$b2 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($t -replace "`r`n","`n")))
if ($b2 -match '[^A-Za-z0-9+/=]') { throw 'BAD-B64' }
$dep = 'rm -rf ~/vcm/overnight/FINALIZE_LOCK; rm -f ~/vcm/overnight/assigned_ai-n00*.txt; echo ' + $b2 + ' | base64 -d > ~/vcm/overnight_pipeline2.sh; bash -n ~/vcm/overnight_pipeline2.sh && echo SYNTAX-OK; wc -l ~/vcm/overnight_pipeline2.sh; grep -c run_alive ~/vcm/overnight_pipeline2.sh'
& ssh @sa3 $dep

Write-Output '===== 3. single n002 instance, all 4 runs ====='
$c2 = 'RUNS=''v2a_bcresnet_11c:bcresnet:11: v2a_bcresnet_10c:bcresnet:10:--merge-10 v2a_v2cnn_11c:v2cnn:11: v2a_v2cnn_10c:v2cnn:10:--merge-10'' nohup bash ~/vcm/overnight_pipeline2.sh > ~/vcm/overnight2_recover_n002.log 2>&1 < /dev/null & echo N002-PID=$!; sleep 30; echo STATE=$(cat ~/vcm/overnight/STATE_ai-n002); ls -d ~/vcm/overnight/FINALIZE_LOCK 2>/dev/null || echo no-lock; tail -n 2 ~/vcm/overnight/overnight_ai-n002.log; ls ~/vcm/overnight/train_v2a_v2cnn_10c.pid 2>/dev/null || echo 10c-not-launched-yet; tail -n 1 ~/vcm/overnight/train_v2a_v2cnn_10c.log 2>/dev/null; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
& ssh @sa2 $c2
