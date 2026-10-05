# LAUNCH v2 pipeline once per node (single instance each — DO NOT RE-RUN this file)
$ErrorActionPreference = 'Continue'
$sa2 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n002.ai.internal')
$sa3 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n003.ai.internal')
Write-Output '===== launch n003 (v2cnn pair) ====='
$c3 = 'RUNS=''v2a_v2cnn_11c:v2cnn:11: v2a_v2cnn_10c:v2cnn:10:--merge-10'' nohup bash ~/vcm/overnight_pipeline2.sh > ~/vcm/overnight2_launch_n003.log 2>&1 < /dev/null & echo N003-PID=$!; sleep 60; cat ~/vcm/overnight/STATE_ai-n003; tail -n 3 ~/vcm/overnight/overnight_ai-n003.log'
& ssh @sa3 $c3 2>$null | Select-Object
Write-Output '===== launch n002 (bcresnet pair) ====='
$c2 = 'RUNS=''v2a_bcresnet_11c:bcresnet:11: v2a_bcresnet_10c:bcresnet:10:--merge-10'' nohup bash ~/vcm/overnight_pipeline2.sh > ~/vcm/overnight2_launch_n002.log 2>&1 < /dev/null & echo N002-PID=$!; sleep 60; cat ~/vcm/overnight/STATE_ai-n002; tail -n 3 ~/vcm/overnight/overnight_ai-n002.log'
& ssh @sa2 $c2 2>$null | Select-Object
