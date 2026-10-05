# n002 check + kill n003 broken instances
$ErrorActionPreference = 'Continue'
$sa2 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n002.ai.internal')
$sa3 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n003.ai.internal')
Write-Output '===== n002 (retry) ====='
$c = 'echo ---GPUS---; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; echo ---11c---; tail -n 8 ~/vcm/overnight/train_v2a_bcresnet_11c.log 2>/dev/null; echo ---10c---; tail -n 8 ~/vcm/overnight/train_v2a_bcresnet_10c.log 2>/dev/null; echo ---PROCS---; ps -ef | grep -e train_v2.py -e overnight_pipeline2.sh | grep -v -e grep -e ''bash -c'''
& ssh @sa2 $c 2>$null | Select-Object
Write-Output '===== kill n003 pipeline2 instances ====='
& ssh @sa3 'pkill -f overnight_pipeline2.sh; sleep 1; ps -ef | grep overnight_pipeline2.sh | grep -v -e grep -e ''bash -c'' | wc -l' 2>$null | Select-Object
Write-Output '===== kill n002 pipeline2 instances ====='
& ssh @sa2 'pkill -f overnight_pipeline2.sh; sleep 1; ps -ef | grep overnight_pipeline2.sh | grep -v -e grep -e ''bash -c'' | wc -l' 2>$null | Select-Object
