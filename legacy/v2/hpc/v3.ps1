# v3 verify training health on both nodes
$ErrorActionPreference = 'Continue'
$sa2 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n002.ai.internal')
$sa3 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n003.ai.internal')
Write-Output '===== n002 ====='
$c = 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; echo ---11c---; tail -n 6 ~/vcm/overnight/train_v2a_bcresnet_11c.log 2>/dev/null; echo ---10c---; tail -n 6 ~/vcm/overnight/train_v2a_bcresnet_10c.log 2>/dev/null; echo ---PROCS---; ps -ef | grep train_v2.py | grep -v -e grep -e ''bash -c'' | wc -l'
& ssh @sa2 $c 2>$null | Select-Object
Write-Output '===== n003 ====='
$c3 = 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; echo ---11c---; tail -n 6 ~/vcm/overnight/train_v2a_v2cnn_11c.log 2>/dev/null; echo ---10c---; tail -n 6 ~/vcm/overnight/train_v2a_v2cnn_10c.log 2>/dev/null; echo ---PROCS---; ps -ef | grep train_v2.py | grep -v -e grep -e ''bash -c'' | wc -l; echo ---INSTANCES---; ps -ef | grep overnight_pipeline2.sh | grep -v -e grep -e ''bash -c'''
& ssh @sa3 $c3 2>$null | Select-Object
