# Verify v2 training health on both nodes
$ErrorActionPreference = 'Continue'
$sa2 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n002.ai.internal')
$sa3 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n003.ai.internal')
Write-Output '===== n002 ====='
$c = 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; echo ---TRAIN-LOG---; tail -n 8 ~/vcm/overnight/train_v2a_bcresnet_11c.log 2>/dev/null; echo ---PROC---; ps -ef | grep train_v2.py | grep -v -e grep -e ''bash -c'''
& ssh @sa2 $c 2>$null | Select-Object
Write-Output '===== n003 ====='
$c3 = 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; echo ---TRAIN-LOG---; tail -n 8 ~/vcm/overnight/train_v2a_v2cnn_11c.log 2>/dev/null; echo ---PROC---; ps -ef | grep train_v2.py | grep -v -e grep -e ''bash -c'''
& ssh @sa3 $c3 2>$null | Select-Object
