# n003 GPU occupancy investigation
$ErrorActionPreference = 'Continue'
$sa3 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n003.ai.internal')
$c = 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; echo ---COMPUTE-APPS---; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | head -25; echo ---MY-PROCS---; ps -u $(whoami) -o pid,etime,cmd | grep -e train_v2 -e make_tts -e overnight | grep -v grep; echo ---FULL-PIPELINE-LOG---; cat ~/vcm/overnight/overnight.log'
& ssh @sa3 $c 2>$null | Select-Object
