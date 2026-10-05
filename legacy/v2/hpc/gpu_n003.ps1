# n003 GPU status
$ErrorActionPreference = 'Continue'
$sa = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n003.ai.internal')
$cmd = 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader'
& ssh @sa $cmd 2>&1 | Select-String -NotMatch '^#|^\s*#|^\s*$'
