# Clean restart of wake trainings via run_wake_train.sh (single, ordered)
$ErrorActionPreference = 'Continue'
$sa2 = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR', 'jan.rhey.lagana@n002.ai.internal')

& ssh @sa2 'pkill -f train_[w]ake.py; sleep 2; rm -rf ~/vcm/data/noise16k_raw ~/ai231_me2_v2/runs/v2w; echo cleaned'

$p = 'C:\Users\Jan\hpc_overnight\run_wake_train.sh'
$t = [IO.File]::ReadAllText($p)
[IO.File]::WriteAllText($p, ($t -replace "`r`n","`n"))
& scp -o BatchMode=yes -o ConnectTimeout=15 $p 'jan.rhey.lagana@n002.ai.internal:~/ai231_me2_v2/run_wake_train.sh'
Write-Output "scp exit=$LASTEXITCODE"

& ssh @sa2 'bash ~/ai231_me2_v2/run_wake_train.sh; sleep 30; echo ---V2CNN---; tail -6 ~/ai231_me2_v2/runs/wake_v2cnn.log; echo ---BCRESNET---; tail -6 ~/ai231_me2_v2/runs/wake_bcresnet.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8'
