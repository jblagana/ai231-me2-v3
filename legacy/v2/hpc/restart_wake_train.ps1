# Redeploy fixed train_wake.py, sanity-check datasets, restart both trainings
$ErrorActionPreference = 'Continue'
$sa2 = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR', 'jan.rhey.lagana@n002.ai.internal')

$p = 'C:\Users\Jan\Desktop\AI231 ME2 V2\src\train_wake.py'
$t = [IO.File]::ReadAllText($p)
[IO.File]::WriteAllText($p, ($t -replace "`r`n","`n"))
& scp -o BatchMode=yes -o ConnectTimeout=15 $p 'jan.rhey.lagana@n002.ai.internal:~/ai231_me2_v2/src/train_wake.py'
$s = 'C:\Users\Jan\hpc_overnight\sanity_wake_ds.py'
$t = [IO.File]::ReadAllText($s)
[IO.File]::WriteAllText($s, ($t -replace "`r`n","`n"))
& scp -o BatchMode=yes -o ConnectTimeout=15 $s 'jan.rhey.lagana@n002.ai.internal:~/ai231_me2_v2/sanity_wake_ds.py'
Write-Output "scp exit=$LASTEXITCODE"

& ssh @sa2 'cd ~/ai231_me2_v2 && ~/.conda/envs/vcm/bin/python sanity_wake_ds.py'

& ssh @sa2 'pkill -f train_[w]ake.py; sleep 2; bash ~/ai231_me2_v2/run_wake_train.sh; sleep 30; echo ---V2CNN---; tail -4 ~/ai231_me2_v2/runs/wake_v2cnn.log; echo ---BCRESNET---; tail -4 ~/ai231_me2_v2/runs/wake_bcresnet.log'
