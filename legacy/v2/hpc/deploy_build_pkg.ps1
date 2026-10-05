# Deploy build_pkg.sh to HPC (LF-normalized)
$ErrorActionPreference = 'Continue'
$p = 'C:\Users\Jan\hpc_overnight\build_pkg.sh'
$t = [IO.File]::ReadAllText($p)
[IO.File]::WriteAllText($p, ($t -replace "`r`n","`n"))
& scp -o BatchMode=yes -o ConnectTimeout=15 $p 'jan.rhey.lagana@n002.ai.internal:~/ai231_me2_v2/build_pkg.sh'
Write-Output "scp exit=$LASTEXITCODE"
