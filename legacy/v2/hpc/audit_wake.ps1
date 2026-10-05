# scp the audit script to HPC (LF-normalized), then run it
$ErrorActionPreference = 'Continue'
$t = [IO.File]::ReadAllText('C:\Users\Jan\hpc_overnight\audit_wake.sh')
[IO.File]::WriteAllText('C:\Users\Jan\hpc_overnight\audit_wake.sh', ($t -replace "`r`n","`n"))
$sa3 = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR', 'jan.rhey.lagana@n003.ai.internal')
& scp -o BatchMode=yes -o ConnectTimeout=15 'C:\Users\Jan\hpc_overnight\audit_wake.sh' 'jan.rhey.lagana@n003.ai.internal:~/vcm/audit_wake.sh'
Write-Output "scp exit=$LASTEXITCODE"
& ssh @sa3 'bash ~/vcm/audit_wake.sh'

