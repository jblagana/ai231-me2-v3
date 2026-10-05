# Deploy Pi-side files to HPC repo (LF-normalized)
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\Jan\Desktop\AI231 ME2 V2\pi'
$remote = 'jan.rhey.lagana@n002.ai.internal'
foreach ($f in 'pi_demo.py', 'me2_ui.py', 'run_demo.sh') {
    $p = Join-Path $root $f
    $t = [IO.File]::ReadAllText($p)
    [IO.File]::WriteAllText($p, ($t -replace "`r`n","`n"))
    & scp -o BatchMode=yes -o ConnectTimeout=15 $p "${remote}:~/ai231_me2_v2/pi/"
    Write-Output "scp $f exit=$LASTEXITCODE"
}
