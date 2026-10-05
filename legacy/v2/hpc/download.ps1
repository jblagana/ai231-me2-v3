# Pull final artifacts to laptop (run when STATE_ALL=done / OVERNIGHT_STATUS.md exists)
$ErrorActionPreference = 'Continue'
$dst = 'C:\Users\Jan\hpc_overnight\downloaded'
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$sa3 = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'jan.rhey.lagana@n003.ai.internal')
Write-Output '===== check remote artifacts ====='
$c = 'ls -lh ~/vcm/data/pi_pkg_v2.tar.gz 2>/dev/null || echo NO-TARBALL; ls -l ~/vcm/overnight/OVERNIGHT_STATUS.md 2>/dev/null || echo NO-STATUS; echo STATE-ALL=$(cat ~/vcm/overnight/STATE_ALL 2>/dev/null || echo none)'
& ssh @sa3 $c
Write-Output '===== scp tarball ====='
& scp -o BatchMode=yes -o ConnectTimeout=15 'jan.rhey.lagana@n003.ai.internal:~/vcm/data/pi_pkg_v2.tar.gz' "$dst\pi_pkg_v2.tar.gz"
Write-Output "tarball exit=$LASTEXITCODE"
Write-Output '===== scp status + eval reports + winner history ====='
& scp -o BatchMode=yes -o ConnectTimeout=15 'jan.rhey.lagana@n003.ai.internal:~/vcm/overnight/OVERNIGHT_STATUS.md' "$dst\OVERNIGHT_STATUS.md"
& scp -o BatchMode=yes -o ConnectTimeout=15 "jan.rhey.lagana@n003.ai.internal:~/vcm/overnight/eval_*.log" "$dst\"
Write-Output '===== local result ====='
Get-ChildItem $dst | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize | Out-String
