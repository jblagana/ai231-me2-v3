# Run sanity_v2.py on HPC via base64 (heredoc over ssh proved flaky)
$ErrorActionPreference = 'Continue'
$sa3 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n003.ai.internal')
$t = [IO.File]::ReadAllText('C:\Users\Jan\hpc_overnight\sanity_v2.py')
$b = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($t -replace "`r`n","`n")))
if ($b -match '[^A-Za-z0-9+/=]') { throw 'BAD-B64' }
$c = 'echo ' + $b + ' | base64 -d > ~/ai231_me2_v2/sanity_v2.py; cd ~/ai231_me2_v2 && ~/.conda/envs/vcm/bin/python sanity_v2.py 2>&1 | tail -n 15'
& ssh @sa3 $c 2>$null | Select-Object
