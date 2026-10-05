# Deploy overnight pipeline to n003 via base64 (immune to all quoting layers)
$ErrorActionPreference = 'Continue'
$t = [IO.File]::ReadAllText('C:\Users\Jan\hpc_overnight\pipeline.sh')
$t = $t -replace "`r`n", "`n"
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($t))
if ($b64 -match '[^A-Za-z0-9+/=]') { throw 'BAD-B64' }
Write-Output ("B64-LEN=" + $b64.Length)
$sa3 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n003.ai.internal')
$c = 'echo ' + $b64 + ' | base64 -d > ~/vcm/overnight_pipeline.sh; bash -n ~/vcm/overnight_pipeline.sh >/tmp/nchk.txt 2>&1; echo BASHN=$? >>/tmp/nchk.txt; head -2 ~/vcm/overnight_pipeline.sh >>/tmp/nchk.txt; cat /tmp/nchk.txt'
& ssh @sa3 $c 2>$null | Select-Object
$c2 = '~/.conda/envs/vcm/bin/python -c ''import torch, imageio_ffmpeg'' && echo TORCH-FF-OK; ~/.conda/envs/vcm/bin/python -c ''import onnxruntime'' && echo ORT-OK || echo ORT-MISSING; pgrep -af overnight_pipeline || echo NO-EXISTING-RUN'
& ssh @sa3 $c2 2>$null | Select-Object


