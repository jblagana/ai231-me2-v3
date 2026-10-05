# Restart wake TTS: kill old run, clean dir, redeploy updated script, launch
$ErrorActionPreference = 'Continue'
$sa2 = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR', 'jan.rhey.lagana@n002.ai.internal')

& ssh @sa2 'pkill -f make_wake_tts.py; rm -rf ~/vcm/data/wake_v2; echo killed-and-cleaned'

$p = 'C:\Users\Jan\Desktop\AI231 ME2 V2\tools\make_wake_tts.py'
$t = [IO.File]::ReadAllText($p)
[IO.File]::WriteAllText($p, ($t -replace "`r`n","`n"))
& scp -o BatchMode=yes -o ConnectTimeout=15 $p 'jan.rhey.lagana@n002.ai.internal:~/ai231_me2_v2/tools/make_wake_tts.py'
Write-Output "scp exit=$LASTEXITCODE"

& ssh @sa2 'cd ~/ai231_me2_v2 && nohup ~/.conda/envs/vcm/bin/python tools/make_wake_tts.py --out ~/vcm/data/wake_v2 > ~/vcm/wake_tts.log 2>&1 & sleep 8; tail -8 ~/vcm/wake_tts.log'
