# Deploy wake pipeline files to HPC (LF-normalized), launch wake TTS on n002
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\Jan\Desktop\AI231 ME2 V2'
$remote = 'jan.rhey.lagana@n002.ai.internal'
$opts = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR')

$files = @(
    "tools\make_wake_tts.py",
    "src\train_wake.py",
    "src\export_wake.py"
)
foreach ($f in $files) {
    $p = Join-Path $root $f
    $t = [IO.File]::ReadAllText($p)
    [IO.File]::WriteAllText($p, ($t -replace "`r`n","`n"))
    $r = ($f -replace '\\','/')
    & scp @opts $p "${remote}:~/ai231_me2_v2/$r"
    Write-Output "scp $r exit=$LASTEXITCODE"
}

$launch = 'cd ~/ai231_me2_v2 && ls -la ~/vcm/data/ | grep -i wake; nohup ~/.conda/envs/vcm/bin/python tools/make_wake_tts.py --out ~/vcm/data/wake_v2 > ~/vcm/wake_tts.log 2>&1 & sleep 5; echo ---LOG---; tail -5 ~/vcm/wake_tts.log; echo ---GPU---; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'
& ssh @opts $remote $launch
