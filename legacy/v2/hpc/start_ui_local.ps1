# Start the ME2 UI locally on :8399 (leaves it running) + 2 SAMPLE fires
$ErrorActionPreference = 'Continue'
$c = Get-NetTCPConnection -LocalPort 8399 -State Listen -ErrorAction SilentlyContinue
if ($c) { $c.OwningProcess | Select-Object -Unique | ForEach-Object { try { Stop-Process -Id $_ -Force } catch {} } }
Start-Sleep 1
$p = Start-Process python -ArgumentList @('"C:\Users\Jan\Desktop\AI231 ME2 V2\pi\me2_ui.py"',
        '--dir', '"C:\Users\Jan\Desktop\AI231 ME2 V2\pi"', '--port', '8399') `
        -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput C:\Users\Jan\hpc_overnight\ui_live.log `
        -RedirectStandardError C:\Users\Jan\hpc_overnight\ui_live_err.log
Start-Sleep 2
Write-Output "UI pid=$($p.Id) alive=$(-not $p.HasExited)"
$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$b1 = '{"ts":' + ($now - 45) + ',"cmd":"set_alarm","conf":0.9731,"slot":"seven am","slot_index":26,"infer_ms":12.3,"wake_conf":0.9912,"e2e_ms":842}'
$b2 = '{"ts":' + ($now - 8)  + ',"cmd":"play_music","conf":0.9412,"slot":"any","slot_index":0,"infer_ms":11.8,"wake_conf":0.9871,"e2e_ms":913}'
Invoke-WebRequest -Uri http://127.0.0.1:8399/fire -Method POST -Body $b1 -ContentType 'application/json' -UseBasicParsing | Out-Null
Start-Sleep 1
Invoke-WebRequest -Uri http://127.0.0.1:8399/fire -Method POST -Body $b2 -ContentType 'application/json' -UseBasicParsing | Out-Null
$st = (Invoke-WebRequest -Uri http://127.0.0.1:8399/state -UseBasicParsing).Content
Write-Output "state -> $st"
