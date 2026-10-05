# Debug UI start: capture stdout/stderr
$ErrorActionPreference = 'Continue'
$c = Get-NetTCPConnection -LocalPort 8399 -State Listen -ErrorAction SilentlyContinue
if ($c) { $c.OwningProcess | Select-Object -Unique | ForEach-Object { try { Stop-Process -Id $_ -Force } catch {} } }
Start-Sleep 1
$p = Start-Process python -ArgumentList @('"C:\Users\Jan\Desktop\AI231 ME2 V2\pi\me2_ui.py"',
        '--dir', '"C:\Users\Jan\Desktop\AI231 ME2 V2\pi"', '--port', '8399') `
        -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput C:\Users\Jan\hpc_overnight\ui_out.log `
        -RedirectStandardError C:\Users\Jan\hpc_overnight\ui_err.log
Start-Sleep 4
Write-Output "alive: $(-not $p.HasExited)"
Write-Output "---stdout---"
Get-Content C:\Users\Jan\hpc_overnight\ui_out.log -ErrorAction SilentlyContinue
Write-Output "---stderr---"
Get-Content C:\Users\Jan\hpc_overnight\ui_err.log -ErrorAction SilentlyContinue
try {
    $html = (Invoke-WebRequest -Uri http://127.0.0.1:8399/ -UseBasicParsing).Content
    Write-Output "GET / -> $($html.Length) bytes, has BOOTS: $($html.Contains('BOOTS'))"
    $body = '{"ts":1761990000,"cmd":"set_alarm","conf":0.9731,"slot":"seven am","slot_index":26,"infer_ms":12.3,"wake_conf":0.9912,"e2e_ms":842}'
    $fire = (Invoke-WebRequest -Uri http://127.0.0.1:8399/fire -Method POST -Body $body -ContentType 'application/json' -UseBasicParsing).Content
    Write-Output "POST /fire -> $fire"
    $st = (Invoke-WebRequest -Uri http://127.0.0.1:8399/state -UseBasicParsing).Content
    Write-Output "GET /state -> $st"
} catch { Write-Output "HTTP FAILED: $($_.Exception.Message)" }
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
Write-Output "DEBUG-DONE"

