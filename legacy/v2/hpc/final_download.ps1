# Wait ~3 min for export+verify to finish, then pull artifacts to laptop
$ErrorActionPreference = 'Continue'
Start-Sleep -Seconds 180
& powershell -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\Jan\hpc_overnight\download.ps1'
