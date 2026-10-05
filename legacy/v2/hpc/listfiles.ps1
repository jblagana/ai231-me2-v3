# List hpc_overnight .ps1 files newest-first with first line
Get-ChildItem 'C:\Users\Jan\hpc_overnight' -Filter *.ps1 |
  Sort-Object LastWriteTime -Descending |
  Select-Object Name, Length, LastWriteTime |
  Format-Table -AutoSize | Out-String
Get-ChildItem 'C:\Users\Jan\hpc_overnight' -Filter *.ps1 |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5 |
  ForEach-Object {
    Write-Output ('-- ' + $_.Name)
    Get-Content $_.FullName -TotalCount 1
  }
