# Compare HPC working-tree src files against local HEAD before the E/F
# push. HPC has NO github access (fetch fails) and uncommitted local
# work (wake A/B, pi UI) — so the only safe sync is targeted scp.
$ErrorActionPreference = 'Continue'
$host3 = 'jan.rhey.lagana@n003.ai.internal'
$d = "$env:TEMP\hpc_check"
New-Item -ItemType Directory -Force $d | Out-Null
scp -o BatchMode=yes "${host3}:~/ai231_me2_v2/src/models.py" "$d\hpc_models.py"
scp -o BatchMode=yes "${host3}:~/ai231_me2_v2/src/train_v2.py" "$d\hpc_train_v2.py"
scp -o BatchMode=yes "${host3}:~/ai231_me2_v2/src/eval_v2.py" "$d\hpc_eval_v2.py"
cd "C:\Users\Jan\Desktop\AI231 ME2 V2"
cmd /c "git show HEAD:src/models.py > $d\loc_models.py"
cmd /c "git show HEAD:src/train_v2.py > $d\loc_train_v2.py"
cmd /c "git show HEAD:src/eval_v2.py > $d\loc_eval_v2.py"
function Diff-It([string]$a, [string]$b) {
  $la = [System.IO.File]::ReadAllLines($a)
  $lb = [System.IO.File]::ReadAllLines($b)
  $c = Compare-Object $la $lb
  if ($c) {
    Write-Output "DIFFERENT ($($c.Count) line diffs):"
    $c | Select-Object -First 25 | Format-Table -AutoSize | Out-String -Width 200
  } else { Write-Output 'IDENTICAL' }
}
Write-Output '===== models.py ====='
Diff-It "$d\loc_models.py" "$d\hpc_models.py"
Write-Output '===== train_v2.py ====='
Diff-It "$d\loc_train_v2.py" "$d\hpc_train_v2.py"
Write-Output '===== eval_v2.py ====='
Diff-It "$d\loc_eval_v2.py" "$d\hpc_eval_v2.py"