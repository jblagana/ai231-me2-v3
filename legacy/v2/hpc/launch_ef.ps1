# Launch Runs E/F (4 trains) on n003 - per-run process guards make
# re-runs idempotent (already-running runs are skipped). v2e was
# launched by the first (aborted) run; re-running completes the rest.
# v2a NOT included (boss call 10-02: no v2a retrain / comparison yet).
# Sync = targeted scp (HPC has no github access; its tree holds
# uncommitted wake/UI work - never git checkout/reset on HPC).
$ErrorActionPreference = 'Continue'
$host3 = 'jan.rhey.lagana@n003.ai.internal'
$sa3 = @('-o', 'BatchMode=yes', $host3)
$repo = 'C:\Users\Jan\Desktop\AI231 ME2 V2'

Write-Output '===== push 4 files ====='
foreach ($f in @('src/models.py', 'src/train_v2.py', 'src/eval_v2.py', 'tools/split_eval.py')) {
  scp -o BatchMode=yes "$repo\$f" "${host3}:~/ai231_me2_v2/$f"
}

Write-Output '===== md5 verify (remote vs local) ====='
$c = 'cd ~/ai231_me2_v2 && md5sum src/models.py src/train_v2.py src/eval_v2.py tools/split_eval.py'
& ssh @sa3 $c 2>$null | Select-Object
foreach ($f in @('src/models.py', 'src/train_v2.py', 'src/eval_v2.py', 'tools/split_eval.py')) {
  Write-Output ("local " + $f + " = " + (Get-FileHash "$repo\$f" -Algorithm MD5).Hash.ToLower())
}

Write-Output '===== pre-launch: slot weights + manifest split ====='
$c = 'cd ~/ai231_me2_v2 && ls -la data/runb_slot_weights.json && ~/.conda/envs/me2/bin/python tools/split_eval.py --root ~/vcm/data/raw_v2'
& ssh @sa3 $c 2>$null | Select-Object

$PY = '~/.conda/envs/me2/bin/python'
$BASE = '--model bcresnet --merge-10 --class-weights --epochs 60 --patience 8 --batch 32 --lr 1e-3'
$COMMON = '--data ~/vcm/data/raw_v2 --real-noise ~/vcm/data/noise16k --save-all'
function Launch([string]$gpu, [string]$name, [string]$extra) {
  $done = & ssh @sa3 ('grep -c saved ~/vcm/train_' + $name + '.log') 2>$null
  if ($done -and [int]$done -gt 0) {
    Write-Output "===== $name already FINISHED - skip ====="
    return
  }
  $count = & ssh @sa3 ('ps -ef | grep ' + $name + ' | grep -v grep | grep -v bash | wc -l') 2>$null
  if ($count -and [int]$count -gt 0) {
    Write-Output "===== $name already running ($count) - skip ====="
    return
  }
  Write-Output "===== launch $name (gpu $gpu) ====="
  $c = 'cd ~/ai231_me2_v2 && CUDA_VISIBLE_DEVICES=' + $gpu + ' nohup ' + $PY + ' src/train_v2.py ' + $BASE + ' ' + $extra + ' ' + $COMMON + ' --out runs/' + $name + ' > ~/vcm/train_' + $name + '.log 2>&1 < /dev/null & echo ' + $name + '-PID=$!'
  & ssh @sa3 $c 2>$null | Select-Object
}
Launch 0 v2e_bcresnet_10c '--slot-w 0.5'
Launch 1 v2e2_bcresnet_10c '--slot-w 1.0'
Launch 3 v2f_cmd_bcresnet_10c '--cmd-only'
Launch 4 v2f_slot_bcresnet_10c '--slot-only --jit 0.90,1.10 --slot-weights data/runb_slot_weights.json'

Write-Output '===== verify (90s) ====='
Start-Sleep -Seconds 90
$c = 'for f in v2e_bcresnet_10c v2e2_bcresnet_10c v2f_cmd_bcresnet_10c v2f_slot_bcresnet_10c; do echo "--- $f ---"; tail -n 4 ~/vcm/train_$f.log 2>/dev/null; done; echo ---GPU---; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
& ssh @sa3 $c 2>$null | Select-Object