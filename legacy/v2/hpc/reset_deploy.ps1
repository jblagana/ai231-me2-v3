# Reset overnight state, deploy fixed train_v2.py + pipeline2.sh, sanity checks
$ErrorActionPreference = 'Continue'
$sa2 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n002.ai.internal')
$sa3 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n003.ai.internal')

# --- local base64 of the two fixed files (home is SHARED: deploy once) ---
$t1 = [IO.File]::ReadAllText('C:\Users\Jan\Desktop\AI231 ME2 V2\src\train_v2.py')
$b1 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($t1 -replace "`r`n","`n")))
$t2 = [IO.File]::ReadAllText('C:\Users\Jan\hpc_overnight\pipeline2.sh')
$b2 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($t2 -replace "`r`n","`n")))
if ($b1 -match '[^A-Za-z0-9+/=]' -or $b2 -match '[^A-Za-z0-9+/=]') { throw 'BAD-B64' }

Write-Output '===== 1. kill strays (both nodes) ====='
& ssh @sa3 'pkill -f ''[o]vernight_pipeline2.sh''; pkill -f ''[t]rain_v2.py''; sleep 1; echo n003-left: $(ps -ef | grep -e ''[o]vernight_pipeline2.sh'' -e ''[t]rain_v2.py'' | wc -l)' 2>$null | Select-Object
& ssh @sa2 'pkill -f ''[o]vernight_pipeline2.sh''; pkill -f ''[t]rain_v2.py''; sleep 1; echo n002-left: $(ps -ef | grep -e ''[o]vernight_pipeline2.sh'' -e ''[t]rain_v2.py'' | wc -l)' 2>$null | Select-Object

Write-Output '===== 2. clean old overnight state ====='
$c = 'cd ~/vcm/overnight 2>/dev/null && { for f in train_v2a_*.log; do [ -f $f ] && mv -f $f $f.crash1; done; rm -f train_v2a_*.pid train_v2a_*.done assigned_ai-n00*.txt STATE STATE_ai-n00*.txt overnight_ai-n00*.log eval_summary.tsv; rm -rf FINALIZE_LOCK; ls; }'
& ssh @sa3 $c 2>$null | Select-Object

Write-Output '===== 3. deploy fixed files ====='
$dep = 'echo ' + $b1 + ' | base64 -d > ~/ai231_me2_v2/src/train_v2.py; echo ' + $b2 + ' | base64 -d > ~/vcm/overnight_pipeline2.sh; ~/.conda/envs/vcm/bin/python -m py_compile ~/ai231_me2_v2/src/train_v2.py && echo PY-COMPILE-OK; bash -n ~/vcm/overnight_pipeline2.sh && echo BASH-SYNTAX-OK; grep -n "sv, sidx" ~/ai231_me2_v2/src/train_v2.py | head -2'
& ssh @sa3 $dep 2>$null | Select-Object

Write-Output '===== 4. sanity: slots selftest + dataset init on real data ====='
$san = 'cd ~/ai231_me2_v2 && ~/.conda/envs/vcm/bin/python - <<EOF
import sys; sys.path.insert(0, "src")
from slots import selftest, extract_slot
assert selftest(), "slots selftest FAILED"
print("spot play a song:", extract_slot("play_music", "play a song"))
print("spot turn off the music:", extract_slot("play_music", "turn off the music"))
from pathlib import Path
from train_v2 import VCMDatasetV2
for split in ("train", "eval"):
    ds = VCMDatasetV2(Path("/home/jan.rhey.lagana/vcm/data/raw_v2"), split)
    nslot = sum(1 for it in ds.items if it[3] >= 0)
    print(split, "n=", len(ds), "slot_items=", nslot)
print("DATASET-OK")
EOF'
& ssh @sa3 $san 2>$null | Select-Object
