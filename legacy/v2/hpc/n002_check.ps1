# n002 GPU check + raw integrity spot-check (n003)
$ErrorActionPreference = 'Continue'
$sa2 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n002.ai.internal')
$c = 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; echo ---APPS---; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | head -12'
& ssh @sa2 $c 2>$null | Select-Object
$sa3 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n003.ai.internal')
$c2 = 'echo ---RAW-SIZE-SAMPLE---; for f in $(find ~/vcm/data/raw_v2/train -name ''*.raw'' | head -5); do ls -l $f; done; echo ---RAW-COUNT-BY-SPLIT---; echo train=$(find ~/vcm/data/raw_v2/train -name ''*.raw'' | wc -l); echo eval=$(find ~/vcm/data/raw_v2/eval -name ''*.raw'' | wc -l); echo ---TINY-FILES---; find ~/vcm/data/raw_v2 -name ''*.raw'' -size -20k | wc -l'
& ssh @sa3 $c2 2>$null | Select-Object
