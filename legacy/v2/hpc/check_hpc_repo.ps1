# Inspect HPC ~/ai231_me2_v2 divergence before the Runs E/F pull
$ErrorActionPreference = 'Continue'
$sa3 = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n003.ai.internal')
$c = 'cd ~/ai231_me2_v2 && git fetch origin 2>&1 | tail -1; echo ---BEHIND-ORIGIN---; git log --oneline HEAD..origin/main | head -6; echo ---LOCAL-CHANGE-STAT---; git diff HEAD --stat; echo ---UNTRACKED---; git status --short | grep ^??; echo ---DIFF-DETAIL---; git diff HEAD -- src/ pi/ | head -140'
& ssh @sa3 $c 2>$null | Select-Object