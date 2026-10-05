# test n002 -> n003 inter-node ssh
$ErrorActionPreference = 'Continue'
$sa = @('-o', 'BatchMode=yes', 'jan.rhey.lagana@n002.ai.internal')
$c = 'ssh -o BatchMode=yes -o ConnectTimeout=10 jan.rhey.lagana@n003.ai.internal hostname 2>/dev/null && echo INTERNODE-OK || echo INTERNODE-FAIL'
& ssh @sa $c 2>&1 | Select-String -NotMatch '^#|^\s*#|^\s*$'
