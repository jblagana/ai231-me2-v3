# Verify training health, errors visible
$ErrorActionPreference = 'Continue'
$sa2 = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR', 'jan.rhey.lagana@n002.ai.internal')
$sa3 = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR', 'jan.rhey.lagana@n003.ai.internal')
Write-Output '===== n003 ====='
$c3 = 'for n in v2a_v2cnn_11c v2a_v2cnn_10c; do echo ---$n---; tail -n 4 ~/vcm/overnight/train_$n.log 2>/dev/null || echo "not started"; done; echo ---PROCS---; ps -ef | grep [t]rain_v2.py | wc -l'
& ssh @sa3 $c3
Write-Output '===== n002 ====='
$c2 = 'for n in v2a_bcresnet_11c v2a_bcresnet_10c; do echo ---$n---; tail -n 4 ~/vcm/overnight/train_$n.log 2>/dev/null || echo "not started"; done; echo ---PROCS---; ps -ef | grep [t]rain_v2.py | wc -l'
& ssh @sa2 $c2

