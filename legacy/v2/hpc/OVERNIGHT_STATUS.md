# V2 overnight pipeline — status

finished: Thursday, 01 October, 2026 12:44:41 PM PST (finalizer node: ai-n002)

## TTS
DONE: 47200 ok, 0 failed (2026-10-01 07:27)
decode: raw=47200 fails=0 (train 23600 / eval 23600)

## EVAL-SYN (target > 0.8618, v1i baseline)
v2a_v2cnn_11c	0.9201	0.5971
v2a_v2cnn_10c	0.9333	0.5914
v2a_bcresnet_11c	0.9446	0.5273
v2a_bcresnet_10c	0.9514	0.5512

## Winner
run: v2a_bcresnet_10c
cmd_acc=0.9514 slot_acc=0.5512
ckpt: /home/jan.rhey.lagana/vcm/runs/v2a_bcresnet_10c/vcm_bcresnet_10c_best.pt
pi pkg: /home/jan.rhey.lagana/vcm/data/pi_pkg_v2/v2a_bcresnet_10c
tarball: /home/jan.rhey.lagana/vcm/data/pi_pkg_v2.tar.gz

## Notes
All 4 runs finalized.

## Train logs (last 3 each)
### v2a_v2cnn_11c
ep 29  loss=2.9278  cmd=0.920  slot=0.598  (16.7s)
ep 30  loss=2.9199  cmd=0.920  slot=0.597  (16.6s)
saved -> /home/jan.rhey.lagana/vcm/runs/v2a_v2cnn_11c  (best EVAL-SYN cmd acc 0.920)
### v2a_v2cnn_10c
ep 29  loss=2.9289  cmd=0.933  slot=0.591  (19.9s)  *best*
ep 30  loss=2.8965  cmd=0.933  slot=0.591  (18.4s)  *best*
saved -> /home/jan.rhey.lagana/vcm/runs/v2a_v2cnn_10c  (best EVAL-SYN cmd acc 0.933)
### v2a_bcresnet_11c
ep 29  loss=2.8236  cmd=0.943  slot=0.524  (32.4s)
ep 30  loss=2.8635  cmd=0.945  slot=0.527  (32.8s)  *best*
saved -> /home/jan.rhey.lagana/vcm/runs/v2a_bcresnet_11c  (best EVAL-SYN cmd acc 0.945)
### v2a_bcresnet_10c
ep 29  loss=2.6702  cmd=0.950  slot=0.550  (30.8s)
ep 30  loss=2.6677  cmd=0.951  slot=0.551  (32.0s)  *best*
saved -> /home/jan.rhey.lagana/vcm/runs/v2a_bcresnet_10c  (best EVAL-SYN cmd acc 0.951)
