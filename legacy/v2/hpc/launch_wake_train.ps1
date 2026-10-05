# Launch wake-gate training, both architectures, parallel on n002
$ErrorActionPreference = 'Continue'
$sa2 = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR', 'jan.rhey.lagana@n002.ai.internal')
$cmd = 'mkdir -p ~/ai231_me2_v2/runs/v2w && cd ~/ai231_me2_v2 && CUDA_VISIBLE_DEVICES=6 nohup ~/.conda/envs/vcm/bin/python src/train_wake.py --model v2cnn --wake ~/vcm/data/wake_v2 --cmd ~/vcm/data/raw_v2 --real-noise ~/vcm/data/noise16k --out ~/ai231_me2_v2/runs/v2w/v2cnn > ~/ai231_me2_v2/runs/wake_v2cnn.log 2>&1 < /dev/null & CUDA_VISIBLE_DEVICES=2 nohup ~/.conda/envs/vcm/bin/python src/train_wake.py --model bcresnet --wake ~/vcm/data/wake_v2 --cmd ~/vcm/data/raw_v2 --real-noise ~/vcm/data/noise16k --out ~/ai231_me2_v2/runs/v2w/bcresnet > ~/ai231_me2_v2/runs/wake_bcresnet.log 2>&1 < /dev/null & sleep 25; echo ---V2CNN---; tail -5 ~/ai231_me2_v2/runs/wake_v2cnn.log; echo ---BCRESNET---; tail -5 ~/ai231_me2_v2/runs/wake_bcresnet.log'
& ssh @sa2 $cmd
