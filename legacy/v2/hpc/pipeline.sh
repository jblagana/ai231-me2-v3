#!/bin/bash
# ME2 V2 overnight pipeline — runs on n003 (GPU node). Shared JuiceFS home,
# so TTS state (n002) is visible here via ~/vcm/tts_v2.log.
# Stages: wait_tts -> decode -> train x4 (GPUs 2-5) -> eval -> export -> verify
set -u
V=$HOME/vcm
R=$HOME/ai231_me2_v2
PY=$HOME/.conda/envs/vcm/bin/python
O=$V/overnight
mkdir -p "$O"
LOG=$O/overnight.log
S=$O/STATE
: > "$LOG"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
st()  { echo "$1" > "$S"; say "STATE -> $1"; }
NAMES="v2a_v2cnn_11c v2a_v2cnn_10c v2a_bcresnet_11c v2a_bcresnet_10c"

say "=== V2 overnight pipeline start (n003) ==="
st waiting_tts

# ---------- Stage 1: wait for TTS DONE line in shared log ----------
prev_count=""
stall_since=""
while true; do
  if grep -q '^DONE: ' "$V/tts_v2.log" 2>/dev/null; then break; fi
  cur=$(grep -o 'cum ok=[0-9]*' "$V/tts_v2.log" 2>/dev/null | tail -1)
  now=$(date +%s)
  if [ -n "$cur" ] && [ "$cur" != "$prev_count" ]; then
    prev_count=$cur
    stall_since=$now
  fi
  if [ -n "$stall_since" ] && [ $((now - stall_since)) -gt 7200 ]; then
    say "TTS STALLED: no progress 2h (last: $cur) — aborting for diagnosis"
    st stalled_tts
    exit 1
  fi
  sleep 120
done
DONE_LINE=$(grep '^DONE: ' "$V/tts_v2.log" | tail -1)
say "TTS DONE: $DONE_LINE"
N_MP3=$(find "$V/data/raw_v2" -name '*.mp3' | wc -l)
say "mp3 count on disk: $N_MP3 (expect 47200)"
st decoding

# ---------- Stage 2: parallel decode (same ffmpeg as tools/decode_tts.py) ----------
FF=$("$PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
say "ffmpeg binary: $FF"
export FF O
find "$V/data/raw_v2" -name '*.mp3' -print0 | \
  xargs -0 -P 16 -I{} bash -c '
    mp3="$1"; raw="${mp3%.mp3}.raw"
    [ -s "$raw" ] || "$FF" -y -loglevel error -i "$mp3" -ar 16000 -ac 1 -f s16le "$raw" \
      || echo "DECODE-FAIL $mp3" >> "$O/decode_fails.txt"
  ' _ {}
N_RAW=$(find "$V/data/raw_v2" -name '*.raw' | wc -l)
NF=$(wc -l < "$O/decode_fails.txt" 2>/dev/null || echo 0)
say "decode complete: raw=$N_RAW fails=$NF"
st training

# ---------- Stage 3: GPU guard + launch 4 parallel runs ----------
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
launch() { # $1=gpu $2=outname $3=model $4=extra-flags
  local g=$1 name=$2 model=$3 extra=$4
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" "$R/src/train_v2.py" \
    --data "$V/data/raw_v2" --model "$model" $extra \
    --out "$V/runs/$name" > "$O/train_$name.log" 2>&1 &
  echo $! > "$O/train_$name.pid"
  say "launched $name gpu=$g pid=$(cat "$O/train_$name.pid")"
}
while true; do
  ok=1
  for g in 2 3 4 5; do
    u=$(gpu_used $g)
    if [ -z "$u" ] || [ "$u" -ge 2048 ]; then ok=0; fi
  done
  [ "$ok" -eq 1 ] && break
  say "GPUs 2-5 not all free yet (retry in 15 min)"
  sleep 900
done
launch 2 v2a_v2cnn_11c v2cnn ""
launch 3 v2a_v2cnn_10c v2cnn "--merge-10"
launch 4 v2a_bcresnet_11c bcresnet ""
launch 5 v2a_bcresnet_10c bcresnet "--merge-10"
st waiting_training
# ---------- Stage 4: wait for all 4 to exit (with stall guard) ----------
while true; do
  alive=0
  for name in $NAMES; do
    pid=$(cat "$O/train_$name.pid" 2>/dev/null)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1
  done
  [ "$alive" -eq 0 ] && break
  newest=$(ls -t "$O"/train_*.log 2>/dev/null | head -1)
  if [ -n "$newest" ] && [ $(( $(date +%s) - $(stat -c %Y "$newest") )) -gt 10800 ]; then
    say "TRAINING STALLED: no log growth for 3h — aborting for diagnosis"
    st stalled_training
    exit 1
  fi
  sleep 300
done
for name in $NAMES; do
  echo "--- train_$name tail ---" >> "$LOG"
  tail -n 4 "$O/train_$name.log" 2>/dev/null | sed "s/^/[$name] /" >> "$LOG"
done
sleep 60
st evaluating

# ---------- Stage 5: EVAL-SYN for each completed run ----------
: > "$O/eval_summary.tsv"
for spec in "v2a_v2cnn_11c:v2cnn:11:" "v2a_v2cnn_10c:v2cnn:10:--merge-10" \
            "v2a_bcresnet_11c:bcresnet:11:" "v2a_bcresnet_10c:bcresnet:10:--merge-10"; do
  IFS=: read name model tag extra <<< "$spec"
  ckpt="$V/runs/$name/vcm_${model}_${tag}c_best.pt"
  if [ ! -f "$ckpt" ]; then
    say "EVAL SKIP $name (no $ckpt)"
    echo -e "$name\tNOCHECKPOINT\tNA" >> "$O/eval_summary.tsv"
    continue
  fi
  CUDA_VISIBLE_DEVICES=2 "$PY" "$R/src/eval_v2.py" --model "$model" $extra \
    --ckpt "$ckpt" --data "$V/data/raw_v2" --split eval \
    > "$O/eval_$name.log" 2>&1
  acc=$(awk '/^command acc:/ {print $3}' "$O/eval_$name.log" | head -1)
  sacc=$(awk '/^slot acc:/ {print $3}' "$O/eval_$name.log" | head -1)
  say "EVAL $name cmd=${acc:-NA} slot=${sacc:-NA}"
  echo -e "$name\t${acc:-NA}\t${sacc:-NA}" >> "$O/eval_summary.tsv"
done

# ---------- Stage 6: pick winner, export, verify ----------
winner=$(awk -F'\t' '$2 ~ /^[0-9.]+$/ {print $2"\t"$0}' "$O/eval_summary.tsv" \
  | sort -rn | head -1 | cut -f2-)
if [ -z "$winner" ]; then
  say "NO RUN PRODUCED A VALID EVAL — pipeline stopping at eval"
  st eval_failed
  exit 1
fi
IFS=$'\t' read wname wacc wsacc <<< "$winner"
say "WINNER: $wname  EVAL-SYN cmd=$wacc (target > 0.8618)"
case $wname in
  v2a_v2cnn_11c)    wmodel=v2cnn;    wnc=11 ;;
  v2a_v2cnn_10c)    wmodel=v2cnn;    wnc=10 ;;
  v2a_bcresnet_11c) wmodel=bcresnet; wnc=11 ;;
  v2a_bcresnet_10c) wmodel=bcresnet; wnc=10 ;;
esac
wckpt="$V/runs/$wname/vcm_${wmodel}_${wnc}c_best.pt"
wav=$(find "$V/data/raw_v2/eval" -name '*.raw' | head -1)
CUDA_VISIBLE_DEVICES=2 "$PY" "$R/src/export_v2.py" --model "$wmodel" \
  --n-classes "$wnc" --ckpt "$wckpt" \
  --out "$V/data/pi_pkg_v2/$wname" --wav "$wav" > "$O/export.log" 2>&1
if [ $? -eq 0 ]; then
  say "EXPORT PASS -> $V/data/pi_pkg_v2/$wname"
  tar czf "$V/data/pi_pkg_v2.tar.gz" -C "$V/data" pi_pkg_v2 2>>"$LOG"
  say "tarball: $V/data/pi_pkg_v2.tar.gz ($(du -h "$V/data/pi_pkg_v2.tar.gz" | cut -f1))"
  CUDA_VISIBLE_DEVICES=2 "$PY" "$R/tools/verify_pi_v2.py" \
    --pkg "$V/data/pi_pkg_v2/$wname" --ckpt "$wckpt" \
    --data "$V/data/raw_v2/eval" --clips 8 > "$O/verify.log" 2>&1
  say "verify exit=$? (log: $O/verify.log)"
  st done
else
  say "EXPORT FAIL (log: $O/export.log)"
  st export_failed
  exit 1
fi

# ---------- Stage 7: status report ----------
{
  echo "# V2 overnight pipeline — status"
  echo
  echo "finished: $(date)"
  echo
  echo "## TTS"
  echo "done line: $DONE_LINE"
  echo "mp3=$N_MP3 raw=$N_RAW decode_fails=$NF"
  echo
  echo "## EVAL-SYN (target > 0.8618, v1i baseline)"
  cat "$O/eval_summary.tsv"
  echo
  echo "## Winner"
  echo "run: $wname"
  echo "cmd_acc=$wacc slot_acc=$wsacc"
  echo "ckpt: $wckpt"
  echo "pi pkg dir: $V/data/pi_pkg_v2/$wname"
  echo "tarball: $V/data/pi_pkg_v2.tar.gz"
  echo
  echo "## Train logs (last 3 lines each)"
  for name in $NAMES; do
    echo "### $name"; tail -n 3 "$O/train_$name.log" 2>/dev/null
  done
} > "$O/OVERNIGHT_STATUS.md"
say "=== pipeline complete — status in $O/OVERNIGHT_STATUS.md ==="

