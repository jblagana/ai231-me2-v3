#!/bin/bash
# ME2 V2 overnight pipeline v2 — adaptive, per-node, partial-launch.
# Launch: RUNS="name:model:tag[:extra] ..." nohup bash pipeline2.sh
# Per node: launch assigned runs as GPUs free -> wait -> write .done markers ->
# mkdir-lock finalizer: eval all 4 (skip missing) -> winner -> export -> verify.
set -u
V=$HOME/vcm
R=$HOME/ai231_me2_v2
PY=$HOME/.conda/envs/vcm/bin/python
O=$V/overnight
NODE=$(hostname -s)
LOG=$O/overnight_${NODE}.log
S=$O/STATE_${NODE}
LOGALL=$O/overnight.log
: > "$LOG"
say() { echo "[$(date '+%F %T')] [$NODE] $*" | tee -a "$LOG" | tee -a "$LOGALL"; }
st()  { echo "$1" > "$S"; say "STATE -> $1"; }
ALL4="v2a_v2cnn_11c v2a_v2cnn_10c v2a_bcresnet_11c v2a_bcresnet_10c"
RUNS="${RUNS:?RUNS env required}"

say "=== pipeline v2 start on $NODE runs=$RUNS ==="
st waiting_gpu

gpu_free() { # first GPU 0-7 with <2GB used, no live process assigned to it
  local g u line gp pid
  for g in 0 1 2 3 4 5 6 7; do
    if [ -f "$O/assigned_$NODE.txt" ]; then
      for line in $(cat "$O/assigned_$NODE.txt"); do
        gp=${line%%:*}; pid=${line##*:}
        if [ "$gp" = "$g" ] && kill -0 "$pid" 2>/dev/null; then
          continue 2
        fi
      done
    fi
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [ -z "$u" ] && continue
    [ "$u" -ge 2048 ] && continue
    echo "$g"; return 0
  done
  return 1
}
launch_one() { # $1=name $2=model $3=tag $4=extra
  local name=$1 model=$2 tag=$3 extra=$4 g pid
  g=$(gpu_free) || return 1
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" "$R/src/train_v2.py" \
    --data "$V/data/raw_v2" --model "$model" $extra \
    --out "$V/runs/$name" > "$O/train_$name.log" 2>&1 &
  pid=$!
  echo $pid > "$O/train_$name.pid"
  echo "$g:$pid" >> "$O/assigned_$NODE.txt"
  say "launched $name gpu=$g pid=$pid"
  return 0
}

# ---------- Stage 1: launch this node's runs as GPUs free ----------
while true; do
  all=1
  for spec in $RUNS; do
    IFS=: read -r name model tag extra <<< "$spec"
    [ -f "$O/train_$name.pid" ] || all=0
  done
  [ "$all" -eq 1 ] && break
  for spec in $RUNS; do
    IFS=: read -r name model tag extra <<< "$spec"
    [ -f "$O/train_$name.pid" ] && continue
    launch_one "$name" "$model" "$tag" "$extra" || true
  done
  sleep 300
done
st waiting_training

# ---------- Stage 2: wait for this node's runs (3h stall guard) ----------
run_alive() { # $1=pid $2=runname -> 0 iff pid is a live train_v2 for that run
  local pid=$1 name=$2
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "runs/$name" || return 1
  return 0
}
while true; do
  alive=0
  for spec in $RUNS; do
    IFS=: read -r name rest <<< "$spec"
    pid=$(cat "$O/train_$name.pid" 2>/dev/null)
    if [ -n "$pid" ] && run_alive "$pid" "$name"; then
      alive=1
    fi
  done
  [ "$alive" -eq 0 ] && break
  newest=$(ls -t "$O"/train_*.log 2>/dev/null | head -1)
  if [ -n "$newest" ] && [ $(( $(date +%s) - $(stat -c %Y "$newest") )) -gt 10800 ]; then
    say "TRAINING STALLED 3h on $NODE — writing .done (failed) and exiting"
    for spec in $RUNS; do
      IFS=: read -r name rest <<< "$spec"
      [ -f "$O/train_$name.done" ] || echo "FAILED-stalled" > "$O/train_$name.done"
    done
    st stalled_training
    exit 1
  fi
  sleep 300
done
for spec in $RUNS; do
  IFS=: read -r name model tag extra <<< "$spec"
  ck="$V/runs/$name/vcm_${model}_${tag}c_best.pt"
  if [ -f "$ck" ]; then
    echo "OK" > "$O/train_$name.done"
  else
    echo "FAILED-no-checkpoint" > "$O/train_$name.done"
    say "RUN FAILED (no checkpoint): $name"
  fi
  echo "--- $name tail ---" >> "$LOG"
  tail -n 4 "$O/train_$name.log" 2>/dev/null | sed "s/^/[$name] /" >> "$LOG"
done
say "this node's runs done (.done written)"
st training_done

# ---------- Stage 3: acquire finalizer lock (shared, atomic mkdir) ----------
LOCK=$O/FINALIZE_LOCK
if ! mkdir "$LOCK" 2>/dev/null; then
  say "sibling node holds FINALIZE_LOCK — exiting (not the finalizer)"
  st not_finalizer
  exit 0
fi
st finalizing
# wait up to 180 min for all 4 .done markers (sibling node's runs)
deadline=$(( $(date +%s) + 10800 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  n=0
  for name in $ALL4; do [ -f "$O/train_$name.done" ] && n=$((n+1)); done
  [ "$n" -eq 4 ] && break
  sleep 120
done
missing=""
for name in $ALL4; do [ -f "$O/train_$name.done" ] || missing="$missing $name"; done
[ -n "$missing" ] && say "WARNING: no .done within 3h for:$missing — proceeding with available ckpts"
# ---------- Stage 4: EVAL-SYN for each available ckpt ----------
: > "$O/eval_summary.tsv"
EG=""
gdeadline=$(( $(date +%s) + 10800 ))
while [ -z "$EG" ] && [ "$(date +%s)" -lt "$gdeadline" ]; do
  EG=$(gpu_free) || true
  [ -z "$EG" ] && sleep 900
done
if [ -n "$EG" ]; then say "eval on GPU $EG"; else say "NO FREE GPU in 3h — eval on CPU"; fi
for spec in "v2a_v2cnn_11c:v2cnn:11:" "v2a_v2cnn_10c:v2cnn:10:--merge-10" \
            "v2a_bcresnet_11c:bcresnet:11:" "v2a_bcresnet_10c:bcresnet:10:--merge-10"; do
  IFS=: read -r name model tag extra <<< "$spec"
  ckpt="$V/runs/$name/vcm_${model}_${tag}c_best.pt"
  if [ ! -f "$ckpt" ]; then
    say "EVAL SKIP $name (no $ckpt)"
    echo -e "$name\tNOCHECKPOINT\tNA" >> "$O/eval_summary.tsv"
    continue
  fi
  CUDA_VISIBLE_DEVICES=$EG "$PY" "$R/src/eval_v2.py" --model "$model" $extra \
    --ckpt "$ckpt" --data "$V/data/raw_v2" --split eval \
    > "$O/eval_$name.log" 2>&1
  acc=$(awk '/^command acc:/ {print $3}' "$O/eval_$name.log" | head -1)
  sacc=$(awk '/^slot acc:/ {print $3}' "$O/eval_$name.log" | head -1)
  say "EVAL $name cmd=${acc:-NA} slot=${sacc:-NA}"
  echo -e "$name\t${acc:-NA}\t${sacc:-NA}" >> "$O/eval_summary.tsv"
done

# ---------- Stage 5: winner -> export -> verify ----------
winner=$(awk -F'\t' '$2 ~ /^[0-9.]+$/ {print $2"\t"$0}' "$O/eval_summary.tsv" \
  | sort -rn | head -1 | cut -f2-)
if [ -z "$winner" ]; then
  say "NO VALID EVAL — pipeline stopping"
  st eval_failed
  exit 1
fi
IFS=$'\t' read -r wname wacc wsacc <<< "$winner"
say "WINNER: $wname EVAL-SYN cmd=$wacc (target > 0.8618)"
case $wname in
  v2a_v2cnn_11c)    wmodel=v2cnn;    wnc=11 ;;
  v2a_v2cnn_10c)    wmodel=v2cnn;    wnc=10 ;;
  v2a_bcresnet_11c) wmodel=bcresnet; wnc=11 ;;
  v2a_bcresnet_10c) wmodel=bcresnet; wnc=10 ;;
esac
wckpt="$V/runs/$wname/vcm_${wmodel}_${wnc}c_best.pt"
wav=$(find "$V/data/raw_v2/eval" -name '*.raw' -size +20k | head -1)
CUDA_VISIBLE_DEVICES=$EG "$PY" "$R/src/export_v2.py" --model "$wmodel" \
  --n-classes "$wnc" --ckpt "$wckpt" \
  --out "$V/data/pi_pkg_v2/$wname" --wav "$wav" > "$O/export.log" 2>&1
if [ $? -eq 0 ]; then
  say "EXPORT PASS -> $V/data/pi_pkg_v2/$wname"
  tar czf "$V/data/pi_pkg_v2.tar.gz" -C "$V/data" pi_pkg_v2 2>>"$LOG"
  say "tarball: $V/data/pi_pkg_v2.tar.gz ($(du -h "$V/data/pi_pkg_v2.tar.gz" | cut -f1))"
  CUDA_VISIBLE_DEVICES=$EG "$PY" "$R/tools/verify_pi_v2.py" \
    --pkg "$V/data/pi_pkg_v2/$wname" --ckpt "$wckpt" \
    --data "$V/data/raw_v2/eval" --clips 8 > "$O/verify.log" 2>&1
  say "verify exit=$? (log: $O/verify.log)"
  st done
else
  say "EXPORT FAIL (log: $O/export.log)"
  st export_failed
  exit 1
fi

# ---------- Stage 6: status report ----------
{
  echo "# V2 overnight pipeline — status"
  echo
  echo "finished: $(date) (finalizer node: $NODE)"
  echo
  echo "## TTS"
  echo "DONE: 47200 ok, 0 failed (2026-10-01 07:27)"
  echo "decode: raw=47200 fails=0 (train 23600 / eval 23600)"
  echo
  echo "## EVAL-SYN (target > 0.8618, v1i baseline)"
  cat "$O/eval_summary.tsv"
  echo
  echo "## Winner"
  echo "run: $wname"
  echo "cmd_acc=$wacc slot_acc=$wsacc"
  echo "ckpt: $wckpt"
  echo "pi pkg: $V/data/pi_pkg_v2/$wname"
  echo "tarball: $V/data/pi_pkg_v2.tar.gz"
  echo
  echo "## Notes"
  if [ -n "$missing" ]; then
    echo "NOT FINAL FOR:$missing — still training at finalize time;"
    echo "eval them once done; if a later cmd_acc beats $wacc, re-export that winner."
  else
    echo "All 4 runs finalized."
  fi
  echo
  echo "## Train logs (last 3 each)"
  for name in $ALL4; do
    echo "### $name"; tail -n 3 "$O/train_$name.log" 2>/dev/null
  done
} > "$O/OVERNIGHT_STATUS.md"
echo "FINAL: done" > "$O/STATE_ALL"
say "=== pipeline v2 complete — status in $O/OVERNIGHT_STATUS.md ==="

