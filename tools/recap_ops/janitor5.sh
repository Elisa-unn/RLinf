#!/bin/bash
# Keep the newest N complete checkpoints. Parameterised: $1 = experiment tag
# substring, $2 = how many to keep (default 3).
#
# Run directories are named after the CONFIG, not experiment_name, so discover
# rather than glob, and log the watched dirs at startup so a pattern that
# matches nothing is visible immediately. "Complete" needs all four files AND
# three minutes without modification (DCP writes the .distcp shards before
# full_weights.pt). sort -n or global_step_999 outranks global_step_3000.
# Liveness is the heartbeat file, never pgrep.
EXP=${1:?usage: janitor5.sh <exp-substring> [keep]}
KEEP=${2:-3}
LOG=/workspace/janitor_${EXP}.log
HEARTBEAT=/workspace/janitor_${EXP}.heartbeat

find_ckpt_dirs () {
  find /workspace/RLinf/logs -maxdepth 4 -type d -path "*${EXP}*" -name checkpoints 2>/dev/null
}

echo "$(date -u) janitor start exp=$EXP keep=$KEEP; watching:" >> "$LOG"
find_ckpt_dirs | sed 's/^/  /' >> "$LOG"
[ -z "$(find_ckpt_dirs)" ] && echo "$(date -u) WARNING: no checkpoints dir matched yet" >> "$LOG"

is_complete () {
  local d=$1
  [ -f "$d/actor/dcp_checkpoint/.metadata" ]         || return 1
  [ -f "$d/actor/dcp_checkpoint/__0_0.distcp" ]      || return 1
  [ -f "$d/actor/dcp_checkpoint/__1_0.distcp" ]      || return 1
  [ -f "$d/actor/model_state_dict/full_weights.pt" ] || return 1
  [ -z "$(find "$d" -newermt '-3 minutes' -print -quit 2>/dev/null)" ] || return 1
  return 0
}

while true; do
  while IFS= read -r C; do
    [ -d "$C" ] || continue
    mapfile -t done < <(
      for d in "$C"/global_step_*; do
        [ -d "$d" ] && is_complete "$d" && echo "${d##*global_step_} $d"
      done | sort -n -k1 | cut -d' ' -f2-
    )
    n=${#done[@]}
    if [ "$n" -gt "$KEEP" ]; then
      for ((i=0; i<n-KEEP; i++)); do
        echo "$(date -u) pruning ${done[$i]}" >> "$LOG"
        rm -rf "${done[$i]}"
      done
      echo "$(date -u) kept $KEEP, free: $(df -h / | awk 'NR==2{print $4}')" >> "$LOG"
    fi
  done < <(find_ckpt_dirs)
  date -u +%s > "$HEARTBEAT"
  sleep 120
done
