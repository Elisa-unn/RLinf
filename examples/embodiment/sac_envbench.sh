#!/bin/bash
# Franka-Sim SAC env-parallelism sweep. One point per total_num_envs value,
# each killed on a fixed wall-clock budget so the points are comparable.
#
# Budget rather than a step count: step time is what is being measured, so a
# fixed number of steps would give each point a different amount of wall clock
# and the slow points would dominate the sweep's duration.
set -u
cd /workspace/RLinf
ulimit -n 65535
# .venv has franka_sim (installed from .venv/serl/franka_sim); .venv_recap has
# mani_skill/libero/openpi but NOT franka_sim, so it fails with
# ModuleNotFoundError: No module named 'franka_sim'. run_async.sh calls bare
# `python` and activates nothing, so this has to be right here.
source /workspace/RLinf/.venv/bin/activate

BUDGET=${BUDGET:-480}          # seconds of wall clock per point
POINTS=${POINTS:-"1 2 4 8 16"}

for N in $POINTS; do
  LOG=/workspace/envbench_${N}env.log
  echo "=== $(date -u +%H:%M:%S) N=$N budget=${BUDGET}s -> $LOG ==="
  rm -f "$LOG"

  # run_async.sh exports MUJOCO_GL=egl and dispatches to train_async.py.
  setsid bash examples/embodiment/run_async.sh "frankasim_sac_envbench_${N}env" \
      > "$LOG" 2>&1 &
  LAUNCH_PID=$!

  # Sample GPU memory while it runs; peak is part of the result.
  PEAK0=0; PEAK1=0
  END=$(( $(date +%s) + BUDGET ))
  while [ "$(date +%s)" -lt "$END" ]; do
    read -r M0 M1 <<< "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ')"
    [ "${M0:-0}" -gt "$PEAK0" ] && PEAK0=$M0
    [ "${M1:-0}" -gt "$PEAK1" ] && PEAK1=$M1
    sleep 10
  done
  echo "peak_gpu0=${PEAK0}MiB peak_gpu1=${PEAK1}MiB" >> "$LOG"

  # Self-match-safe patterns: a bare name would also match this script's own
  # command line (see RECAP_NOTES.md §1.9).
  for pat in "[t]rain_async.py" "[r]un_async.sh"; do
    for p in $(pgrep -f "$pat"); do kill "$p" 2>/dev/null; done
  done
  sleep 5
  ray stop --force >/dev/null 2>&1
  sleep 10
  echo "=== $(date -u +%H:%M:%S) N=$N done, gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/) ==="
done
echo "SWEEP DONE $(date -u)"
