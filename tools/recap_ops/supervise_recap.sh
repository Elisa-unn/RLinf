#!/bin/bash
# Supervisor entry point for the remaining RECAP pipeline. Also the recovery path
# after a container stop: each chain script guards itself and resumes, so running
# them in order is safe.
#
# THE GUARD BELOW IS LOAD-BEARING. Without it this wrapper exec'd chain5.sh while
# a chain5 launched by hand was already waiting; both would have cleared the
# "chain4 gone" wait at the same moment and launched two trainings onto the same
# two GPUs. Caught before it fired, but only because the two instances appended
# two start banners to one log.
exec >> /workspace/supervise_recap.log 2>&1
echo "=== supervisor wrapper fired $(date -u) ==="
sleep 60   # let vast_boot.d finish syncing /workspace

for s in chain4 chain5; do
  if pgrep -f "[${s:0:1}]${s:1}.sh" >/dev/null 2>&1; then
    echo "$s.sh already running; this wrapper has nothing to do"
    exit 0
  fi
done

if [ "$(pgrep -cf '[t]rain_cfg.py')" != "0" ]; then
  echo "training already running with no chain supervising it; leaving it alone"
  exit 0
fi

echo "--- chain4 (BC-all: resume, eval, prune; skips the seeded constant-lr phase) ---"
[ -x /workspace/chain4.sh ] && bash /workspace/chain4.sh
echo "--- chain5 (filtered BC) ---"
[ -x /workspace/chain5.sh ] && exec bash /workspace/chain5.sh
