#!/bin/bash
# Supervisor entry for RLT. chain7.sh is NOT resumable yet (Stage 1 has no
# resume wired here), so this only relaunches it when nothing is running; a
# container stop mid-Stage-1 therefore costs the steps since the last save.
exec >> /workspace/supervise_rlt.log 2>&1
echo "=== wrapper fired $(date -u) ==="
sleep 60
if ps -eo args --no-headers | awk '$NF=="/workspace/chain7.sh"' | grep -q .; then
  echo "chain7 already running; nothing to do"; exit 0
fi
if [ "$(ps -eo args --no-headers | grep -c 'train_vla_sft[.]py\|train_cfg[.]py')" != "0" ]; then
  echo "a training is already running; leaving it alone"; exit 0
fi
exec bash /workspace/chain7.sh
