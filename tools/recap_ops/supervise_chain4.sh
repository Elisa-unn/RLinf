#!/bin/bash
# Supervisor wrapper for chain4.sh. Lives in /workspace (the persistent volume)
# so it survives a container stop; the supervisor conf that calls it is in
# /etc/supervisor/conf.d/ and may or may not survive — if it does not, we are no
# worse off than before, which is why installing this is a one-way improvement.
#
# WHY: this box idled three times waiting for a human to notice it had come back
# (4.5 h, 22 min, 2 h 16 m). chain4.sh refuses to start if training is already
# running and resumes each phase from its own newest checkpoint, so being
# launched blindly at boot is safe.
exec >> /workspace/supervise_chain4.log 2>&1
echo "=== supervisor wrapper fired $(date -u) ==="

# Give the box a moment to finish its own boot scripts (vast_boot.d syncs
# /workspace and the home dirs) before touching anything under them.
sleep 60

if [ "$(pgrep -cf '[t]rain_cfg.py')" != "0" ]; then
  echo "training already running; nothing to do"
  exit 0
fi
if [ ! -x /workspace/chain4.sh ]; then
  echo "chain4.sh missing or not executable; nothing to do"
  exit 0
fi

echo "launching chain4.sh"
exec bash /workspace/chain4.sh
