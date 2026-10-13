#!/bin/bash
# RLT Stage 1 on ManiSkill PegInsertionSideWideClearance-v1.
#
# Runs a PROBE before the real thing. The probe is max_steps=2 with
# save_interval=1, so it exercises load -> forward -> backward -> optimizer ->
# SAVE and then the checkpoint is verified by reading the zip central directory.
# That ordering is deliberate: six earlier memory probes in this project all
# reported "FITS" while running with save_interval: -1, and saving turned out to
# be a separate code path with a separate failure. It also catches the one known
# risk here -- pi05_base_pytorch was converted with --config_name pi05_libero
# while this config uses pi05_rlt_maniskill_joint.
#
# run_vla_sft.sh does NOT forward extra args (its CMD has no "$@"), the same
# papercut as run_async.sh, so the probe calls train_vla_sft.py directly.
LOG=/workspace/chain7.log
exec >> "$LOG" 2>&1
echo "################ chain7 start $(date -u) ################"
set -u
cd /workspace/RLinf || exit 1
ulimit -n 65535
source /workspace/RLinf/.venv_recap/bin/activate
export EMBODIED_PATH=/workspace/RLinf/examples/sft
export REPO_PATH=/workspace/RLinf
export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-}
export HF_LEROBOT_HOME=/workspace/datasets/lerobot
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
unset HF_HUB_ENABLE_HF_TRANSFER

# ---- 0. wait for chain6's repeats, don't race them --------------------------
W=0
while ps -eo args --no-headers | awk '$NF=="/workspace/chain6.sh"' | grep -q .; do
  sleep 30; W=$((W+1)); [ $((W % 10)) -eq 0 ] && echo "  waiting for chain6, $((W/2)) min"
  [ $W -gt 240 ] && { echo "ABORT: chain6 still running after 2 h"; exit 1; }
done
echo "chain6 gone $(date -u)"
if [ "$(ps -eo args --no-headers | grep -c 'train_cfg[.]py\|train_vla_sft[.]py')" != "0" ]; then
  echo "ABORT: a training is already running"; exit 1
fi

# ---- 1. free disk. Only checkpoints whose eval numbers are recorded. --------
echo "=== prune, free before: $(df -h / | awk 'NR==2{print $4}') ==="
# RECAP run: keep 3066 (the headline, 0.512 over 250 trials). 2500 scored 0.58, recorded.
# BC-all run: keep 3066 (0.46 over 250 trials). 2500 scored 0.46, 3000 scored 0.50, recorded.
for d in \
  "$(find /workspace/RLinf/logs -maxdepth 5 -type d -path '*a100b_upstream_ep1*' -name checkpoints | head -1)/global_step_2500" \
  "$(find /workspace/RLinf/logs -maxdepth 5 -type d -path '*a100b_bc_allpos*' -name checkpoints | head -1)/global_step_2500" \
  "$(find /workspace/RLinf/logs -maxdepth 5 -type d -path '*a100b_bc_allpos*' -name checkpoints | head -1)/global_step_3000" ; do
  [ -d "$d" ] && { echo "  deleting $d ($(du -sh "$d" | cut -f1))"; rm -rf "$d"; }
done
echo "=== kept ==="
find /workspace/RLinf/logs -maxdepth 6 -type d -name "global_step_*" | sort
echo "free after: $(df -h / | awk 'NR==2{print $4}')"

# ---- 2. PROBE: 2 steps, and it must SAVE -----------------------------------
echo "=== $(date -u) PROBE (max_steps=2, save_interval=1) ==="
PLOG=/workspace/rlt_probe.log
timeout 3600 python examples/sft/train_vla_sft.py \
  --config-path /workspace/RLinf/examples/sft/config/ \
  --config-name maniskill_rlt_stage1_a100 \
  runner.logger.log_path=/workspace/RLinf/logs/rlt_probe \
  runner.logger.experiment_name=a100b_rlt_probe \
  runner.max_steps=2 \
  runner.save_interval=1 > "$PLOG" 2>&1
PRC=$?
echo "probe rc=$PRC"
echo "--- probe log tail ---"; tail -25 "$PLOG"
echo "--- did it load the weights? ---"
grep -niE "missing key|unexpected key|size mismatch|error loading|state_dict" "$PLOG" | head -8
echo "--- peak GPU during probe ---"
grep -oE "memory[^,]*" "$PLOG" | tail -3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

PCK=$(find /workspace/RLinf/logs/rlt_probe -maxdepth 5 -type d -name checkpoints 2>/dev/null | head -1)
echo "--- probe checkpoints: $PCK ---"
ls -d "$PCK"/global_step_* 2>/dev/null
PSTEP=$(ls -d "$PCK"/global_step_* 2>/dev/null | sed 's#.*global_step_##' | sort -n | tail -1)
if [ -z "${PSTEP:-}" ]; then
  echo "PROBE FAILED: no checkpoint was written. Saving is a separate code path;"
  echo "do NOT launch the 2000-step run until this is understood."
  exit 1
fi
find "$PCK/global_step_$PSTEP" -type f -printf "%14s  %P\n" | sort -k2
python -c "
import zipfile,sys
p='$PCK/global_step_$PSTEP/actor/model_state_dict/full_weights.pt'
try: print('  zip central dir OK, entries =', len(zipfile.ZipFile(p).namelist()))
except Exception as e: print('  PROBE FAILED, checkpoint unreadable:', e); sys.exit(1)
" || exit 1
echo "=== PROBE PASSED: loads, trains, and saves ==="
echo "  reclaiming probe checkpoints: $(du -sh "$PCK" | cut -f1)"
rm -rf /workspace/RLinf/logs/rlt_probe
echo "  free: $(df -h / | awk 'NR==2{print $4}')"

# ---- 3. the real run -------------------------------------------------------
FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
echo "=== free before stage 1: ${FREE}G ==="
[ "$FREE" -lt 110 ] && { echo "ABORT: need >=110G for 3 x 32G concurrent, have ${FREE}G"; exit 1; }
setsid bash /workspace/janitor5.sh a100b_rlt_stage1 2 >/dev/null 2>&1 &
sleep 5
echo "--- janitor ---"; head -4 /workspace/janitor_a100b_rlt_stage1.log 2>/dev/null
echo "=== $(date -u) launching RLT stage 1 (2000 steps) ==="
bash examples/sft/run_vla_sft.sh maniskill_rlt_stage1_a100 > /workspace/a100b_rlt_stage1.log 2>&1
echo "=== stage 1 exited rc=$? $(date -u) ==="
for p in $(pgrep -f "[j]anitor5.sh"); do kill "$p" 2>/dev/null; done
CK=$(find /workspace/RLinf/logs -maxdepth 6 -type d -path "*a100b_rlt_stage1*" -name checkpoints 2>/dev/null | head -1)
echo "stage 1 checkpoints: $CK"; ls -d "$CK"/global_step_* 2>/dev/null
FINAL=$(ls -d "$CK"/global_step_* 2>/dev/null | sed 's#.*global_step_##' | sort -n | tail -1)
[ -n "${FINAL:-}" ] && python -c "
import zipfile
print('  final checkpoint entries =', len(zipfile.ZipFile('$CK/global_step_$FINAL/actor/model_state_dict/full_weights.pt').namelist()))
"
echo "--- loss ---"
grep -oE "train/loss=[0-9.e+-]+" /workspace/a100b_rlt_stage1.log 2>/dev/null | cut -d= -f2 | awk '
  {v[NR]=$1} END{ if(NR<10){print "  too few points:",NR; exit}
    s=0; n=(NR<40?NR:40); for(i=1;i<=n;i++)s+=v[i]; printf "  first %d mean %.5f\n",n,s/n;
    s=0; for(i=NR-n+1;i<=NR;i++)s+=v[i]; printf "  last  %d mean %.5f\n",n,s/n; print "  points:",NR }'
echo "################ chain7 done $(date -u) ################"
