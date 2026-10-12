#!/bin/bash
# Phase C replacement: FILTERED BC instead of the constant-lr run.
#
# Why not edit chain4.sh: bash reads a script lazily from the file, so editing
# one that is mid-execution corrupts the running flow. Instead chain4's own
# skip-guard is used — /workspace/constlr_results.tsv is pre-seeded so its
# Phase C is skipped — and this script does the replacement work.
#
# Safe to launch blindly, same three properties as chain4:
#   1. waits for chain4 to finish rather than racing it;
#   2. refuses to train if training is already running;
#   3. resumes from its own newest complete checkpoint, and skips the whole
#      phase if its result TSV already has rows.
LOG=/workspace/chain5.log
exec >> "$LOG" 2>&1
echo "################ chain5 start $(date -u) ################"
set -u
cd /workspace/RLinf || exit 1
ulimit -n 65535
source /workspace/RLinf/.venv_recap/bin/activate
export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-}

RES=/workspace/succonly_results.tsv
[ -s "$RES" ] || printf "tag\tstep\tguidance\tsuccess_once\treturn\trc\n" > "$RES"
if [ "$(wc -l < "$RES")" -gt 1 ]; then
  echo "already has results, nothing to do"; column -t -s $'\t' "$RES"; exit 0
fi

# --- wait for chain4 (BC training + its eval + the prune) --------------------
echo "waiting for chain4 to finish..."
W=0
while pgrep -f "[c]hain4.sh" >/dev/null 2>&1; do
  sleep 120; W=$((W+2))
  [ $((W % 60)) -eq 0 ] && echo "  still waiting, ${W} min, bc rows=$(( $(wc -l < /workspace/bc_results.tsv 2>/dev/null || echo 1) - 1 ))"
  [ $W -gt 2880 ] && { echo "ABORT: chain4 still running after 48 h"; exit 1; }
done
echo "chain4 gone $(date -u)"
echo "--- BC results it produced ---"; column -t -s $'\t' /workspace/bc_results.tsv 2>/dev/null

# anything left mid-flight
for pat in "[e]val_embodied_agent.py" "[r]un_eval.sh" "[j]anitor5.sh"; do
  for p in $(pgrep -f "$pat"); do echo "killing stale $pat pid $p"; kill "$p" 2>/dev/null; done
done
sleep 5; ray stop --force >/dev/null 2>&1; sleep 10

if [ "$(pgrep -cf '[t]rain_cfg.py')" != "0" ]; then
  echo "training still running after chain4 exited; refusing to start a second one"
  exit 1
fi

newest_ckpt () {
  local exp=$1 CK
  CK=$(find /workspace/RLinf/logs -maxdepth 5 -type d -path "*${exp}*" -name checkpoints 2>/dev/null | head -1)
  [ -n "$CK" ] || return 0
  for st in $(ls -d "$CK"/global_step_* 2>/dev/null | sed 's#.*global_step_##' | sort -rn); do
    local D="$CK/global_step_$st"
    [ -f "$D/actor/dcp_checkpoint/.metadata" ] &&
    [ -f "$D/actor/dcp_checkpoint/__0_0.distcp" ] &&
    [ -f "$D/actor/dcp_checkpoint/__1_0.distcp" ] &&
    [ -f "$D/actor/model_state_dict/full_weights.pt" ] && { echo "$D"; return 0; }
  done
}

run_eval () {
  local ck=$1 st=$2 g=$3 tag=$4 out=$5
  local L=/workspace/eval_${tag}.log
  echo "=== $(date -u +%H:%M:%S) eval tag=$tag step=$st guidance=$g"
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  timeout 1800 bash evaluations/run_eval.sh libero libero_10_task0_cfg_eval \
    runner.ckpt_path="$ck/global_step_${st}/actor/model_state_dict/full_weights.pt" \
    rollout.model.model_path=/workspace/models/pi05_base_pytorch \
    runner.logger.experiment_name="eval_${tag}" \
    rollout.model.openpi.cfgrl_guidance_scale="$g" > "$L" 2>&1
  local rc=$? so ret
  so=$(grep -oE "success_once=[0-9.]+" "$L" | tail -1 | cut -d= -f2)
  ret=$(grep -oE "return=[0-9.]+"      "$L" | tail -1 | cut -d= -f2)
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$st" "$g" "${so:-NA}" "${ret:-NA}" "$rc" >> "$out"
  echo "    -> success_once=${so:-NA} return=${ret:-NA} rc=$rc"
  for pat in "[e]val_embodied_agent.py" "[r]un_eval.sh"; do
    for p in $(pgrep -f "$pat"); do kill "$p" 2>/dev/null; done
  done
  sleep 5; ray stop --force >/dev/null 2>&1; sleep 10
}

############################# FILTERED BC #############################
FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
echo "=== free before filtered BC: ${FREE}G ==="
[ "$FREE" -lt 110 ] && { echo "ABORT: need >=110G, have ${FREE}G"; exit 1; }

RESUME=$(newest_ckpt a100b_bc_succonly)
EXTRA=""
if [ -n "$RESUME" ]; then echo "RESUMING from $RESUME"; EXTRA="runner.resume_dir=$RESUME"
else echo "starting from scratch"; fi

setsid bash /workspace/janitor5.sh a100b_bc_succonly 2 >/dev/null 2>&1 &
sleep 5
echo "--- janitor ---"; head -4 /workspace/janitor_a100b_bc_succonly.log 2>/dev/null
echo "=== $(date -u) launching cfg_a100b_bc_succonly ${EXTRA:+with $EXTRA} ==="
bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh cfg_a100b_bc_succonly $EXTRA \
  > /workspace/a100b_succonly.log 2>&1
echo "=== exited rc=$? $(date -u) ==="
for p in $(pgrep -f "[j]anitor5.sh"); do kill "$p" 2>/dev/null; done

CK=$(find /workspace/RLinf/logs -maxdepth 5 -type d -path "*a100b_bc_succonly*" -name checkpoints 2>/dev/null | head -1)
echo "checkpoints: $CK"; ls -d "$CK"/global_step_* 2>/dev/null
for st in $(ls -d "$CK"/global_step_* 2>/dev/null | sed 's#.*global_step_##' | sort -n); do
  python -c "
import zipfile,sys
try: zipfile.ZipFile('$CK/global_step_$st/actor/model_state_dict/full_weights.pt').namelist()
except Exception as e: print('  SKIP $st unreadable:', e); sys.exit(1)
" || continue
  run_eval "$CK" "$st" 1.0 "succonly_s${st}" "$RES"
done
FINAL=$(ls -d "$CK"/global_step_* 2>/dev/null | sed 's#.*global_step_##' | sort -n | tail -1)
[ -n "$FINAL" ] && run_eval "$CK" "$FINAL" 1.5 "succonly_s${FINAL}_g1.5" "$RES"

echo "=== CHAIN5 DONE $(date -u) ==="
echo "--- BC-all ---";       column -t -s $'\t' /workspace/bc_results.tsv 2>/dev/null
echo "--- filtered BC ---";  column -t -s $'\t' "$RES" 2>/dev/null
