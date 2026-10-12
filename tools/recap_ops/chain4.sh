#!/bin/bash
# RESUMABLE successor to chain3.sh. Phase A (the 5 eval repeats) is DONE and its
# numbers are committed, so this starts at the BC control.
#
#   B. BC control  (advantage column all True)  ~25 h  -> eval
#   C. prune, then constant-lr run              ~25 h  -> eval
#
# Designed to be relaunched blindly after the container stops. Three properties
# make that safe, because this box has now idled three times waiting for a human
# (4.5 h, 22 min, and 2 h 16 m):
#   1. it refuses to start if training is already running;
#   2. each training phase RESUMES from its own newest checkpoint if one exists,
#      so a restart costs at most save_interval = 500 steps (~4 h), not the run;
#   3. a phase whose result TSV already has rows is skipped.
LOG=/workspace/chain4.log
exec >> "$LOG" 2>&1
echo "################ chain4 start $(date -u) ################"
set -u
cd /workspace/RLinf || exit 1
ulimit -n 65535
source /workspace/RLinf/.venv_recap/bin/activate
export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-}

# --- single-instance guard -------------------------------------------------
RUNNING=$(pgrep -cf "[t]rain_cfg.py")
if [ "$RUNNING" != "0" ]; then
  echo "training already running ($RUNNING procs); chain4 exiting without action"
  exit 0
fi
OTHER=$(pgrep -f "[c]hain4.sh" | grep -v "^$$\$" | wc -l)
echo "other chain4 processes: $OTHER (this pid $$)"

for pat in "[c]hain3.sh" "[e]val_embodied_agent.py" "[r]un_eval.sh" "[j]anitor5.sh"; do
  for p in $(pgrep -f "$pat"); do echo "killing stale $pat pid $p"; kill "$p" 2>/dev/null; done
done
sleep 5; ray stop --force >/dev/null 2>&1; sleep 10

newest_ckpt () {  # exp-substring -> prints the newest complete checkpoint dir, or nothing
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

run_eval () {   # ckpt_dir, step, guidance, tag, out_tsv
  local ck=$1 st=$2 g=$3 tag=$4 out=$5
  local L=/workspace/eval_${tag}.log
  echo "=== $(date -u +%H:%M:%S) eval tag=$tag step=$st guidance=$g"
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  timeout 1800 bash evaluations/run_eval.sh libero libero_10_task0_cfg_eval \
    runner.ckpt_path="$ck/global_step_${st}/actor/model_state_dict/full_weights.pt" \
    rollout.model.model_path=/workspace/models/pi05_base_pytorch \
    runner.logger.experiment_name="eval_${tag}" \
    rollout.model.openpi.cfgrl_guidance_scale="$g" > "$L" 2>&1
  local rc=$?
  local so ret
  so=$(grep -oE "success_once=[0-9.]+" "$L" | tail -1 | cut -d= -f2)
  ret=$(grep -oE "return=[0-9.]+"      "$L" | tail -1 | cut -d= -f2)
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$st" "$g" "${so:-NA}" "${ret:-NA}" "$rc" >> "$out"
  echo "    -> success_once=${so:-NA} return=${ret:-NA} rc=$rc"
  for pat in "[e]val_embodied_agent.py" "[r]un_eval.sh"; do
    for p in $(pgrep -f "$pat"); do kill "$p" 2>/dev/null; done
  done
  sleep 5; ray stop --force >/dev/null 2>&1; sleep 10
}

train_run () {  # config_name, exp_substring, logfile
  local cfgname=$1 exp=$2 tlog=$3
  local FREE; FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  echo "=== free before $cfgname: ${FREE}G ==="
  [ "$FREE" -lt 110 ] && { echo "ABORT $cfgname: need >=110G, have ${FREE}G"; return 1; }

  local RESUME EXTRA=""
  RESUME=$(newest_ckpt "$exp")
  if [ -n "$RESUME" ]; then
    echo "RESUMING $cfgname from $RESUME"
    EXTRA="runner.resume_dir=$RESUME"
  else
    echo "starting $cfgname from scratch"
  fi

  setsid bash /workspace/janitor5.sh "$exp" 2 >/dev/null 2>&1 &
  sleep 5
  echo "--- janitor ---"; head -4 "/workspace/janitor_${exp}.log" 2>/dev/null
  echo "=== $(date -u) launching $cfgname ${EXTRA:+with $EXTRA} ==="
  bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh "$cfgname" $EXTRA > "$tlog" 2>&1
  local rc=$?
  echo "=== $cfgname exited rc=$rc $(date -u) ==="
  for p in $(pgrep -f "[j]anitor5.sh"); do kill "$p" 2>/dev/null; done
  return $rc
}

eval_run_ckpts () {  # exp-substring, tagprefix, out_tsv
  local exp=$1 pref=$2 out=$3 CK FINAL
  CK=$(find /workspace/RLinf/logs -maxdepth 5 -type d -path "*${exp}*" -name checkpoints 2>/dev/null | head -1)
  echo "$pref checkpoints: $CK"; ls -d "$CK"/global_step_* 2>/dev/null
  [ -n "$CK" ] || { echo "no checkpoints for $exp"; return 1; }
  for st in $(ls -d "$CK"/global_step_* 2>/dev/null | sed 's#.*global_step_##' | sort -n); do
    grep -q "^${pref}_s${st}	" "$out" 2>/dev/null && { echo "  already evaluated $st, skipping"; continue; }
    python -c "
import zipfile,sys
try: zipfile.ZipFile('$CK/global_step_$st/actor/model_state_dict/full_weights.pt').namelist()
except Exception as e: print('  SKIP $st unreadable:', e); sys.exit(1)
" || continue
    run_eval "$CK" "$st" 1.0 "${pref}_s${st}" "$out"
  done
  FINAL=$(ls -d "$CK"/global_step_* 2>/dev/null | sed 's#.*global_step_##' | sort -n | tail -1)
  if [ -n "$FINAL" ] && ! grep -q "^${pref}_s${FINAL}_g1.5	" "$out" 2>/dev/null; then
    run_eval "$CK" "$FINAL" 1.5 "${pref}_s${FINAL}_g1.5" "$out"
  fi
}

############################# PHASE B: BC CONTROL #############################
BCRES=/workspace/bc_results.tsv
[ -s "$BCRES" ] || printf "tag\tstep\tguidance\tsuccess_once\treturn\trc\n" > "$BCRES"
if [ "$(wc -l < "$BCRES")" -gt 1 ]; then
  echo "=== PHASE B already has results, skipping train ==="
  column -t -s $'\t' "$BCRES"
else
  train_run cfg_a100b_bc_allpos a100b_bc_allpos /workspace/a100b_bc.log \
    && eval_run_ckpts a100b_bc_allpos bc "$BCRES"
fi
echo "=== PHASE B DONE $(date -u) ==="; column -t -s $'\t' "$BCRES" 2>/dev/null

############################# PRUNE #############################
OLDCK=$(find /workspace/RLinf/logs -maxdepth 5 -type d -path "*a100b_upstream_ep1*" -name checkpoints 2>/dev/null | head -1)
echo "=== prune before phase C, free before: $(df -h / | awk 'NR==2{print $4}') ==="
# Only checkpoints whose eval numbers are already recorded. NEVER global_step_3066
# (the headline, 0.512 over 5 repeats) or global_step_2500 (0.58).
for st in 2000 3000; do
  D="$OLDCK/global_step_$st"
  [ -d "$D" ] && { echo "deleting $D ($(du -sh "$D" | cut -f1))"; rm -rf "$D"; }
done
echo "kept: $(ls -d "$OLDCK"/global_step_* 2>/dev/null | tr '\n' ' ')"
echo "free after: $(df -h / | awk 'NR==2{print $4}')"

############################# PHASE C: CONSTANT LR #############################
CLRES=/workspace/constlr_results.tsv
[ -s "$CLRES" ] || printf "tag\tstep\tguidance\tsuccess_once\treturn\trc\n" > "$CLRES"
if [ "$(wc -l < "$CLRES")" -gt 1 ]; then
  echo "=== PHASE C already has results, skipping train ==="
else
  train_run cfg_a100b_constlr a100b_constlr /workspace/a100b_constlr.log \
    && eval_run_ckpts a100b_constlr constlr "$CLRES"
fi

echo "=== CHAIN4 DONE $(date -u) ==="
echo "--- BC ---";      column -t -s $'\t' "$BCRES" 2>/dev/null
echo "--- constlr ---"; column -t -s $'\t' "$CLRES" 2>/dev/null
