#!/bin/bash
# Supersedes chain2.sh. Order decided with the user: finish the repeat measurement,
# then the BEHAVIOUR-CLONING control (the missing baseline), then the constant-lr
# run. chain2 queued constant-lr first; BC answers the more important question.
#
#   A. reps 4 and 5 of the 0.56 checkpoint  (rep1-3 already recorded: .56 .48 .56)
#   B. BC control: identical config, advantage column all True   ~25 h
#   C. prune, then constant-lr run                               ~25 h
#
# Redirect FIRST: an earlier version put `set -u` and the venv activation above
# the redirect, `$PYTHONPATH` was unset, set -u killed it, and it left no log at
# all -- indistinguishable from never having started.
LOG=/workspace/chain3.log
exec >> "$LOG" 2>&1
echo "################ chain3 start $(date -u) ################"
set -u
cd /workspace/RLinf || exit 1
ulimit -n 65535
source /workspace/RLinf/.venv_recap/bin/activate
export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-}
echo "python: $(which python)"

# chain2 may have left an eval mid-flight
for pat in "[c]hain2.sh" "[e]val_embodied_agent.py" "[r]un_eval.sh"; do
  for p in $(pgrep -f "$pat"); do echo "killing leftover $pat pid $p"; kill "$p" 2>/dev/null; done
done
sleep 5; ray stop --force >/dev/null 2>&1; sleep 10

OLDCK=$(find /workspace/RLinf/logs -maxdepth 5 -type d -path "*a100b_upstream_ep1*" -name checkpoints | head -1)
echo "0.56-run checkpoints: $OLDCK"
[ -d "$OLDCK/global_step_3066" ] || { echo "ABORT: no global_step_3066"; exit 1; }

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
  printf "%s\t%s\t%s\t%s\t%s\n" "$tag" "$st" "$g" "${so:-NA}" "$rc" >> "$out"
  echo "    -> success_once=${so:-NA} rc=$rc"
  for pat in "[e]val_embodied_agent.py" "[r]un_eval.sh"; do
    for p in $(pgrep -f "$pat"); do kill "$p" 2>/dev/null; done
  done
  sleep 5; ray stop --force >/dev/null 2>&1; sleep 10
}

train_run () {  # config_name, exp_substring, logfile
  local cfgname=$1 exp=$2 tlog=$3
  local FREE; FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  echo "=== free before $cfgname: ${FREE}G ==="
  if [ "$FREE" -lt 110 ]; then
    echo "ABORT $cfgname: need >=110G for 3 x 32G concurrent checkpoints, have ${FREE}G"
    return 1
  fi
  setsid bash /workspace/janitor5.sh "$exp" 2 >/dev/null 2>&1 &
  sleep 5
  echo "--- janitor log head ---"; head -4 "/workspace/janitor_${exp}.log" 2>/dev/null
  echo "=== $(date -u) launching $cfgname (3066 steps, ~25 h) ==="
  bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh "$cfgname" > "$tlog" 2>&1
  echo "=== $cfgname exited rc=$? $(date -u) ==="
  for p in $(pgrep -f "[j]anitor5.sh"); do kill "$p" 2>/dev/null; done
  echo "janitor stopped"
}

eval_run_ckpts () {  # exp_substring, tagprefix, out_tsv
  local exp=$1 pref=$2 out=$3
  local CK; CK=$(find /workspace/RLinf/logs -maxdepth 5 -type d -path "*${exp}*" -name checkpoints | head -1)
  echo "$pref checkpoints: $CK"
  ls -d "$CK"/global_step_* 2>/dev/null
  [ -n "$CK" ] || { echo "no checkpoints for $exp"; return 1; }
  for st in $(ls -d "$CK"/global_step_* 2>/dev/null | sed 's#.*global_step_##' | sort -n); do
    python -c "
import zipfile,sys
try: zipfile.ZipFile('$CK/global_step_$st/actor/model_state_dict/full_weights.pt').namelist()
except Exception as e: print('  SKIP $st unreadable:', e); sys.exit(1)
" || continue
    run_eval "$CK" "$st" 1.0 "${pref}_s${st}" "$out"
  done
  # the headline number also at guidance 1.5, for comparability with the 0.56 run
  local FINAL; FINAL=$(ls -d "$CK"/global_step_* 2>/dev/null | sed 's#.*global_step_##' | sort -n | tail -1)
  [ -n "$FINAL" ] && run_eval "$CK" "$FINAL" 1.5 "${pref}_s${FINAL}_g1.5" "$out"
}

############################# PHASE A #############################
REP=/workspace/repeat_results.tsv
for i in 4 5; do
  printf "rep${i}\t3066\t1.0\t" >> /dev/null   # keep columns aligned with chain2's header
  run_eval "$OLDCK" 3066 1.0 "rep${i}" "$REP"
done
echo "=== PHASE A DONE $(date -u) ==="
column -t -s $'\t' "$REP"
python - <<'PY'
import csv, math
rows=list(csv.DictReader(open('/workspace/repeat_results.tsv'),delimiter='\t'))
v=[float(r['success_once']) for r in rows if r.get('success_once') not in (None,'NA','')]
n=50
print('repeats:', v)
if len(set(v))==1: print('WARNING: all repeats identical -> eval deterministic, CI invalid')
k=sum(round(x*n) for x in v); N=n*len(v); p=k/N; z=1.96
d=1+z*z/N; c=(p+z*z/(2*N))/d; hw=z*math.sqrt(p*(1-p)/N+z*z/(4*N*N))/d
print(f'pooled {k}/{N} = {p:.4f}   Wilson 95% CI [{c-hw:.4f}, {c+hw:.4f}]')
print('CI contains 0.665:', c-hw <= 0.665 <= c+hw)
pr=[math.comb(N,i)*0.665**i*(1-0.665)**(N-i) for i in range(N+1)]
print(f'two-sided binomial vs 0.665: p = {sum(x for x in pr if x<=pr[k]+1e-18):.4g}')
print(f'per-run spread: min {min(v):.2f} max {max(v):.2f}  range {max(v)-min(v):.2f}')
PY

############################# PHASE B: BC CONTROL #############################
BCRES=/workspace/bc_results.tsv
printf "tag\tstep\tguidance\tsuccess_once\trc\n" > "$BCRES"
train_run cfg_a100b_bc_allpos a100b_bc_allpos /workspace/a100b_bc.log \
  && eval_run_ckpts a100b_bc_allpos bc "$BCRES"
echo "=== PHASE B DONE $(date -u) ==="; column -t -s $'\t' "$BCRES" 2>/dev/null

############################# PRUNE #############################
# Two 25 h runs cannot both hold checkpoints alongside the 0.56 run's four.
# Delete only checkpoints whose eval numbers are already recorded, and never
# global_step_3066 (the headline) or global_step_2500 (0.58, the plateau point).
echo "=== prune before phase C, free before: $(df -h / | awk 'NR==2{print $4}') ==="
for st in 2000 3000; do
  D="$OLDCK/global_step_$st"
  if [ -d "$D" ]; then echo "deleting $D ($(du -sh "$D" | cut -f1))"; rm -rf "$D"; fi
done
echo "kept: $(ls -d "$OLDCK"/global_step_* 2>/dev/null | tr '\n' ' ')"
echo "free after: $(df -h / | awk 'NR==2{print $4}')"

############################# PHASE C: CONSTANT LR #############################
CLRES=/workspace/constlr_results.tsv
printf "tag\tstep\tguidance\tsuccess_once\trc\n" > "$CLRES"
train_run cfg_a100b_constlr a100b_constlr /workspace/a100b_constlr.log \
  && eval_run_ckpts a100b_constlr constlr "$CLRES"

echo "=== CHAIN3 DONE $(date -u) ==="
echo "--- repeats ---";  column -t -s $'\t' "$REP"    2>/dev/null
echo "--- BC ---";       column -t -s $'\t' "$BCRES"  2>/dev/null
echo "--- constlr ---";  column -t -s $'\t' "$CLRES"  2>/dev/null
