#!/bin/bash
# FINAL PHASE. Decided on cost grounds: kill the filtered-BC run and spend the
# 30 minutes that actually carries the conclusion instead of the 24 hours that
# cannot change it.
#
# WHY filtered BC was dropped: it is a STRONGER baseline than BC-all, so it must
# land >= 0.46, and RECAP is 0.512. Any value in [0.46, 0.57] leaves the finding
# ("advantage weighting buys 0-5 points over plain BC") unchanged. It was worth
# running before BC-all came back; BC-all answered the question it was for.
#
# WHY the repeats were kept: the finding's one weakness is that BC-all has n=50
# against RECAP's n=250, so "p = 0.538, indistinguishable" is currently partly a
# measurement failure rather than a result. Four more evals halve BC-all's error
# bar (+/-0.133 -> +/-0.061) for ~30 minutes.
LOG=/workspace/chain6.log
exec >> "$LOG" 2>&1
echo "################ chain6 start $(date -u) ################"
set -u
cd /workspace/RLinf || exit 1
ulimit -n 65535
source /workspace/RLinf/.venv_recap/bin/activate
export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-}

# ---- 1. stop everything, supervisor first so nothing relaunches -------------
echo "=== stopping the pipeline ==="
supervisorctl stop recap-chain 2>&1 | tail -1
# chain5 before the training it owns, or it will react to the training dying
for pat in "[c]hain5.sh" "[c]hain4.sh"; do
  for p in $(pgrep -f "$pat"); do echo "  kill $pat pid $p"; kill "$p" 2>/dev/null; done
done
sleep 3
for p in $(pgrep -f "[j]anitor5.sh"); do echo "  kill janitor pid $p"; kill "$p" 2>/dev/null; done
for p in $(pgrep -f "[t]rain_cfg.py"); do echo "  kill training pid $p"; kill "$p" 2>/dev/null; done
sleep 20
for p in $(pgrep -f "[t]rain_cfg.py"); do echo "  SIGKILL leftover training pid $p"; kill -9 "$p" 2>/dev/null; done
ray stop --force >/dev/null 2>&1; sleep 15
echo "--- after stop ---"
for s in chain4 chain5; do
  printf "  %s=%s\n" "$s" "$(ps -eo args --no-headers | awk -v p=/workspace/$s.sh '$NF==p' | wc -l)"
done
echo "  train=$(ps -eo args --no-headers | grep -c 'train_cfg[.]py')"
echo "  filtered BC reached step: $(grep -cE 'train/loss=' /workspace/a100b_succonly.log 2>/dev/null) of 3066 (abandoned on purpose)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# ---- 2. four more repeats of BC-all's final checkpoint ----------------------
CK=$(find /workspace/RLinf/logs -maxdepth 5 -type d -path "*a100b_bc_allpos*" -name checkpoints 2>/dev/null | head -1)
echo "=== BC-all checkpoints: $CK ==="
ls -d "$CK"/global_step_* 2>/dev/null
D="$CK/global_step_3066"
python -c "
import zipfile,sys
try: print('  zip central dir OK, entries =', len(zipfile.ZipFile('$D/actor/model_state_dict/full_weights.pt').namelist()))
except Exception as e: print('  ABORT, unreadable:', e); sys.exit(1)
" || exit 1

REP=/workspace/bc_repeat_results.tsv
printf "tag\tstep\tguidance\tsuccess_once\treturn\trc\n" > "$REP"
printf "bcrep1(prior)\t3066\t1.0\t0.46\t2.3\t0\n" >> "$REP"
for i in 2 3 4 5; do
  L=/workspace/eval_bcrep${i}.log
  echo "=== $(date -u +%H:%M:%S) BC-all repeat $i ==="
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  timeout 1800 bash evaluations/run_eval.sh libero libero_10_task0_cfg_eval \
    runner.ckpt_path="$D/actor/model_state_dict/full_weights.pt" \
    rollout.model.model_path=/workspace/models/pi05_base_pytorch \
    runner.logger.experiment_name="eval_bcrep${i}" \
    rollout.model.openpi.cfgrl_guidance_scale=1.0 > "$L" 2>&1
  rc=$?
  so=$(grep -oE "success_once=[0-9.]+" "$L" | tail -1 | cut -d= -f2)
  ret=$(grep -oE "return=[0-9.]+"      "$L" | tail -1 | cut -d= -f2)
  printf "bcrep%s\t3066\t1.0\t%s\t%s\t%s\n" "$i" "${so:-NA}" "${ret:-NA}" "$rc" >> "$REP"
  echo "    -> success_once=${so:-NA} rc=$rc"
  for pat in "[e]val_embodied_agent.py" "[r]un_eval.sh"; do
    for p in $(pgrep -f "$pat"); do kill "$p" 2>/dev/null; done
  done
  sleep 5; ray stop --force >/dev/null 2>&1; sleep 10
done

echo "=== REPEATS DONE $(date -u) ==="
column -t -s $'\t' "$REP"
python - <<'PY'
import csv, math
from math import comb
rows=list(csv.DictReader(open('/workspace/bc_repeat_results.tsv'),delimiter='\t'))
v=[float(r['success_once']) for r in rows if r.get('success_once') not in (None,'NA','')]
n=50; R=len(v); k=sum(round(x*n) for x in v); N=n*R
print('BC-all repeats:', v)
import statistics as st
if R>1:
    m=st.mean(v); sd=st.stdev(v); se=sd/math.sqrt(R)
    print(f'  repeat-level: mean {m:.4f}  sd {sd:.4f}  SE {se:.4f}  95% CI(t,df={R-1}) [{m-2.776*se:.4f}, {m+2.776*se:.4f}]')
p=k/N; z=1.96; d=1+z*z/N; c=(p+z*z/(2*N))/d
hw=z*math.sqrt(p*(1-p)/N+z*z/(4*N*N))/d
print(f'  pooled {k}/{N} = {p:.4f}  Wilson 95% CI [{c-hw:.4f}, {c+hw:.4f}]')
def fisher(a,b,c_,d_):
    tot=comb(a+b+c_+d_,a+c_); pr=lambda x: comb(a+b,x)*comb(c_+d_,a+c_-x)/tot
    obs=pr(a); return sum(pr(x) for x in range(0,a+b+1) if 0<=a+c_-x<=c_+d_ and pr(x)<=obs+1e-18)
print()
print('=== THE COMPARISON ===')
print(f'  RECAP  128/250 = 0.5120')
print(f'  BC-all {k}/{N} = {p:.4f}')
print(f'  Fisher two-sided p = {fisher(128,122,k,N-k):.4f}')
print(f'  difference = {0.512-p:+.4f}')
PY

# ---- 3. make sure nothing restarts and starts billing ----------------------
echo "=== disarming the supervisor so a container restart costs nothing ==="
rm -f /etc/supervisor/conf.d/recap-chain.conf
supervisorctl reread 2>&1 | tail -1
supervisorctl update 2>&1 | tail -1
supervisorctl status recap-chain 2>&1 | tail -1
echo "--- final process check (all should be 0) ---"
echo "  train=$(ps -eo args --no-headers | grep -c 'train_cfg[.]py')"
echo "  eval=$(ps -eo args --no-headers | grep -c 'eval_embodied_agent[.]py')"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
df -h / | tail -1
echo "################ chain6 done $(date -u) ################"
