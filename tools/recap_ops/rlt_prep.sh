#!/bin/bash
# RLT prep. Non-GPU work (plus one tiny env smoke test) so that when the RECAP
# queue drains, Stage 1 starts with zero idle time. This box has idled three
# times waiting for a human; prep is the cheapest way to not idle a fourth.
LOG=/workspace/rlt_prep.log
exec >> "$LOG" 2>&1
echo "################ rlt_prep start $(date -u) ################"
set -u
cd /workspace/RLinf || exit 1
source /workspace/RLinf/.venv_recap/bin/activate
unset HF_HUB_ENABLE_HF_TRANSFER        # see dl_rlt.sh
export HF_LEROBOT_HOME=/workspace/datasets/lerobot
export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-}
DEST="${HF_LEROBOT_HOME}/maniskill_peginsertionside_joint"

# ---- 1. wait for the download, then verify the ARTIFACT not the exit code ----
echo "=== waiting for download ==="
W=0
while ps -eo args --no-headers | awk '$NF=="/workspace/dl_rlt.sh"' | grep -q . ; do
  sleep 60; W=$((W+1))
  [ $((W % 10)) -eq 0 ] && echo "  ${W} min, $(du -sh "$DEST" 2>/dev/null | cut -f1)"
  [ $W -gt 180 ] && { echo "ABORT: download still running after 3 h"; exit 1; }
done
echo "download process gone $(date -u)"
N=$(find "$DEST/data" -name "episode_*.parquet" 2>/dev/null | wc -l)
echo "episode parquets: $N   size: $(du -sh "$DEST" | cut -f1)"
if [ "$N" -lt 390 ]; then echo "ABORT: expected ~400 episode parquets, found $N"; exit 1; fi
for p in meta/info.json meta/episodes.jsonl; do
  [ -e "$DEST/$p" ] || { echo "ABORT: missing $p"; exit 1; }
done
# every parquet must actually parse -- a resumed download once left a half-written
# file that passed an existence check and failed later with OUT_OF_RANGE
python - <<'PY' || exit 1
import glob, os, sys, pandas as pd
dest = os.environ["HF_LEROBOT_HOME"] + "/maniskill_peginsertionside_joint"
fs = sorted(glob.glob(dest + "/data/**/episode_*.parquet", recursive=True))
bad = []
rows = 0
for f in fs:
    try:
        rows += len(pd.read_parquet(f, columns=None))
    except Exception as e:
        bad.append((os.path.basename(f), type(e).__name__))
print(f"  parsed {len(fs)-len(bad)}/{len(fs)} parquets, {rows:,} frames total")
if bad:
    print("  UNPARSEABLE:", bad[:10]); sys.exit(1)
df = pd.read_parquet(fs[0])
print("  columns:", list(df.columns))
PY
echo "=== DOWNLOAD VERIFIED $(date -u) ==="

# ---- 2. norm stats ----------------------------------------------------------
if [ -f "$DEST/norm_stats.json" ]; then
  echo "=== norm_stats.json already present, skipping ==="
else
  echo "=== computing norm stats $(date -u) ==="
  python toolkits/lerobot/calculate_norm_stats.py \
    --config-name pi05_rlt_maniskill_joint \
    --repo-id maniskill_peginsertionside_joint
  echo "  rc=$?"
fi
echo "--- where did it land? ---"
find /workspace/datasets/lerobot /root/.cache/openpi "$HOME" -name "norm_stats.json" -newermt '-6 hours' 2>/dev/null | head
python - <<'PY'
import json, glob, os
hits = glob.glob("/workspace/datasets/lerobot/**/norm_stats.json", recursive=True)
hits += glob.glob(os.path.expanduser("~") + "/.cache/openpi/**/norm_stats.json", recursive=True)
if not hits:
    print("  NO norm_stats.json FOUND -- stage 1 will need openpi_data.norm_stats_path fixed")
for h in hits:
    try:
        d = json.load(open(h))
        print(f"  {h}  keys={list(d)[:6]}")
        for k in list(d)[:3]:
            v = d[k]
            if isinstance(v, dict):
                print(f"     {k}: {list(v)[:6]}")
    except Exception as e:
        print(f"  {h}  UNREADABLE {e}")
PY
echo "=== NORM STATS PHASE DONE $(date -u) ==="

# ---- 3. env smoke test (small; BC training is using ~75 of 80 GiB per card) --
echo "=== env smoke test $(date -u) ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python - <<'PY'
import traceback
try:
    import torch, gymnasium as gym, mani_skill.envs  # noqa
    from rlinf.envs.maniskill.peg_insertion_side_variants import (
        register_rlinf_peg_insertion_side_variants as reg)
    reg()
    ids = [k for k in gym.registry if "PegInsertion" in k]
    print("  registered:", ids)
    assert "PegInsertionSideWideClearance-v1" in ids
    env = gym.make("PegInsertionSideWideClearance-v1", num_envs=2,
                   obs_mode="rgb", control_mode="pd_joint_delta_pos",
                   sim_backend="gpu")
    obs, info = env.reset(seed=0)
    print("  reset OK. action_space:", env.action_space)
    def shapes(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items(): shapes(v, p + "/" + k)
        else:
            print(f"    {p:40s} {tuple(getattr(o,'shape',()))} {getattr(o,'dtype',None)}")
    shapes(obs)
    a = env.action_space.sample()
    for i in range(3):
        obs, rew, term, trunc, info = env.step(a)
    print("  3 steps OK. reward:", rew)
    print("  SMOKE TEST PASSED")
    env.close()
except Exception:
    print("  SMOKE TEST FAILED")
    traceback.print_exc()
PY
echo "################ rlt_prep done $(date -u) ################"
