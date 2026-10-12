#!/bin/bash
# Download the RLT ManiSkill joint-control LeRobot dataset.
#
# Deliberately does NOT set HF_HUB_ENABLE_HF_TRANSFER: with it set and
# hf_transfer absent from the venv, the hub REFUSES every download instead of
# falling back, which once left install.sh exiting 0 with nothing installed.
# Unset it here in case it is exported globally.
exec >> /workspace/dl_rlt.log 2>&1
echo "=== download start $(date -u) ==="
unset HF_HUB_ENABLE_HF_TRANSFER
set -u
source /workspace/RLinf/.venv_recap/bin/activate

export HF_LEROBOT_HOME=/workspace/datasets/lerobot
DEST="${HF_LEROBOT_HOME}/maniskill_peginsertionside_joint"
mkdir -p "$DEST"
echo "dest: $DEST"

huggingface-cli download RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ \
  --repo-type dataset --local-dir "$DEST"
RC=$?
echo "=== huggingface-cli rc=$RC $(date -u) ==="

# verify the ARTIFACT, not the exit code
echo "--- size ---"; du -sh "$DEST"
echo "--- expected layout ---"
for p in meta/info.json meta/episodes.jsonl data; do
  if [ -e "$DEST/$p" ]; then echo "  OK   $p"; else echo "  MISS $p"; fi
done
echo "--- episode parquet count (expect ~400) ---"
find "$DEST/data" -name "episode_*.parquet" 2>/dev/null | wc -l
echo "--- can one be read, and does it have the required fields? ---"
python - <<'PY'
import glob, pandas as pd, json, os
dest = os.environ["HF_LEROBOT_HOME"] + "/maniskill_peginsertionside_joint"
fs = sorted(glob.glob(dest + "/data/**/episode_*.parquet", recursive=True))
print("  parquet files:", len(fs))
if fs:
    df = pd.read_parquet(fs[0])
    print("  first file:", os.path.basename(fs[0]), "rows", len(df))
    print("  columns:", list(df.columns))
    need = ["state", "actions"]
    for n in need:
        hit = [c for c in df.columns if n in c]
        print(f"   {n!r:12s} -> {hit}")
    for c in df.columns:
        v = df[c].iloc[0]
        try: print(f"   {c:28s} shape={getattr(v,'shape',None)} type={type(v).__name__}")
        except Exception: pass
try:
    print("  info.json:", json.dumps(json.load(open(dest+"/meta/info.json")), indent=2)[:900])
except Exception as e:
    print("  info.json unreadable:", e)
PY
echo "=== DOWNLOAD PHASE DONE $(date -u) ==="
