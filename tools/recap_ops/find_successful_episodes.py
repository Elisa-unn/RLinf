import pandas as pd, numpy as np, pathlib, json
ROOT = pathlib.Path("/workspace/datasets/RECAP-Libero10-Task0-48succ-Data")
sp = "libero10_task0_train"
df = pd.read_parquet(ROOT/sp/"meta"/"advantages_fail300_N10_ckpt3000_q30.parquet")
print("rows", len(df), "episodes", df.episode_index.nunique())

g = df.groupby("episode_index")
summary = pd.DataFrame({
    "n_frames":   g.size(),
    "ret_first":  g["return"].first(),
    "ret_last":   g["return"].last(),
    "ret_max":    g["return"].max(),
    "ret_min":    g["return"].min(),
    "rsr_first":  g["reward_sum_raw"].first(),
    "rsr_last":   g["reward_sum_raw"].last(),
    "adv_rate":   g["advantage"].mean(),
})
print("\n=== per-episode summary describe ===")
print(summary.describe().to_string())
print("\n=== ret_last value counts (top 15) ===")
print(summary.ret_last.value_counts().head(15).to_string())
print("\n=== ret_max value counts (top 15) ===")
print(summary.ret_max.value_counts().head(15).to_string())
print("\n=== reward_sum_raw last: value counts ===")
print(summary.rsr_last.value_counts().head(15).to_string())
print("\n=== candidate success rule: ret_max == 0 or close ===")
for thr in [0.0, -1.0, -5.0, -10.0, -20.0]:
    m = summary.ret_max >= thr
    print(f"  ret_max >= {thr:7.1f}:  {m.sum():5d} / {len(summary)} episodes  ({100*m.mean():.1f}%)  frames {summary.n_frames[m].sum():,}")
print("\n=== is there episode metadata with success? ===")
for p in sorted((ROOT/sp/"meta").glob("*")):
    print("  ", p.name, p.stat().st_size)
