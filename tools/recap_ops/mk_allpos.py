"""Write an advantage parquet whose `advantage` column is all True.

This turns RECAP Step 4 into plain behaviour cloning on the same data: with no
negative frames there is nothing to route to the unconditional branch beyond the
`unconditional_prob` dropout, so the conditional branch — the one inference uses
at guidance 1.0 — is trained on every frame regardless of its advantage. Every
other knob (model, unfreezing, steps, lr, batch, schedule) is untouched, so the
difference against the 0.56 run isolates the advantage weighting itself.

Only the boolean column is rewritten; advantage_continuous and the value columns
are left in place so the file stays diffable against the source.
`relabel_advantages.py` cannot do this: it requires 0 < positive_quantile < 1.
"""
import pandas as pd, pathlib, sys

SRC_TAG = "fail300_N10_ckpt3000_q30"
DST_TAG = "bcctrl_allpos"
ROOT = pathlib.Path("/workspace/datasets/RECAP-Libero10-Task0-48succ-Data")

splits = ["libero10_task0_sft", "libero10_task0_train"]
for sp in splits:
    src = ROOT / sp / "meta" / f"advantages_{SRC_TAG}.parquet"
    dst = ROOT / sp / "meta" / f"advantages_{DST_TAG}.parquet"
    if not src.exists():
        sys.exit(f"MISSING {src}")
    df = pd.read_parquet(src)
    before = float(df["advantage"].mean())
    df["advantage"] = True
    df.to_parquet(dst, index=False)
    chk = pd.read_parquet(dst)
    ok = (chk["advantage"].dtype == bool and bool(chk["advantage"].all())
          and len(chk) == len(df) and list(chk.columns) == list(df.columns))
    print(f"{sp}: rows={len(chk)} true_rate {before:.4f} -> {chk['advantage'].mean():.4f} "
          f"dtype={chk['advantage'].dtype} cols_match={list(chk.columns)==list(df.columns)} "
          f"{'OK' if ok else 'FAILED'}")
    if not ok:
        sys.exit("verification failed")
print("all splits written and verified")
