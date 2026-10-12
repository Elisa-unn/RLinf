"""Write an advantage parquet that marks every frame of a SUCCESSFUL episode
positive and every frame of a failed episode negative.

This is the filtered-behaviour-cloning arm: the classic strong offline-RL
baseline, "throw the failures away and imitate the successes", which needs no
value model, no advantage and no quantile.

Success is unambiguous in this dataset. Grouping the rollout split by episode,
`return` at the last frame takes exactly two values:

    -300.0   2097 episodes   (failed; the `fail300` tag means failures score -300)
       0.0   1999 episodes   (succeeded)

and 1999 / 4096 = 48.80%, which is the 48.8% "few-shot SFT baseline" the RECAP
tutorial reports. That number is the success rate of the rollout data itself.

The sft split is expert demonstrations, already all-positive, and is left alone.

CAVEAT, recorded deliberately: in CFG training the negative frames still train
the UNCONDITIONAL branch, so this is not literally "drop the failures". It is
"the conditional branch — the only branch inference uses at guidance 1.0 — sees
only successful frames". The failures still touch the shared trunk. Keeping them
is what makes all three arms differ in exactly one parquet column:

    bcctrl_allpos   every frame positive            -> BC on everything
    bcctrl_succonly every successful frame positive -> filtered BC
    fail300_..._q30 top 30% by advantage positive   -> RECAP
"""
import pandas as pd, pathlib, sys

SRC_TAG = "fail300_N10_ckpt3000_q30"
DST_TAG = "bcctrl_succonly"
ROOT = pathlib.Path("/workspace/datasets/RECAP-Libero10-Task0-48succ-Data")

for sp, mode in [("libero10_task0_sft", "all"), ("libero10_task0_train", "success")]:
    src = ROOT / sp / "meta" / f"advantages_{SRC_TAG}.parquet"
    dst = ROOT / sp / "meta" / f"advantages_{DST_TAG}.parquet"
    if not src.exists():
        sys.exit(f"MISSING {src}")
    df = pd.read_parquet(src)
    before = float(df["advantage"].mean())

    if mode == "all":
        df["advantage"] = True
        detail = "expert demos, all positive"
    else:
        last_ret = df.groupby("episode_index")["return"].last()
        succ = set(last_ret.index[last_ret == 0.0])
        fail = set(last_ret.index[last_ret == -300.0])
        other = set(last_ret.index) - succ - fail
        if other:
            sys.exit(f"UNEXPECTED return values for episodes {sorted(other)[:5]}")
        df["advantage"] = df["episode_index"].isin(succ)
        detail = (f"{len(succ)} succ / {len(succ)+len(fail)} eps "
                  f"= {100*len(succ)/(len(succ)+len(fail)):.2f}%")

    df.to_parquet(dst, index=False)
    chk = pd.read_parquet(dst)
    ok = (chk["advantage"].dtype == bool and len(chk) == len(df)
          and list(chk.columns) == list(df.columns)
          and bool((chk["advantage"].values == df["advantage"].values).all()))
    print(f"{sp}: rows={len(chk)}  true_rate {before:.4f} -> {chk['advantage'].mean():.4f}  "
          f"({detail})  {'OK' if ok else 'FAILED'}")
    if not ok:
        sys.exit("verification failed")

    if mode == "success":
        # the positive frames are what the conditional branch will see
        print(f"   positive frames: {int(chk['advantage'].sum()):,} of {len(chk):,}")
        # sanity: no successful episode should contain a negative frame and vice versa
        per_ep = chk.groupby("episode_index")["advantage"].nunique()
        print(f"   episodes with mixed labels (must be 0): {(per_ep > 1).sum()}")
print("done")
