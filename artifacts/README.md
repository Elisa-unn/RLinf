# Preserved artifacts

Everything from this project that is **expensive to regenerate and small enough
to version**. The large checkpoints are not here — see `RESTART.md` for what was
lost and what it costs to rebuild.

## What is here

| Path | Bytes | What it cost to produce |
| --- | --- | --- |
| `recap/returns_fail300.parquet` | 1,088,783 | RECAP Step 1, minutes |
| `recap/advantages_fail300_N10_ckpt3000_q30.parquet` | 30,502,659 | **Step 2 (21 h) + Step 3 (5 h 21 m)** |
| `recap/advantages_bcctrl_allpos.parquet` | 29,918,138 | derived, minutes (`tools/recap_ops/mk_allpos.py`) |
| `recap/advantages_bcctrl_succonly.parquet` | 29,920,701 | derived, minutes (`tools/recap_ops/mk_succonly.py`) |
| `recap/sft/*.parquet` | ~205 KB each | the same, for the 8005-frame sft split |
| `results/*.tsv` | < 1 KB each | every evaluation number in `RECAP_NOTES.md` |
| `all_logs.tar.gz` | 730,946 | all 46 training and eval logs |
| `tensorboard.tar.gz` | 2,204,540 | 23 event files — the full training curves |
| `norm_stats.json` | 2,149 | ships with the RLT dataset; kept so the path in the config can be reproduced |

**The advantage parquet is the single most valuable file here.** It is 26 GPU-hours
of value in 30 MB: a 898M-parameter value model trained for 3000 steps, then used
to label 1,561,874 frames. Regenerating it means redoing RECAP Steps 1-3.

## Which parquet is which

All three are the same 1,561,874 rows and the same columns. They differ in one
boolean column, `advantage`, which is the only thing RECAP Step 4 reads from
them (`advantages_lookup: dict[tuple[int, int], bool]` in
`rlinf/data/datasets/recap/cfg_model.py`):

| tag | `advantage` true-rate | meaning | result |
| --- | --- | --- | --- |
| `fail300_N10_ckpt3000_q30` | 0.3000 | top 30% by TD advantage | RECAP, **0.512** |
| `bcctrl_allpos` | 1.0000 | every frame | BC on everything, **0.456** |
| `bcctrl_succonly` | 0.3555 | every frame of the 1999 successful episodes | filtered BC, **never run** |

To use one, copy it to
`<dataset>/libero10_task0_{train,sft}/meta/advantages_<tag>.parquet` and set
`data.advantage_tag: <tag>`.

## A fact worth not rediscovering

Grouping the rollout split by episode, `return` at the last frame takes exactly
two values: `-300.0` for 2097 episodes and `0.0` for 1999. **1999 / 4096 =
48.80%, which is the 48.8% the RECAP tutorial reports as its baseline** — that
published number is the success rate of this rollout data itself.
`tools/recap_ops/find_successful_episodes.py` reproduces it.

## Reading the parquets

`pyarrow==17.0.0` with `numpy==1.26.4`. A fresh install resolves pyarrow 26,
which requires numpy >= 2, while gym pins numpy < 2 — the result is that these
files become **completely unreadable**. Pin pyarrow.
