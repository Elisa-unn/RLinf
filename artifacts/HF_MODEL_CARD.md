---
license: apache-2.0
tags:
  - robotics
  - vla
  - pi0.5
  - offline-rl
  - recap
  - rlt
  - libero
  - maniskill
---

# RLinf reproduction checkpoints — RECAP (LIBERO-10 Task 0) and RLT Stage 1 (ManiSkill)

Private backup of the three policies produced while reproducing two RLinf
recipes on 2x A100-80GB. The findings, every configuration, the advantage
parquets and the restart guide live in the repository this came from
(`RECAP_NOTES.md`, `RESTART.md`, `artifacts/`); **this repo is only the weights**,
which are too large to version.

## Contents

| Folder | Policy | LIBERO-10 Task 0, guidance 1.0 | n | GPU-hours |
| --- | --- | --- | --- | --- |
| `recap_advantage_weighted_0.512/` | RECAP Step 4, advantage-weighted CFG | **0.512** | 250 | 25 |
| `bc_on_everything_0.456/` | the same run with every frame labelled positive — behaviour cloning on the same data, failures included | **0.456** | 250 | 25 |
| `rlt_stage1_step2000/` | RLT Stage 1, ManiSkill `PegInsertionSideWideClearance-v1` | not evaluated (Stage 2 not run) | — | 13.4 |

Each folder holds:

- `full_weights.pt` — **this is what you need to evaluate or re-initialise.**
  `runner.ckpt_path` points straight at it, and RLT Stage 2's
  `actor.model.model_path` resolves to it as well.
- `dcp_checkpoint/` — distributed checkpoint with optimizer state, needed only
  to *resume training exactly* (`runner.resume_dir` points at the parent).

`rlt_stage1_step2000/` has only `full_weights.pt`: its 46 GiB of optimizer state
did not fit under the Hub free tier's private storage limit. That costs nothing
— the Stage 1 run finished at 2000/2000 so there is nothing to resume, and
Stage 2 reads `full_weights.pt`.

## What these three measure

All three start from `pi05_base` (which scores **0.00** on this task) and see the
same 1,561,874 frames of Task-0 rollouts for the same 3066 steps with the same
learning rate, batch size and schedule. They differ in exactly one boolean
column of the advantage parquet, and in one case in one boolean flag:

```
frozen VLM  (train_expert_only: True)        0.10          <- not in this repo
BC on everything                             0.456
RECAP, top 30% by advantage                  0.512
published                                    0.665
```

```
unfreezing the VLM     +41 points
advantage weighting    +5.6 points   (Welch t on run-level means, p = 0.043)
```

So **unfreezing the vision-language model accounts for roughly 7x what RECAP's
own value model and advantage machinery contribute** in this setup. The frozen
policy reached for the basket rather than the object the instruction names, and
scored 0.00 on Task 7 — which shares the alphabet soup, the basket and the
sentence form with Task 0 and differs only in the second object.

## Evaluating

```bash
bash evaluations/run_eval.sh libero libero_10_task0_cfg_eval \
    runner.ckpt_path=<folder>/full_weights.pt \
    rollout.model.model_path=<path>/pi05_base_pytorch \
    rollout.model.openpi.cfgrl_guidance_scale=1.0
```

`model_path` must still point at the converted base directory — the builder
reads the tokenizer and norm-stats assets from it.

Three things that are easy to get wrong:

- **Repeat the eval five times.** The per-run spread is 8 points; a single n=50
  number is not quotable. The first measurement of the 0.512 policy was 0.56,
  which turned out to be the maximum of five runs.
- **Placement must be `rollout: 0 / env: 1`.** Both on one GPU deadlocks
  silently on a container without `CAP_SYS_PTRACE`, because RLinf uses CUDA IPC
  for two workers sharing one accelerator.
- **`cfgrl_guidance_scale: 1.0` is not "guidance off".** The model computes
  `(1 - s) * v_uncond + s * v_cond`, so 1.0 is the pure conditional prediction
  and only `s > 1` extrapolates. Above ~1.5 the success rate drops.

## Reproduced from

[RLinf](https://github.com/RLinf/RLinf) — the
[RECAP](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)
and
[RLT](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html)
recipes, with `RLinf/RECAP-Libero10-Task0-48succ-Data` and
`RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ`.
