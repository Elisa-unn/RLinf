# Restarting this work on a fresh machine

Written for whoever picks this up later, including me. It says what survives in
this repository, what does not, what the missing pieces cost in GPU-hours, and
the order to rebuild.

`RECAP_NOTES.md` is the findings. This file is the logistics.

---

## 1. What survives here, and therefore never needs redoing

| | Where |
| --- | --- |
| Every finding, number, error bar and correction | `RECAP_NOTES.md` |
| **Advantage + returns parquets** — 26 GPU-hours in 92 MB | `artifacts/recap/` |
| Every evaluation number as raw TSV | `artifacts/results/` |
| All 46 logs, 23 tensorboard event files | `artifacts/*.tar.gz` |
| Every config actually used, annotated with its measurements | `examples/offline_rl/config/cfg_a100b_*.yaml`, `examples/sft/config/maniskill_rlt_stage1_a100.yaml`, `evaluations/libero/libero_10_task0_cfg_eval.yaml` |
| The operational scripts and the bug each one avoids | `tools/recap_ops/` |
| The one framework behaviour fix | `rlinf/workers/actor/fsdp_sac_policy_worker.py` (`save_with_checkpoint`) |
| The Hub model card, as published | `artifacts/HF_MODEL_CARD.md` |
| The partial-unfreeze option | `rlinf/models/embodiment/openpi_cfg/` (`unfreeze_last_n_vlm_layers`) |

**Because the advantage parquet survived, RECAP Steps 1-3 never have to run
again.** Step 4 reads only the boolean `advantage` column from it.

---

## 2. What does not survive, and what it costs

Model weights are 32-62 GiB each and cannot be versioned, so they went to a
**private Hub repo instead**:

### `Elisals/rlinf-recap-rlt-checkpoints` (private)

| Folder | Policy | Task 0 @ g=1.0 | GPU-hours |
| --- | --- | --- | --- |
| `recap_advantage_weighted_0.512/` | RECAP Step 4, advantage-weighted CFG | **0.512** (n=250) | 25 |
| `bc_on_everything_0.456/` | the same run with every frame positive | **0.456** (n=250) | 25 |
| `rlt_stage1_step2000/` | RLT Stage 1, ManiSkill | not evaluated | 13.4 |

Verified against the local files byte for byte, 78.8 GiB in 11 files:

```
         2,303,958  recap_advantage_weighted_0.512/dcp_checkpoint/.metadata
    13,048,039,511  recap_advantage_weighted_0.512/dcp_checkpoint/__0_0.distcp
    12,538,527,191  recap_advantage_weighted_0.512/dcp_checkpoint/__1_0.distcp
     8,526,557,492  recap_advantage_weighted_0.512/full_weights.pt      -> 34,115,428,152  = local
         2,303,952  bc_on_everything_0.456/dcp_checkpoint/.metadata
    13,048,039,511  bc_on_everything_0.456/dcp_checkpoint/__0_0.distcp
    12,538,527,191  bc_on_everything_0.456/dcp_checkpoint/__1_0.distcp
     8,526,557,492  bc_on_everything_0.456/full_weights.pt              -> 34,115,428,146  = local
    16,401,038,312  rlt_stage1_step2000/full_weights.pt                 -> optimizer state absent
```

**The distinction between the two parts matters, and it is why the one gap is
harmless:**

- `full_weights.pt` — what `runner.ckpt_path` wants, and also what RLT Stage 2's
  `actor.model.model_path` resolves to (`resolve_full_weights` in
  `openpi_rlinf/utils/rlt_utils.py`; the `get_model` docstring says a model path
  may be "an RLinf FSDP SFT checkpoint containing `full_weights.pt`"). **Enough
  to evaluate a policy, initialise a new run, or feed Stage 2.**
- `dcp_checkpoint/` — optimizer state for `runner.resume_dir`. Needed **only to
  resume that exact run**, and three times larger.

**RLT Stage 1's optimizer state (46 GiB) is missing**, because the Hub free tier
stops at ~79 GiB of private storage:

```
403 Forbidden: Private storage limit reached for user Elisals
```

That costs nothing here: the Stage 1 run finished at 2000/2000 so there is
nothing to resume, and Stage 2 reads `full_weights.pt`, which is preserved. If
it is ever wanted, re-upload it after freeing space or upgrading the plan.

A restart that only wants to evaluate or branch from these policies downloads a
quarter of the bytes.

### Still not preserved, and what it costs

| | Size | Cost |
| --- | --- | --- |
| `pi05_base_pytorch` | 6.8 GiB | ~2 h (download + convert); the exact verified command is in `RECAP_NOTES.md` section 1.2 |
| RECAP dataset | 82 GiB | download from `RLinf/RECAP-Libero10-Task0-48succ-Data` |
| RLT dataset | 9.2 GiB | download from `RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ` |
| RLT Stage 1 step 1500 | 62 GiB | intermediate only; step 2000 is preserved |
| frozen-VLM policies (0.10, 0.16) | 19 GiB each | the machine holding them was lost; **superseded anyway** — their only role was as the frozen arm of the A/B |

The three policies that carry every result in `RECAP_NOTES.md` are safe. Nothing
left unpreserved costs more than a download plus ~2 h of conversion.

> Three machines were lost during this project before this backup existed. If a
> future run produces a checkpoint worth keeping, push it the day it finishes.

---

## 3. Rebuild order

### 3.1 Environment

Follow `RECAP_NOTES.md` section 5. The three install traps, each of which cost a
full rebuild:

1. **Unset `HF_HUB_ENABLE_HF_TRANSFER` for the whole install.** A fresh venv has
   no `hf_transfer`, and the hub then *refuses* every download rather than
   falling back — `install.sh` exits 0 with openpi and mani_skill never
   installed. Install only `hf_transfer`, never `huggingface_hub[hf_transfer]`,
   which drags in a hub version that breaks transformers 4.53.
2. **Pin `pyarrow==17.0.0`** with `numpy==1.26.4`. Otherwise pyarrow 26 is
   resolved, it needs numpy >= 2, gym forbids numpy 2, and `artifacts/recap/`
   becomes unreadable.
3. **Assert the artifacts, not the exit code:**
   `python -c "import openpi, libero, mani_skill"` must pass.

### 3.2 RECAP — skip Steps 1-3 entirely

```bash
# copy the preserved labels back in
for sp in libero10_task0_train libero10_task0_sft; do
  cp artifacts/recap/advantages_*.parquet  <dataset>/$sp/meta/   # train split
done   # the sft split's three files are in artifacts/recap/sft/
cp artifacts/recap/returns_fail300.parquet <dataset>/libero10_task0_train/meta/
```

Then Step 4 directly, choosing the arm by `data.advantage_tag`:

| config | `advantage_tag` | reproduces |
| --- | --- | --- |
| `cfg_a100b_upstream_ep1.yaml` | `fail300_N10_ckpt3000_q30` | RECAP, 0.512 |
| `cfg_a100b_bc_allpos.yaml` | `bcctrl_allpos` | BC on everything, 0.456 |
| `cfg_a100b_bc_succonly.yaml` | `bcctrl_succonly` | filtered BC — **built and verified, never run** |

25 h each on 2x A100-80GB. **Run `tools/recap_ops/janitor5.sh <exp> 2`** or the
disk fills: these checkpoints are 32 GiB and `save_interval` is 500.

Evaluate with `evaluations/libero/libero_10_task0_cfg_eval.yaml`, placement
`rollout: 0 / env: 1` (both on one GPU deadlocks — section 1.4), and
**repeat each eval 5 times**: the per-run spread is 8 points and a single n=50
number is not quotable (section 3.11).

### 3.3 RLT — Stage 1 only

```bash
export HF_LEROBOT_HOME=<root>
huggingface-cli download RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ \
    --repo-type dataset --local-dir $HF_LEROBOT_HOME/maniskill_peginsertionside_joint
```

It ships its own `norm_stats.json`, so `calculate_norm_stats.py` is **not**
needed. `tools/recap_ops/rlt_prep.sh` verifies the download and smoke-tests the
env.

Then `examples/sft/config/maniskill_rlt_stage1_a100.yaml`, which already carries
the two fixes this took:

- **`gradient_checkpointing: True`.** Upstream ships `False`, which OOMs at
  micro_batch 8 on an 80 GiB card (76.72 GiB allocated, 87 MiB free). Measured
  grid in `artifacts/results/rlt_grid.tsv`.
- `logger_backends: tensorboard`, not wandb.

13.4 h, 2000 steps, ~24 s/step, final loss 0.365. Checkpoints are **62 GiB** —
janitor with keep=2, and that still needs ~190 GiB free.

### 3.4 RLT Stage 2 — needs at least 3 GPUs, and this is not negotiable

Stage 2 runs `actor` + `rollout` + `env`, and **all three pairs exchange
tensors**: env↔rollout every step (observations and actions), env→actor every
rollout epoch (trajectories), actor↔rollout on weight sync.

RLinf selects CUDA IPC whenever two communicating workers each have exactly one
visible accelerator and it is the same one
(`_check_same_device_with_peer`, `collective_group.py`). **There is no
environment variable or config flag to force NCCL instead** — grep for one
returns nothing.

Upstream's `component_placement: actor,env,rollout: all` is a *collocated*
placement, so it requires CUDA IPC by design.

On the rented container used here CUDA IPC is impossible, verified directly
rather than inferred:

```
pidfd_getfd(pidfd, 0, 0)  ->  EPERM
/proc/sys                 ->  mounted read-only; ptrace_scope cannot be changed even as root
CapEff 0xa80405fb         ->  bit 19 (CAP_SYS_PTRACE) clear
```

With three GPU-resident components and two GPUs, one pair must share a card and
that pair deadlocks — 15 minutes of silence, no error, every thread parked in
`futex_wait` / `ep_poll` / `unix_stream_data_wait`.

**So: rent 3 or 4 GPUs and give each component its own**, e.g.
`actor: 0-0 / rollout: 1-1 / env: 2-3`.

Two further things to settle before spending on that:

- `max_steps: -1` with `max_epochs: 5000` — Stage 2 is online RL with **no
  defined end**. Decide a budget first.
- `expert_takeover.enable: True` points at
  `rlt_maniskill_joint_pi05_sft_more_steps`, a second longer-trained SFT
  checkpoint. **It is not published** — all 100+ RLinf models on the Hub were
  listed and none is an RLT model; only the dataset is published. Set
  `expert_takeover.enable: False` (eval already does) or train one.

---

## 4. The five things most likely to waste a day again

1. **`pgrep -f` / `pkill -f` match the command line doing the matching.** Six
   occurrences in this project, the last one with the pgrep pattern correctly
   bracketed but a downstream `grep` for an unbracketed literal. Count with
   `ps -eo args --no-headers | awk '$NF=="/path/script.sh"'`.
2. **`grep -c` counts lines, and tqdm output is not line-delimited.** It
   reported 4 training steps where `grep -o | wc -l` reported 1717.
3. **Existence is not completeness.** DCP writes the `.distcp` shards before
   `full_weights.pt`; a resumed download skips half-written files; a staleness
   check on an empty directory is vacuously true. Verify by reading the zip
   central directory.
4. **A probe that skips saving proves nothing.** Six "FITS" verdicts ran with
   `save_interval: -1`; saving is a separate code path with separate failures.
5. **Never edit a shell script that is mid-execution** — bash reads it lazily by
   byte offset. Use the script's own skip-guard instead.
6. **`$?` after a pipe is the last command's status.** This is documented in
   `RECAP_NOTES.md` section 1.14 and was hit again anyway, during the Hub
   upload: the python uploader was piped through `tr | grep | tail`, so a
   403 "private storage limit reached" was recorded as success — and the
   script then printed a derived rate of "39s 1203 MiB/s", which was the speed
   of failing, not of uploading. **A rate computed from a wall clock is not
   evidence that anything transferred.** The gap was found only by listing the
   Hub repo and comparing byte counts against local.

---

## 5. Where it stood

| policy | g=1.0 | n |
| --- | --- | --- |
| untrained `pi05_base` | 0.00 | 50 |
| frozen VLM, 1 epoch | 0.10 | 50 |
| frozen VLM, 2 epochs | 0.16 | 50 |
| BC on everything | 0.456 | 250 |
| RECAP, advantage-weighted | **0.512** | 250 |
| published | 0.665 | not stated |

```
unfreezing the VLM     +41 points
advantage weighting    +5.6 points   (Welch on run-level means, p = 0.043)
```

RLT Stage 1 reproduced (loss 2.235 -> 0.365). Stage 2 not run, for the reason in
section 3.4.

The open question is the remaining 0.512 -> 0.665. Section 3.9 of
`RECAP_NOTES.md` sets out what it can and cannot be attributed to; the two
documented under-trainings are the value model (3000 steps against upstream's
18000, which the advantage tag `ckpt18000` records) and Step 4 (3066 against
30000). They are **not identifiable** from the single (3000, 3066) observation
that exists.
