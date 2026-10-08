# Franka-Sim + RECAP reproduction notes

Working notes from reproducing two RLinf recipes on **2x RTX 5090 (31.36 GiB
each, 61 CPU quota, 503 GB RAM, 200 GB disk)** inside an unprivileged container:

1. **Franka-Sim CNN + asynchronous SAC** — reproduced successfully (92-98%).
2. **RECAP** (4-stage offline pipeline on LIBERO-10 Task 0) — pipeline runs
   end to end; result fell short of the published 66.5% for reasons traced to
   a hardware-forced simplification.

The machine these ran on was lost. Everything needed to redo the work without
repeating the debugging is here.

---

## 1. Traps that cost real time

These are ordered by how much time each one burned. The first three are **not
documented upstream** and will hit anyone on any machine; the rest are specific
to a GPU-poor / unprivileged-container setup.

### 1.1 `MUJOCO_GL` unset → CPU software rendering (~93x slower)

Launching `python examples/embodiment/train_embodied_agent.py` (or
`train_async.py`) **directly** bypasses `run_embodiment.sh` / `run_async.sh`,
which are what `export MUJOCO_GL=egl`. MuJoCo then falls back to osmesa.

Measured on `PandaPickCubeVision-v0`:

| backend | per env step |
| --- | --- |
| `osmesa` (software) | **457 ms** |
| `egl` (GPU) | **4.9 ms** |

Symptom: `env/env_interact_step` ≈ 449 ms per step, matching osmesa exactly.

**`evaluations/run_eval.sh` defaults to osmesa**
(`export MUJOCO_GL="${MUJOCO_GL:-osmesa}"`), unlike the training scripts. Always
override it explicitly when evaluating a rendering env.

### 1.2 The openpi checkpoint converter must be run with `python -m`

```
ModuleNotFoundError: No module named 'openpi.models'
```

`rlinf/utils/ckpt_convertor/` contains a local `openpi/` subpackage. Running the
converter **by path** puts that directory first on `sys.path`, shadowing the
installed `openpi`. Correct invocation:

```bash
cd $REPO_PATH
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python \
    --checkpoint_dir <cache>/openpi-assets/checkpoints/pi05_base \
    --config_name pi05_libero \
    --output_path /path/to/models/pi05_base_pytorch \
    --precision bfloat16
```

Two more details:

- `--checkpoint_dir` takes the **parent** of `params/` (the script appends
  `/params/` itself at the `restore_params` call). The script's assets-copy
  step contradicts this — it does `Path(checkpoint_dir).parent / "assets"` — so
  assets are silently not copied. See 1.3.
- Pin JAX to CPU. JAX preallocates ~75% of GPU memory by default and will
  evict anything else training on the box. Conversion is weight reshuffling;
  CPU takes ~2.5 min.

### 1.3 LIBERO norm stats must be downloaded separately

CFG training with `config_name: pi05_libero` fails at worker init with:

```
FileNotFoundError: Norm stats file not found at:
  <model_path>/physical-intelligence/libero/norm_stats.json
```

`pi05_base`'s own assets only cover arx / droid / franka / trossen / ur5e —
**no libero** — and the converter doesn't copy assets anyway. Fetch it (1.9 KB)
from the fine-tuned checkpoint's assets:

```bash
mkdir -p /path/to/models/pi05_base_pytorch/physical-intelligence/libero
curl -o /path/to/models/pi05_base_pytorch/physical-intelligence/libero/norm_stats.json \
  https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json
```

Verify: it should contain `state` (8-dim) and `actions` (7-dim), each with
`mean/std/q01/q99`.

### 1.4 CUDA IPC is blocked → never collocate two workers on one GPU

In an unprivileged container (`ptrace_scope=1`, no `CAP_SYS_PTRACE`):

```
RuntimeError: pidfd_getfd: Operation not permitted
  rlinf/scheduler/collective/collective_group.py  _recv_tensor_list_via_ipc
```

Same-device workers exchange GPU tensors via CUDA IPC and there is **no config
switch** for it. This bit three separate times:

- **SAC async training**: `env,actor,rollout: 0` → crash at weight sync.
  Fix: `env: 0 / actor: 0 / rollout: 1`.
- **LIBERO eval**: `env,rollout: all` → **silent deadlock**, not a crash.
  Driver in `futex_wait_queue_me`, env parents idle in `ep_poll`, all 60 env
  subprocesses in `unix_stream_data_wait`. 15 minutes, zero log output, no
  error. Fix: one rank each on separate GPUs (`rollout: 0 / env: 1`).
- Any attempt to attach a debugger (gdb, py-spy) to a running worker also
  fails, for the same ptrace reason — so Python stacks are unavailable and
  hangs must be diagnosed from `/proc/<pid>/wchan` and CPU-time deltas.

**Rule of thumb for this class of host: one worker per GPU, single rank.**

### 1.5 `ulimit -n` is 1024 by default

Any job with many dataloader workers dies with:

```
RuntimeError: unable to open shared memory object ... Too many open files
```

Hard limit is 1048576, so `ulimit -n 65535` in the launcher fixes it. Needed for
RECAP Step 3 (2 ranks x 28 workers) and for LIBERO eval.

### 1.6 SAC checkpoints bundle the whole replay buffer

`fsdp_sac_policy_worker.save_checkpoint()` unconditionally writes the replay
buffer alongside the weights, with no opt-out. At ~196 KB/sample that is 24 GB
at 127k samples while the policy itself is 213 MB. Consequence: **every value
of `save_interval` is wrong** —

- `-1` → nothing is ever saved, not even at `max_steps` (`check_progress()`
  gates the train-end save behind `save_interval > 0`);
- any positive value → the disk fills within minutes (this killed one run with
  `/tmp/ray ... is over 95% full` → `RuntimeError: unexpected pos`).

Fixed by the patch in this branch (see §4): a new
`algorithm.replay_buffer.save_with_checkpoint` flag, default `True` so other
configs are unaffected. **Both the save and load paths must be guarded** —
patching only the save side produces checkpoints that crash `resume_dir` on a
missing buffer.

### 1.7 A large file write evicts the dataset from the page cache

Writing a 19 GiB checkpoint pushed the training data out of the page cache, and
the next few hours of steps then went to disk for real: `time/step` rose
22.6 s → 28.5 s (+26%) with no config change. Freeing the space and letting the
cache re-warm restored it to 22.7 s. The same failure mode, more acutely, on a
resume: `rchar` crawling at 1.2 MB/s with CPU at 0% while a raw `dd` of the same
file did 1.1 GB/s — a cold cache over overlayfs. Fixed there by `cat`-ing the
11.8 GiB checkpoint to `/dev/null` first (57 s).

On overlayfs the page cache is a load-bearing but invisible dependency. Warm it
before a resume, and expect a throughput dip after every checkpoint save.

### 1.8 `utilization.gpu` does not measure speed

`utilization.gpu` reports only that *a* kernel was resident — not how fast it
ran. Both GPUs read 100% throughout a 20% slowdown. The metrics that show
whether the GPU is actually delivering are `clocks.sm` (against
`clocks.max.sm`), `power.draw` (against `power.limit`) and
`clocks_event_reasons.*`:

```
nvidia-smi --query-gpu=index,power.draw,power.limit,clocks.sm,clocks.max.sm,\
temperature.gpu,clocks_event_reasons.sw_power_cap,\
clocks_event_reasons.hw_thermal_slowdown --format=csv
```

On 2x RTX 5090 this run sat continuously at `sw_power_cap: Active`, 570-580 W
against a 575 W cap, SM clock 2650-2810 MHz against a 3105 MHz maximum — i.e.
**~12% below max boost for the entire run**, thermal slowdown never triggered.
Note this is a *constant* background condition: it does not explain a step
change in throughput, and attributing one to it was wrong — see §3.6.

### 1.9 `pgrep -f` / `pkill -f` match the command line that invokes them

Checking a background job over SSH with

```bash
ssh host 'pgrep -fc my_script.sh'      # WRONG — always >= 1
```

matches the remote shell's own command line, which contains `my_script.sh`. This
silently reported a dead janitor as alive for three hours, during which a
checkpoint rotation was missed and the disk fell to 26 GiB. The same pattern with
`pkill -f` is destructive: it kills the invoking shell too.

Use a bracket so the pattern and the literal text differ, and prefer a liveness
signal the process itself emits:

```bash
pgrep -af "[m]y_script.sh"                                  # safe
age=$(( $(date -u +%s) - $(cat /path/heartbeat) ))           # better
```

Corollary: verify that a process-name probe matches the *real* process. This
run's driver is `train_cfg.py`; a probe for `train_cfg_rl` never matched it and
only ever returned the self-match. Advancing step numbers in the log were the
only evidence of liveness that actually held.

### 1.10 Detach background jobs with `setsid`, not `nohup`

`nohup bash script.sh &` inside an SSH command died when the session closed.
`setsid bash script.sh < /dev/null > log 2>&1 &` survives, because it leaves the
process group that receives the hangup.

### 1.11 Smaller papercuts

- **Hydra**: adding a key that isn't in the config needs `+` (e.g.
  `+actor.enable_offload=True`), and keys containing a comma (`env,rollout`)
  cannot be overridden from the CLI at all — edit the file.
- **`run_async.sh` does not forward extra args.** `run_embodiment.sh` has an
  `EXTRA_OVERRIDES` block honouring `STEPS=` / `SAVE_INTER=`; the async script
  does not. Call `train_async.py` directly if you need overrides.
- **Heredocs over SSH are fragile.** An apostrophe inside the heredoc body
  closes the outer single-quoted remote command; heredocs also failed silently
  once, leaving no file. Write files locally and `scp` them.
- **Checkpoint rotation must prune BEFORE the next save**, keeping 1. Keeping 2
  means the disk must transiently hold 3; with 19 GiB checkpoints and 58 GiB
  free that left **208 MB** at the first rotation. The working version prunes on
  a 2-minute timer rather than reacting to a save, so peak use is one old
  checkpoint plus one in flight (38 GiB). It treats a checkpoint as complete only
  when all four files exist **and** nothing in the directory was modified for
  3 minutes — DCP writes the two `.distcp` shards before `full_weights.pt`, so
  file presence alone matches a half-written checkpoint. Sort step numbers with
  `sort -n` or `global_step_999` outranks `global_step_5000`.
- **FSDP `cpu_offload` is unusable** with RLinf's workers: they move the model
  to GPU, so FSDP raises *"An FSDP-managed module with parameter CPU offloading
  enabled has parameters on cuda:0"*.
- **LoRA is not implemented for `model_type: cfg_model`** — there is no lora
  code in `rlinf/models/embodiment/openpi_cfg/`. The `is_lora` field in
  `model/pi0_5.yaml` applies only to the plain `openpi` model type.

---

## 2. Franka-Sim results

### 2.1 MLP + PPO (state observations) — control

`PandaPickCube-v0`, no rendering, so unaffected by §1.1.

```
success_once: 0.96 -> 0.97 -> 0.98 -> 0.99 -> 1.0
73.4 s/step, 4h05m total, checkpoints at global_step_{50,100,150,200} (4.4 MB each)
```

Its value is as a **control**: 100% success proves the env, reward and success
plumbing are sound, so a failing vision run is not an environment problem.

### 2.2 CNN + SAC (vision) — `frankasim_sac_cnn_async_fix.yaml`

Before the fixes in §1.1/§1.4: **0% success after 30 hours**. After:

| | before | after |
| --- | --- | --- |
| step time | 127.2 s | **0.95 s** |
| `env/env_interact_step` | 114.9 s | 3.4 s |
| ETA | ~500 h | 3.2 h |

Learning curve (return = `0.3*r_close + 0.7*r_lift` over 100 steps, so a
reach-only policy caps at 30):

```
0.6 -> 2.1 -> 14.3 -> 21.5     plateau at the reach-only ceiling
          (~1 hour)
32.0 -> 45.1 -> 60.3           breaks 30: the cube is actually being lifted
final return ~68, success_once 95.3%
```

Supporting signals: `actor/q_pi` -0.24 → 7.0 monotone, `actor/entropy` 2.7 →
-4.8 (policy sharpening). Note entropy crossing `target_entropy: -2` is *not* a
problem — the breakthrough happened after it.

**Independent evaluation** (`frankasim_sac_cnn_eval.yaml`, 64 episodes, 69 s):
**92.2%** and **98.4%** on two runs. The spread is env-seed variation, so report
a range, or set `use_fixed_reset_state_ids: True`.

---

## 3. RECAP results (LIBERO-10 Task 0)

### 3.1 What each stage produced

| Stage | Config | Time | Key output |
| --- | --- | --- | --- |
| 1 Compute returns | upstream defaults | minutes | `returns_fail300.parquet` x3 |
| 2 Value model SFT | `recap_value_model_sft_gpu1.yaml` | 21 h (3000 steps) | 898M-param critic |
| 3 Compute advantages | `recap_compute_advantages_task0.yaml` | 5h21m | `advantages_..._q30.parquet` |
| 4 CFG training | `cfg_rl_openpi_2gpu.yaml` | 19 h (3066 steps) | 3.62B-param policy |

Validation checkpoints that passed at each stage:

- **Step 1**: sft split `R_raw` mean −9.79, min −10 — exactly right for N=10
  lookahead with r=−1/step and a terminal 0.
- **Step 2**: eval spearman 0.638 → 0.704 → 0.708 → 0.721 → 0.718 → 0.721;
  `cat_acc_best` 0.101 → 0.191 (still climbing when stopped at 3000).
- **Step 3**: unified threshold 0.0110, global positive rate **exactly 30.0%**.
  The real check that Step 2 worked is `value_current`: **−0.232** on expert
  demos vs **−0.517** on mixed rollouts. The near-identical *positive rates*
  across splits (30.9% / 30.0%) are **not** evidence of failure — advantage is
  a TD quantity centred on zero by construction.
- **Step 4**: CFG conditioning correct for all 3066 steps — negatives 100%
  unconditional / 0% conditional (no leakage), positives ~12% dropped to
  unconditional (matching `unconditional_prob: 0.1`).

### 3.2 Final evaluation

50 episodes on Task 0, against the published 48.8% baseline → 66.5% RECAP:

| policy | success |
| --- | --- |
| untrained `pi05_base` (control) | **0.00** |
| CFG-trained, guidance 1.0 | **0.10** |
| CFG-trained, guidance 1.5 | 0.10 |
| CFG-trained, guidance 2.0 | 0.04 |
| CFG-trained, guidance 3.0 | 0.00 |

The control matters: CFG training moved the policy **from zero**, so the
pipeline works. But 10% is far from 66.5%.

### 3.3 Why — diagnosed from video, not guessed

Frame-by-frame comparison at step 450/522 across all 50 envs:

- **Untrained base**: arm stays near home pose, objects undisturbed. No
  meaningful interaction.
- **CFG-trained**: arm reaches into the workspace and grasps confidently — but
  in many envs it grasps **the basket itself**, lifting and tilting it.

The task is *"put both the alphabet soup and the tomato sauce **in the
basket**"*. The policy learned the **motor skill** (reach, grasp) but not
**which object the instruction names** — a vision-language grounding failure.
That is exactly what freezing PaliGemma predicts: the action expert trained, the
perception stack did not, and `pi05_base` had never seen LIBERO (control = 0%).

### 3.4 Deviations from upstream, ranked by effect

**Real simplifications (3):**

| | upstream | here | effect |
| --- | --- | --- | --- |
| Step 4 `train_expert_only` | `False` | **`True`** | **dominant** — see §3.3 |
| Step 4 training | 30000 steps (~10 ep) | 3066 (1 ep) | 1/10 |
| Step 2 training | ~18000 steps | 3000 | 1/6, still improving when stopped |

The first was forced: upstream needs 43.4 GiB/GPU of optimizer state against
31.36 GiB available, and both escapes (LoRA, cpu_offload) are unavailable —
see §1.11.

**Hardware adaptations (no effect on numbers):** Step 2 `micro_batch_size`
32→8 (same `global_batch_size` 256, just finer accumulation); Step 3
`batch_size` 1024→128 (OOM) and workers 12→28 (CPU-bound); Step 4
`save_interval` 3000→1000 (disk); eval placement (deadlock, §1.4);
`lr_warmup_steps` 5000→300 and `total_training_steps` 30000→3066 (**scaling
these with `max_steps` is required, not optional** — otherwise the whole epoch
sits in warmup).

**Matched exactly:** all of Step 1; Step 3's `positive_quantile: 0.3`,
`advantage_lookahead_step: 10`, `gamma: 1.0`; Step 4's `micro_batch_size: 32`,
`global_batch_size: 512`, `sharding_strategy: no_shard`, `lr: 1e-5`,
`unconditional_prob: 0.1`, `cfgrl_guidance_scale: 1.0`,
`positive_only_conditional: true`.

### 3.5 Dataset note

Despite the directory name, `libero10_task0_sft` is **not** Task-0 expert data:
30 episodes / 8005 frames spanning **all 10** LIBERO-10 tasks (3 demos each),
of which only 836 frames are Task 0. It acts as a sparse anti-forgetting
regulariser — `balance_dataset_weights` puts it at 0.5% of the sampling weight.
Only `libero10_task0_train` (4096 episodes / 1.56M frames) is Task 0.
`libero10_task0_eval` is used **only** in Step 2 to monitor overfitting, and is
never advantage-labeled.

---

### 3.6 Second epoch — isolating "not enough steps"

The 1-epoch run scored 10%. Three causes were ranked (§3.3); this run tests only
the second, by resuming to 6132 steps and changing nothing else
(`train_expert_only` stays `True` on purpose). Config:
`cfg_rl_openpi_2gpu_ep2.yaml`.

**Setting `total_training_steps: 6132` is the whole trick.** Epoch 1 ran a cosine
schedule declared over 3066 steps, so at step 3066 the LR had already decayed to
`min_lr: 0`. Resuming without widening the schedule would have trained for 20
hours at LR≈0. torch's `LambdaLR.state_dict()` excludes the lambda, so the
restored state contributes only `last_epoch` and the new, wider cosine takes
effect: the run picked up at 5.4e-6 (≈ 0.54 x 1e-5, as predicted) and decayed to
7.25e-13 at step 6132. Confirming the LR in the first few steps after a resume is
the single check worth doing. Note the LR trajectory is a warm restart, not one
smooth cosine — unavoidable when extending a finished cosine run.

| | value |
| --- | --- |
| steps | 3066 → **6132**, completed |
| wall time | **20 h 39 m** |
| errors / tracebacks | 0 |
| `train/loss` (40-sample mean) | 0.0161 at start → **0.0154** at end |
| final LR | 7.25e-13 |
| final checkpoint | `global_step_6132`, 19 GiB, verified |

**The loss is flat, and that is the result.** Single-sample readings swing ±0.002
(0.0136-0.0183), which is wide enough to read any trend into — an early report of
"steadily decreasing" was just that mistake on three points. Averaged over 40
samples the value does not move across the whole epoch. Epoch 1 ended at the same
level, so the flow-matching objective is already near convergence for the
parameters that are trainable. That *strengthens* the §3.3 diagnosis: the binding
constraint is the frozen vision backbone, not the step count.

Checkpoint integrity was verified at each save by file inventory plus reading the
zip central directory of `full_weights.pt`. The central directory is written last,
at the end of the file, so if it parses (819 tensor entries) the 8.5 GiB write
completed — a cheap, strong completeness check that does not read the payload.

Throughput oscillated between ~22.6 s/step and ~28 s/step over multi-hour
stretches. Ruled out: page cache (no save in the window), CPU contention
(`/proc/pressure/cpu` `some avg300=0.07%`), I/O wait (`wa 0.0`), thermal slowdown
(not active). The GPU power cap is pinned Active the whole time and therefore
cannot explain a *switch*; calling it the cause was premature. Left unresolved —
it does not affect correctness, loss, or the checkpoint.

**Cost of the disk management**: `global_step_3066` from epoch 1 was deleted to
make room, so the 1-epoch policy can no longer be re-evaluated. Its numbers are
recorded in §3.2 and in the eval config's header comment.

**Still open**: the Task-0 evaluation of `global_step_6132`. 10% → 20%+ means
epochs were the limiter; 10% → ~12% means the frozen-VLM wall is real and 2x 80GB
is justified.

## 4. What's in this branch

| Path | Purpose |
| --- | --- |
| `examples/embodiment/config/frankasim_sac_cnn_async_fix.yaml` | SAC training (annotated with all measurements) |
| `examples/embodiment/config/frankasim_sac_cnn_eval.yaml` | SAC evaluation |
| `examples/offline_rl/config/recap_compute_advantages_task0.yaml` | RECAP Step 3 |
| `examples/offline_rl/config/cfg_rl_openpi_2gpu.yaml` | RECAP Step 4, epoch 1 |
| `examples/offline_rl/config/cfg_rl_openpi_2gpu_ep2.yaml` | RECAP Step 4, epoch 2 resume (§3.6) |
| `evaluations/libero/libero_10_task0_cfg_eval.yaml` | LIBERO Task-0 evaluation |
| `rlinf/workers/actor/fsdp_sac_policy_worker.py` | replay-buffer / checkpoint decoupling (§1.6) |

Upstream templates were left untouched; every config above is a new file.
Paths inside them are placeholders (`/path/to/...`) — fill them in per machine.

---

## 5. Rebuild order on a fresh machine

1. **Check the hardware first.** If each GPU has ≥80 GiB, skip every
   simplification in §3.4 and run the upstream configs as published —
   `train_expert_only: False`, `no_shard`, 30000 steps. That is the only way to
   fairly target 66.5%.
2. **Environment**:
   `bash requirements/install.sh embodied --model openpi --env maniskill_libero`.
   Put `ulimit -n 65535` in every launcher (§1.5).
3. **Assets**: dataset `RLinf/RECAP-Libero10-Task0-48succ-Data` (82 GB) from
   HuggingFace; `gs://openpi-assets/checkpoints/pi05_base` (12.4 GB) → convert
   with `python -m` (§1.2) → download LIBERO norm stats (§1.3).
   `RLinf/RLinf-ResNet10-pretrained` for the SAC track.
4. **Run the stages in order**, validating each before proceeding, using the
   checks in §3.1. Each stage's output is cheap to verify and expensive to
   redo.
5. **Evaluate** with `MUJOCO_GL=egl` and one worker per GPU (§1.1, §1.4).

Approximate costs on 2x RTX 5090, now that the debugging is done: SAC ~3 h,
RECAP Step 2 21 h + Step 3 5 h + Step 4 19 h per epoch.
