# Franka-Sim + RECAP reproduction notes

Working notes from reproducing two RLinf recipes on **2x RTX 5090 (31.36 GiB
each, 61 CPU quota, 503 GB RAM, 200 GB disk)** inside an unprivileged container:

1. **Franka-Sim CNN + asynchronous SAC** — reproduced successfully (92-98%).
2. **RECAP** (4-stage offline pipeline on LIBERO-10 Task 0) — pipeline runs
   end to end. On the 5090s the frozen-VLM simplification that 31.36 GiB forced
   capped it at 0.10-0.16 against a published 0.665; moving to 2x A100-80GB and
   removing that one simplification reached **0.56** in the same single epoch
   (§3.8). **The hardware-forced simplification was the whole gap.**

The 5090 machine and its two A100 successors were all lost in turn (§5 is the
rebuild order that survived three migrations). Everything needed to redo the
work without repeating the debugging is here.

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
  Zero log output, no error, GPU utilisation 0%; it recurred later for 27
  minutes when the config on one machine had drifted back to the upstream
  default. Fix: one rank each on separate GPUs (`rollout: 0 / env: 1`).

  **`futex_wait_queue_me` on the driver is NOT the signature** — a healthy run
  shows it continuously too, because the driver waits on worker futexes while
  the workers do the work. The discriminators that actually hold:

  | | deadlocked | healthy |
  | --- | --- | --- |
  | log file size | frozen | growing |
  | per-GPU memory | **identical** on both cards | differs (e.g. 20179 / 32111 MiB) |
  | rank labels in log | two per group | one per group |

  Identical memory on both cards is the tell that both groups landed on both
  GPUs. The env *subprocesses* in `unix_stream_data_wait` and env parents in
  `ep_poll` are meaningful, but the driver's own wchan is not.
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

The worst case was not training but **imports**. `.venv` had not been touched for
18 days; its 9.9 GB of shared objects were entirely cold, and startup sat in
`dlopen` / `_imp_create_dynamic` for over 110 s. Warming it first:

```bash
find "$VENV" -name '*.so*' -type f -print0 | xargs -0 -P 8 cat > /dev/null   # 4m12s
python -c 'import torch, mujoco, franka_sim, ray'                            # 3.4s  (was >110s)
```

A **>30x** difference, and in the background it presents only as "no output at
all" — the process has not crashed and has not hung, it is faulting in pages.
Running it in the foreground once is what shows where it is stuck.

On overlayfs the page cache is a load-bearing but invisible dependency. Warm it
before a resume, warm a venv that has been idle, and expect a throughput dip
after every checkpoint save.

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

This bit three more times after it was first written down:

- `pkill -f ckpt_janitor.sh` over SSH **killed the invoking shell**, truncating
  the command mid-way.
- `pkill -f "<host>.*full_weights"` did the same to a local shell.
- The same class, without a process: after editing `cache_size: 20000` to `500`,
  a `re.search(r'cache_size: (\d+)')` self-check reported `20000` — because the
  **explanatory comment that had just been added** contains the literal text
  `cache_size: 20000`, and that is the first match in the file. Anchor the pattern
  to the syntax, not to prose: `grep -E '^ +cache_size:'`.

The general form: a verification pattern that can match the verifier's own text —
its command line, or a comment it just wrote — proves nothing.

### 1.10 Detach background jobs with `setsid`, not `nohup`

`nohup bash script.sh &` inside an SSH command died when the session closed.
`setsid bash script.sh < /dev/null > log 2>&1 &` survives, because it leaves the
process group that receives the hangup.

### 1.11 The replay buffer allocates `cache_size x traj_len x sample` in one shot

`FlatTrajectoryCache._ensure_capacity` allocates the whole flat buffer on the
**first** `put`: `total_samples = max_size * traj_slot_len`, then
`_alloc_buffer_like` calls `torch.empty` per key. A frankasim vision sample is
256*256*3 = **196,608 B** (the same 196 KB/sample as §1.6, here in RAM rather than
on disk), so the committed `cache_size: 20000` means:

| trajectory slot | implied allocation |
| --- | --- |
| 100 steps | 393 GB |
| 256 steps | **1.007 TB** |

```
RuntimeError: [enforce fail at alloc_cpu.cpp:127] DefaultCPUAllocator:
can't allocate memory: you tried to allocate 1006632960000 bytes
```

**This is latent in the known-good config, not hypothetical.** It never fires at
`total_num_envs: 1` only because a CPU `torch.empty` is a virtual reservation and
this host runs `overcommit_memory=0` (heuristic): the 393 GB request is accepted
and pages in lazily, so resident memory stays small. Verified directly on the box
— `torch.empty(393 GiB)` succeeds, `torch.empty(1007 GiB)` fails with the
identical message. Raise `total_num_envs` to 4 and the request crosses the
heuristic's threshold and the run dies ~50 s in.

**The failure is heavily disguised.** What the log shows is the env worker
`killed by ray.kill`, plus gloo `Connection closed by peer` and `ActorDiedError`
in the actor and rollout. Those are all cascade. The real first exception is in
`AsyncEmbodiedSACFSDPPolicy.run_training`, buried after the cascade, and the ray
per-actor logs (`/tmp/ray/session_*/logs/worker-*.err`) are where it can be read
cleanly. Note the env worker's own log ends normally — it did not fail, it was
killed.

Fix: size `cache_size` from the sample size. 500 slots x 256 steps x 192 KiB =
24 GB, which the sweep in §2.3 used at every N.

### 1.12 Neither launcher activates a virtualenv, and the deps are split across two

`run_eval.sh` and `run_async.sh` both call bare `python` and activate nothing, so
an unactivated shell gets `line 78: python: command not found` and `exit=127` —
once per point in a sweep, which reads as the sweep harness failing.

Worse, on this machine the dependencies are **split across two venvs** and neither
is sufficient:

| | `.venv` | `.venv_recap` |
| --- | --- | --- |
| `franka_sim` | **yes** | no |
| `libero`, `openpi` | no | **yes** |
| `mani_skill` | no | yes |
| torch, ray, mujoco, robosuite | yes | yes |

So Franka-Sim needs `.venv` and the RECAP/LIBERO work needs `.venv_recap`. Check
with `find_spec` rather than guessing:

```bash
"$V/bin/python" -c "import importlib.util as u; print(u.find_spec('franka_sim'))"
```

Also: `run_async.sh`'s last line has **no trailing newline**, so `cat -n` and
`wc -l` both stop at line 42 and the script reads as though it writes `$CMD` to a
log and never runs it. It does run it. The evidence that settles it is the mega
log the script creates: a run that got as far as importing leaves ~18 KB there,
one that died at launch leaves only the ~250-byte echoed command.

### 1.13 A fixed config in git is not a fixed config on the machine

The eval deadlock of §1.4 recurred after it had been diagnosed and fixed,
because the fix lived in a local commit while the remote machine still carried
the upstream `env,rollout: all`. And `component_placement` **cannot** be
repaired from the command line — the key contains a comma, which Hydra's
override grammar rejects. A CLI flag cannot save you; the file has to be right.

Check the placement in the file on the machine you are launching on, every
time:

```bash
sed -n '/^cluster:/,/^runner:/p' <config>.yaml
```

Conversely, paths sanitised to `/path/to/...` for the repo must be overridden
on the CLI (`runner.ckpt_path=`, `rollout.model.model_path=`) — those keys have
no comma, so they can be.

### 1.14 Smaller papercuts

- **Hydra**: adding a key that isn't in the config needs `+` (e.g.
  `+actor.enable_offload=True`), and keys containing a comma (`env,rollout`)
  cannot be overridden from the CLI at all — edit the file.
- **`run_async.sh` does not forward extra args.** `run_embodiment.sh` has an
  `EXTRA_OVERRIDES` block honouring `STEPS=` / `SAVE_INTER=`; the async script
  does not. Call `train_async.py` directly if you need overrides.
- **A driver that redirects a child with `>` cannot append to the same file.**
  The child's file descriptor keeps its own offset, so anything the driver writes
  with `>>` in the meantime gets overwritten as the child continues. This silently
  lost the peak-GPU line from every point of the §2.3 sweep; it had to be
  re-measured into a separate file.
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
- **LoRA is not implemented for `model_type: cfg_model`**, nor for
  `recap_value_model` (whose config says so outright: `is_lora: false  # Not
  supported for this model`). So **nothing in the RECAP chain can use LoRA** —
  the full-unfreeze run below had to be full-parameter.
  The reason is structural: LoRA has to be injected where the linear layers are
  constructed, and `openpi_cfg` is a 1051-line wrapper that imports
  `PI0Pytorch` from the installed `openpi` package rather than owning the
  layers. The one implementation in this repo is hand-written in
  `openpi_rlinf/pi0_model/lora.py` (4476-line module that reimplements Gemma):
  `LoRAConfig(rank, alpha, rslora)`, `scaling_value = alpha/sqrt(rank)` when
  `rslora` else `alpha/rank`, and `lora_b` zero-initialised so the adapter is a
  no-op at step 0. Separately, `rlinf/models/__init__.py` wires peft for
  openvla / openvla-oft. (An earlier version of this note said `is_lora`
  applied to the plain `openpi` type; it does not — plain `openpi` has no LoRA
  code either.)
- **RECAP cannot use `model_type: openpi_rlinf` at all.** `grep -rl
  "guidance|cfgrl|unconditional" openpi_rlinf/` returns nothing: that module
  exposes `sft_forward` / `_rlt_forward` / `default_forward` (logprobs for
  on-policy PPO/GRPO and for RLT), which is a different training objective.
  CFG routing lives only in `openpi_cfg` (`compute_cfg_routing_masks`). The
  three pi0.5 entry points are split by RL regime, not duplicated.
- **`model_type: "pi05"` in `recap_compute_advantages.yaml` is not a
  `SupportedModel`** — `grep pi05 rlinf/config.py` is empty. It is a field on
  the value model's own config, mapped to `openpi.models.model.ModelType.PI05`
  at `rlinf/data/datasets/recap/value_dataset.py:57`. Two same-named things at
  different layers.
- **`pyarrow` 26 and `numpy` 1.26 are incompatible, and a fresh install picks
  pyarrow 26.** It requires numpy >= 2, but gym pins numpy < 2, so every
  advantages/returns parquet becomes unreadable — the artifacts that cost the
  most to produce. Pin `pyarrow==17.0.0`.
- **A global `HF_HUB_ENABLE_HF_TRANSFER=1` poisons `install.sh`.** The fresh
  venv has no `hf_transfer`, so the hub *refuses every download* rather than
  falling back: libero assets failed five times and openpi / mani_skill were
  never installed, with the script still exiting 0. Unset it for the duration
  of the install, and install only `hf_transfer` — `pip install -U
  huggingface_hub[hf_transfer]` drags in a hub version that breaks
  transformers 4.53.
- **A resumed download trusts half-written files.** The second pi05 conversion
  failed with `OUT_OF_RANGE` reading an OCDBT shard because the resume skipped
  a file that was incomplete, not absent. Wipe the cache and verify with
  `restore_params` *before* converting, never after.
- **Run directories are named after the config, not `experiment_name`.** A
  janitor globbing `a100b_upstream_ep1-*` matched nothing for four hours while
  the real directory was `cfg_a100b_upstream_ep1-*`. Discover with `find ...
  -path "*$EXP*" -name checkpoints` and **log the directories being watched at
  startup** — a janitor that prints nothing looks identical to one with nothing
  to prune.
- **Toolkit scripts run by path cannot `import rlinf`.** `sys.path[0]` is the
  script's own directory, so `toolkits/lerobot/calculate_norm_stats.py` fails
  on import. Pass `PYTHONPATH=<repo>`.

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

### 2.3 Env-parallelism throughput sweep — `frankasim_sac_envbench_*env.yaml`

Tests the claim that §2.2's config carried: *"total_num_envs: 1 matches upstream.
Raising it does not help much: with EGL the sim is ~5 ms/step and the bottleneck
is the per-step round trip to the rollout worker."*

**The premise is right and the conclusion is wrong.** Raising it helps a lot,
*because* the round trip is a fixed per-cycle cost that more envs amortise.

One collection cycle is `max_steps_per_rollout_epoch` (256) steps **per env**
(`env_worker.py`: `n_train_chunk_steps = max_steps_per_rollout_epoch //
num_action_chunks`, applied to the whole vector env), so transitions per cycle are
`N x 256`. `micro_batch_size` and `global_batch_size` stay at 256, which is what
isolates the environment side: `actor/update_one_epoch` measured **0.681-0.692 s
at every N**, dead flat.

900 s wall clock per point, placement unchanged (`env: 0 / actor: 0 /
rollout: 1`). Measured training span 819-831 s at every point, i.e. the actor
trains back to back and the comparison is clean.

| N | grad steps | cycles | grad steps/s | **transitions/s** | reuse | cycle s | env step s | ingest s | peak GPU0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1095 | 73 | 1.318 | **22.5** | 15.0x | 11.30 | 3.18 (28%) | 0.071 | 2961 MiB |
| 2 | 1076 | 56 | 1.297 | **34.6** | 9.6x | 14.45 | 5.98 (41%) | 0.082 | — |
| 4 | 1034 | 40 | 1.248 | **49.5** | 6.5x | 19.69 | 10.65 (54%) | 0.115 | — |
| **8** | 1000 | 32 | **1.215** | **79.6** | 3.9x | 23.99 | 14.17 (59%) | 0.142 | 3502 MiB |
| 16 | 881 | 19 | 1.075 | **95.0** | 2.9x | 37.28 | 26.21 (70%) | 0.238 | 2883 MiB |

```
transitions/s   22.5 -> 95.0   = 4.22x
grad steps/s   1.318 -> 1.075  = 0.82x  (an 18% cost)
```

Note "grad steps" is the `Global Step` counter, which is **not** the collection
cycle count: in the async runner the actor keeps training off the buffer while the
envs collect, so at N=1 there were 1095 gradient steps against only 73 cycles.
Reading the `env/*` metric boxes as training steps understates the training rate
by 15x.

**Mechanism.** At N=1 the cycle is 11.30 s of which only 3.18 s (28%) is env
stepping; the other 8.12 s is the round trip, inference and transfer, and
`rollout/predict` is flat at ~2.3 s for every N. So the original premise was
correct. But a fixed per-cycle cost divided over N times more transitions is
exactly what amortisation means. The gain tapers (1.54x, 1.43x, 1.61x, 1.19x)
because `frankasim_env.step()` is a **serial Python loop** over independent MuJoCo
instances (`for i, env in enumerate(self.envs)`, not a batched simulator), so env
stepping grows from 28% to 70% of the cycle and takes over as the bottleneck.

The 18% cost in gradient steps/s is entirely `actor/run_training` minus
`actor/update_one_epoch` — the trajectory-ingest overhead, which grows linearly
(0.071 -> 0.238 s) as the actor digests larger batches.

**Recommendation: `total_num_envs: 8`** — 3.54x the data throughput for an 8% cost
in update rate and 540 MiB of GPU memory. Memory barely moves with N (under 11% of
a 32.6 GiB card at every point), so the change is effectively free.

**Limit of this experiment, stated plainly:** it measures *throughput*, not
time-to-95%. Evaluation and saving were disabled to isolate the timings. Sample
reuse falls from 15x to 3.9x at N=8, and which data/reuse combination converges
fastest for off-policy SAC was **not** measured — that needs a run to convergence,
hours rather than 15 minutes. So N=8 feeds 3.54x more fresh data per second for an
8% cost in update rate; that it converges faster is very likely but untested.

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

The control matters only weakly: CFG training moved the policy **from zero**, so
the pipeline runs end to end. But 10% is far from 66.5%, and see §3.10 — an
untrained base is not a baseline for RECAP, only a smoke test.

> **Superseded by §3.8.** Everything in this section was measured with
> `train_expert_only: True`. Unfreezing the VLM — the only variable changed —
> takes the same one epoch to **0.56**. Read §3.2–§3.6 as the frozen-VLM arm of
> an A/B, not as "what RECAP achieves here".

### 3.2b Transfer to the other nine LIBERO-10 tasks: zero

The policy trained on Task 0 only. The `libero10_task0_sft` split spans all ten
tasks at 0.5% of the sampling weight (§3.5) and was expected to act as an
anti-forgetting regulariser. Evaluated at guidance 1.0, 50 episodes each:

| task | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| success_once | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

450 trials, and `return` is 0.00 as well — not near-misses, no progress at all.

**Attribution limit:** there is no baseline for the untrained model on tasks 1-9,
so this does not establish forgetting. Given the untrained control scores 0.00 on
Task 0, "it never could" is at least as likely as "it learned and forgot". Either
way it points the same direction as §3.3: what was learned is a Task-0-specific
action sequence, not an ability to read an instruction and locate the object it
names.

**Task 7 sharpens this.** The ten LIBERO-10 instructions, read off the installed
benchmark (`get_benchmark("libero_10")`, 50 initial states each):

```
0: put both the alphabet soup and the tomato sauce in the basket   <- trained
1: put both the cream cheese box and the butter in the basket
2: turn on the stove and put the moka pot on it
3: put the black bowl in the bottom drawer of the cabinet and close it
4: put the white mug on the left plate and put the yellow and white mug on the right plate
5: pick up the book and place it in the back compartment of the caddy
6: put the white mug on the plate and put the chocolate pudding to the right of the plate
7: put both the alphabet soup and the cream cheese box in the basket
8: put both moka pots on the stove
9: put the yellow and white mug in the microwave and close it
```

**Task 7 shares the alphabet soup, the basket, and the "put both ... in the
basket" sentence form with Task 0** — only the second object differs. It still
scores 0.00. A policy that had learned *which object the instruction names*
should get at least the first half of Task 7 right. Scoring zero there is
consistent with the §3.3 video diagnosis (the frozen policy reaching for the
basket rather than the named object) and is why the frozen vision backbone was
promoted to the leading suspect — which §3.8 then confirmed.

Note these nine tasks are also *long-horizon* (LIBERO-10 is LIBERO-LONG: every
instruction chains two or more pick-and-place stages, and any failed stage fails
the episode), so zero transfer is less surprising than it would be on a
single-stage suite. The Task 7 point stands regardless.

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
see §1.14.

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

**Result: 10% → 16%, which does not settle the question.**

| policy | guidance | success |
| --- | --- | --- |
| untrained `pi05_base` (control) | — | 0.00 |
| epoch 1 | 1.0 | 0.10 |
| epoch 1 | 1.5 | 0.10 |
| **epoch 2** | **1.0** | **0.16** |
| epoch 2 | 1.5 | 0.14 |
| epoch 2 | 2.0 | 0.12 |
| epoch 2 | 3.0 | 0.04 |
| published | — | 0.665 |

The pre-registered thresholds were "20%+ → epochs were the limiter" and
"~12% → the frozen-VLM wall is real". 16% landed between them, and inside the
noise:

```
5/50 vs 8/50   Fisher exact, two-sided   p = 0.554
0.10   95% CI [0.02, 0.18]
0.16   95% CI [0.06, 0.26]
```

> **Correction (2026-10-11).** The original text here said *"the precision
> cannot be improved by re-running"*, on the grounds that `use_fixed_reset_state_ids:
> True` pins the 50 initial states so repeats vary only policy noise. **That
> conclusion was wrong, and it is wrong for an instructive reason — see §3.9.**
> Those 50 states are Task 0's *entire* initial-state population, not a sample
> from it, so there is no initial-state sampling error to be stuck with. The
> whole remaining uncertainty *is* the policy noise, and policy noise is exactly
> what repeats average down. The error bar on "this policy's success rate over
> Task 0's initial-state set" shrinks with every repeat.
>
> What stands is the narrower point: **designing a decision around a 6-point
> difference without first measuring the repeat spread was a planning mistake.**
> The thresholds should have come from the achievable error bar, not from round
> numbers. The fix is to measure the spread, which costs ~7 minutes per repeat.

What the run does support, weakly but consistently:

- **Doubling the epochs produced no order-of-magnitude change.** Even taking 16%
  at face value, 66.5% is 4x away.
- **The 40-sample mean loss is flat for the entire epoch**, i.e. the trainable
  parameters have essentially converged.
- **Guidance > 1 still does not help**, and the full curve is now measured at
  both epochs:

  | guidance | 1.0 | 1.5 | 2.0 | 3.0 |
  | --- | --- | --- | --- | --- |
  | epoch 1 | 0.10 | 0.10 | 0.04 | 0.00 |
  | epoch 2 | 0.16 | 0.14 | 0.12 | 0.04 |

  Epoch 2 is higher at every scale and the shape is identical — best at 1.0,
  monotonically worse above it. No single point separates from epoch 1 (Fisher
  p = 0.55 at guidance 1.0), but four independent scales moving the same way is
  not what pure noise usually looks like. CFG extrapolation degrading above 1.0
  is what a weak conditional branch looks like, at both epochs.

Two weak signals pointing the same way is an inference, not a measurement. The
frozen vision backbone remains the leading explanation and unfreezing it (2x 80GB)
is the reasonable next experiment, but this evaluation did not cleanly separate it
from "not enough epochs".

### 3.7 Unfreezing the VLM: a full memory and throughput map

§3.4 ranks `train_expert_only: True` as the dominant deviation and calls it
forced. That is right, but the reason recorded there is incomplete, and the fix
is not the one the original note implies.

**The 43.4 GiB figure was never measured.** It is arithmetic — `bf16 weights
7.24G + bf16 grads 7.24G + Adam fp32 m,v 28.96G` for the full 3.62B model under
`no_shard` — and the only full-unfreeze configuration that *was* measured was
`full_shard` at `micro_batch_size: 1`: 28.8 GiB, 0.35 samples/s. One cell of the
grid was tested and one was predicted; the rest was never tried.

Measured here, all of it, with 12 training steps per point:

| config | trainable | shard | mbs | peak GPU | s/step | saves? |
| --- | --- | --- | --- | --- | --- | --- |
| frozen (ep1/ep2) | 0.3B | no_shard | 32 | ~20.0 GiB | 22.7 | yes |
| partial, last 1 of 18 | 0.41B | no_shard | 32 | 30.4 GiB | — | untested |
| partial, last 2 | 0.52B | no_shard | 16 | 22.1 GiB | — | untested |
| partial, last 4 | 0.74B | no_shard | 8 | 18.9 GiB | — | untested |
| partial, last 8 | 1.18B | no_shard | 8 | 21.5 GiB | 26.6 | untested |
| partial, last 12 | 1.62B | no_shard | 8 | 23.9 GiB | 27.7 | untested |
| partial, last 18 (all decoder layers) | 2.28B | no_shard | 8 | **27.6 GiB** | 28.7 | **NO** |
| same, `use_orig_params: False` | 2.28B | no_shard | 8 | — | — | **OOM** |
| upstream `train_expert_only: False` | 2.7B | no_shard | 8 | — | — | **OOM at init** |
| upstream `train_expert_only: False` | 2.7B | **full_shard** | **8** | **31.3 GiB** | **208** | **yes, 31.8 GiB** |

Three things fall out.

**1. Activations dominate, so `micro_batch_size` is the knob, not the layer
count.** Unfreezing *more* layers while lowering `micro_batch_size` lowers peak
memory: last-1 at mbs=32 costs 30.4 GiB, last-18 at mbs=8 costs 27.6 GiB.
Gradient accumulation keeps `global_batch_size` at 512 throughout.

**2. Partial unfreezing trains but cannot save.** A mixed `requires_grad` —
decoder layers trainable, `vision_tower` frozen — makes FSDP's full-state-dict
hook fail:

```
AssertionError: FSDP assumes paligemma_with_expert.paligemma.model.vision_tower
.vision_model.encoder.layers.0.layer_norm1.weight is in the state_dict but the
state_dict only has odict_keys([...])
```

Every memory probe above ran with `save_interval: -1`, so "FITS" meant forward,
backward and optimizer step — **not** saving, which is a separate code path and
a separate failure. Unfreezing whole layers keeps each FSDP *flat parameter*
uniform, which is what training needs; it does nothing for state-dict
collection. `use_orig_params: False` is legal for this shape (no
*FlatParameter requires uniform requires_grad*) but OOMs during training.

**3. The upstream recipe does run on 2x 32 GiB — and costs 9x the throughput.**
`train_expert_only: False` + `full_shard` + `mbs=8` fits at 31.3 of 31.36 GiB,
saves a 31.8 GiB checkpoint, and runs at 208 s/step = 2.46 samples/s against the
frozen configuration's 22.8. mbs=8 is a 7x throughput improvement over the
`mbs=1` measurement in the original table (32 all-gather rounds per step instead
of 512), which confirms the mechanism — and still leaves one epoch at
**177 hours**.

So the corrected statement of the constraint is not "full unfreezing does not
fit on 32 GiB cards". It is:

- it fits, with `full_shard` and a small micro-batch;
- 510 MiB of headroom and a 31.8 GiB checkpoint against 44 GiB of free disk make
  it fragile — a second checkpoint cannot coexist, so no rotation is possible;
- at 2.46 samples/s one epoch is 7.4 days, which is why it was not run here.

**This was the right call: §3.8 ran it on 80 GiB cards and it is the single
change that moves 0.10 -> 0.56.** On 32 GiB cards the same result would have
taken 7.4 days instead of 25 hours.

**This re-justifies renting 80 GB cards, for a different reason than before.**
Not because unfreezing is impossible, but because 80 GiB admits `no_shard` at a
useful micro-batch: the same ~43.4 GiB of replicated optimizer state that does
not fit in 31.36 GiB fits in 80, which removes the per-layer all-gather traffic
and should return throughput to roughly the frozen figure, i.e. ~20 h/epoch
rather than 177 h.

The code change supporting the partial-unfreeze rows is
`openpi.unfreeze_last_n_vlm_layers` in `freeze_vlm()`
(`openpi_cfg_action_model.py`). It needed no config plumbing: `get_model()`
copies every key under `actor.model.openpi` onto the model config object. It is
left in because the memory numbers are useful, but **anything that uses it
cannot checkpoint** until the state-dict path is fixed.

### 3.8 The decisive A/B: unfreezing the VLM — 0.10 to 0.56

Run `cfg_a100b_upstream_ep1` on 2x A100-80GB, 3066 steps (one epoch), finished
2026-10-11 20:49 UTC. **One variable changed from the §3.2 run:**
`train_expert_only: False`. lr 1e-5, warmup 300, cosine to 0, micro-batch 32,
global 512, `no_shard`, `use_orig_params: False`, `total_training_steps: 3066`
— all held at the ep1 values on purpose.

| policy | g=1.0 | g=1.5 | g=2.0 | g=3.0 |
| --- | --- | --- | --- | --- |
| untrained `pi05_base` (control) | 0.00 | | | |
| frozen VLM, 1 epoch (§3.2) | 0.10 | 0.10 | 0.04 | 0.00 |
| frozen VLM, 2 epochs (§3.6) | 0.16 | 0.14 | 0.12 | 0.04 |
| **unfrozen VLM, 1 epoch** | **0.56** | 0.36 | | |
| published | 0.665 | | | |

5/50 -> 28/50 at the same guidance and the same number of steps. That is 84% of
the published figure, reached by changing one boolean.

> **Scope of this result.** It isolates *unfreezing*, and that much is clean. It
> does **not** show that RECAP's advantage weighting contributed anything,
> because there is no behaviour-cloning arm to compare against — see §3.10.

**The frozen PaliGemma was the binding constraint**, and the two epochs of §3.6
were effort spent on the wrong axis (0.10 -> 0.16 is Fisher p = 0.55, i.e. not
distinguishable).

The learning curve:

| checkpoint | g=1.0 |
| --- | --- |
| 2000 | 0.36 |
| 2500 | **0.58** |
| 3066 | 0.56 |

29/50 vs 28/50 between 2500 and 3066 is one episode.

> **This does NOT establish a plateau.** An earlier version of this section read
> it as one ("so a second epoch is not the next thing to try"). That reading
> ignores the schedule: `cosine` with `total_training_steps: 3066` and
> `min_lr: 0.0` puts the learning rate at **exactly zero** by step 3066 and
> already low at 2500, so "saturated" and "the learning rate decayed away"
> predict the same flat tail. §3.9 lays out the elimination order, and
> `cfg_a100b_constlr.yaml` is the run that removes this one confound.

Nor is the 0.56 -> 0.665 gap itself established: the Wilson 95% CI on 28/50 is
[0.423, 0.688], which **contains** 0.665 (two-sided binomial p = 0.13). See §3.9
before attributing it to anything.

**Guidance above 1.0 still degrades** (0.56 -> 0.36 at 1.5), the same direction
as both frozen runs. Note 1.0 is *not* "guidance off": the model computes
`(1 - s)*v_uncond + s*v_cond`, so 1.0 is the pure conditional prediction and
only `s > 1` extrapolates.

#### Training loss did not predict this — and saying so beforehand is what saved the run

| run | first 40 steps | converged |
| --- | --- | --- |
| frozen (ep1, ep2) | — | 0.016, flat the whole way |
| unfrozen | 0.0774 | **0.0129** |

A 19% better loss for a **5.6x** better success rate. An earlier partial-unfreeze
probe was killed at step 525 precisely because its loss had "flattened at the
frozen level" — the note written at the time said not to conclude from it,
because flow-matching loss measures action reconstruction and not
vision-language grounding. That caveat was correct. **On this pipeline the
training loss is close to useless as a model-selection signal; only the
rollout eval is.**

#### Operational notes

- The run survived because `save_interval: 500` plus a janitor keeping the
  newest 3 (§1.14). The previous machine died at step 530 with
  `save_interval: 1000`, losing 5 hours.
- Completion -> checkpoint verification -> janitor kill -> four evals ran
  unattended from one chained script, because two earlier GPU idles (4.5 h and
  22 min) were both "a phase finished and waited for a human".
- Final checkpoint verified by reading the zip central directory, not by file
  presence: `.metadata` 2,303,958 B, `__0_0.distcp` 13,048,039,511 B,
  `__1_0.distcp` 12,538,527,191 B, `full_weights.pt` 8,526,557,492 B, 819
  zip entries.
- Checkpointing works here only because the unfreeze is *total*. The
  `unfreeze_last_n_vlm_layers` option of §3.7 leaves mixed `requires_grad`,
  which breaks FSDP's full-state-dict hook on the frozen `vision_tower`.

---

### 3.9 What the remaining gap to 0.665 can and cannot be attributed to

§3.8 leaves 0.56 against a published 0.665, and the obvious move is to explain
it. **That move is premature, and the order in which the candidate explanations
get eliminated matters more than which one is favoured.**

#### Step 0: the gap has not been established

```
28/50 = 0.56
Wilson 95% CI                   [0.423, 0.688]   <- contains 0.665
two-sided binomial vs p=0.665   p = 0.13
```

The null hypothesis "this run and the published number do not differ" **is not
rejected**. Every sentence of the form "the gap is caused by X" is therefore
explaining a phenomenon that has not been shown to exist. This is the ordinary
failure mode of seeing two unequal numbers and reaching for a cause.

#### Step 1: shrink the error bar first — it is cheap, and it may dissolve the question

See the correction in §3.6. Task 0's 50 initial states are the whole population,
the residual uncertainty is policy sampling noise, and that averages down with
repeats. Mechanism, verified rather than assumed:

- `sample_noise` in the installed openpi is `torch.normal(mean=0., std=1., ...)`
  with **no `generator`**, so it draws from the global RNG;
- **nothing in the repo calls `seed_everything`** (`grep -rn seed_everything
  --include=*.py .` returns only the definition);
- two fresh processes reported `torch.initial_seed()` of 1.09e19 and 1.64e18,
  i.e. torch seeds its default generator nondeterministically per process.

So **re-running the identical eval command is an independent draw.** 5 repeats
is ~37 minutes of GPU time.

> **Do not vary `env.eval.seed` to produce repeats.** The env seed selects the
> initial-state pool (`reset_state_ids_all[seed_offset]`, `libero_env.py`), so
> changing it changes *which* states are evaluated and the repeats stop being
> repeats of the same quantity.

#### Step 2: hyperparameters are already eliminated — by documentation, not by GPU time

§3.4's "matched exactly" list is doing real work here. Step 4's
`micro_batch_size`, `global_batch_size`, `lr`, `sharding_strategy`,
`unconditional_prob`, `cfgrl_guidance_scale`, `guidance_type`,
`positive_only_conditional` and `adv_type` are all upstream values. The only
unmatched optimiser keys are `lr_warmup_steps` and `total_training_steps`, and
those **must** scale with `max_steps` or the whole epoch sits in warmup.

**This is the payoff for recording deviations item by item: an entire class of
explanation gets ruled out without running anything.**

#### Step 3: step count and value-model quality are NOT IDENTIFIABLE here

One observation exists:

```
(Step 2 = 3000 steps, Step 4 = 3066 steps) -> 0.56
```

One point cannot apportion an effect between two variables. Separating them
needs a 2x2:

| | Step 4 = 3066 | Step 4 = 30000 |
| --- | --- | --- |
| **Step 2 = 3000** | 0.56 (have) | ? |
| **Step 2 = 18000** | ? | ? (= upstream) |

~2 weeks of machine time. Until then, any ranking of the two is a prior, not a
measurement. For the record, the two competing priors are:

- **by magnitude of undertraining, Step 4 is worse**: 1.00 epoch vs upstream's
  9.78 (9.8x), against the value model's 0.49 vs 2.94 (6.0x) — and if upstream's
  value model really ran the 8000 steps its config ships rather than the 18000
  its advantage tag names, only 2.7x;
- **by irrecoverability, Step 2 is worse**: it determines the sign of the
  advantage, hence which frames are routed to the conditional branch, hence the
  training objective itself. Step 4 undertraining means "not far enough along a
  correct direction" and is fixed by more steps; Step 2 undertraining means the
  direction is wrong and more Step 4 steps cannot fix it.

Evidence quality also differs, and this is easy to get backwards. The value
model's "still improving when stopped" is clean — it ran at
`lr_scheduler: "constant"`, so a rising `cat_acc_best` (0.101 -> 0.191) is not
confounded. Step 4's apparent plateau is **not** clean, which is §3.9's last
point. And there is a measured reason to think Step 2 is at least functional
rather than broken: `value_current` separates expert demos (-0.232) from mixed
rollouts (-0.517).

#### Step 4: the observed plateau is confounded with the learning-rate schedule

| checkpoint | g=1.0 |
| --- | --- |
| 2000 | 0.36 |
| 2500 | 0.58 |
| 3066 | 0.56 |

Read as "one epoch saturates this data", this would retire the step-count
hypothesis. But the run used `lr_scheduler: "cosine"` with
`total_training_steps: 3066` and `min_lr: 0.0`, so **the learning rate was
exactly zero at step 3066** and already small at 2500. "Saturated" and "the
learning rate decayed away" predict the same curve.

`cfg_a100b_constlr.yaml` removes the confound and nothing else:
`lr_scheduler: "constant"` is warmup-then-flat and ignores
`total_training_steps` (`get_lr_scheduler`, `rlinf/hybrid_engines/fsdp/utils.py`).
~25 h.

- still rising at 3066 -> the plateau was the schedule, and step count is live;
- plateaus again -> one epoch really is saturated, and attention moves to Step 2.

#### The methodological point worth keeping

The strong result in these notes and the weak one differ in **what they are
compared against**, not in how carefully they were measured:

| claim | comparison | strength |
| --- | --- | --- |
| unfreezing the VLM: 0.10 -> 0.56 | **internal** — same machine, data, eval harness, one boolean apart | **strong** |
| the 0.56 -> 0.665 gap | **external** — a published number whose episode count is not stated (0.488 and 0.665 are not multiples of 1/50, so n != 50), whose seeds and checkpoint are unavailable | **weak** |

A published figure whose measurement protocol cannot be reconstructed is a weak
comparator no matter how precisely one's own side is measured. **Design the
experiment so the decisive comparison is internal.**

---

### 3.10 The baseline this project was missing

Raised by the user, and correct: **an untrained `pi05_base` scoring 0.00 is not a
baseline for RECAP.** It answers "did training do anything at all", which is a
check one can only fail embarrassingly. The question RECAP exists to answer is

> does advantage-weighted CFG training on mixed success/failure rollouts beat
> plain imitation learning on *the same data*?

and that needs a behaviour-cloning arm, which this project never ran.

Without it, **0.56 is equally well explained by "train a VLA on 1.56M frames of
Task-0 data and it gets good at Task 0."** The advantage labels, the 30%
quantile and the CFG routing would all be unvalidated. §3.8's A/B is still sound
— it isolates *unfreezing* — but it says nothing about whether RECAP's own
machinery earned anything.

**Why the published baseline cannot simply be reused:** upstream's 48.8% is a
*few-shot SFT policy*, and it is the policy that collected these 4096 rollouts
(hence the dataset name `RECAP-Libero10-Task0-48succ-Data`). Only the dataset is
published, not that checkpoint, so it cannot be evaluated here. Note also that
both RECAP runs — upstream's and this one — initialise Step 4 from
`pi05_base_pytorch` (`cfg_rl_openpi.yaml`, and line 544 of the tutorial), so

- final numbers **are** comparable: 0.56 vs 0.665;
- the *improvements* are **not** the same quantity: `0.00 -> 0.56` here against a
  published `48.8% -> 66.5%` that means "the RECAP policy beats the policy that
  collected its data", not "this model improved by 17.7 points".

#### The control is exact and cheap

The advantage reaches Step 4 as a **boolean column** in the advantage parquet,
consumed as `advantages_lookup: dict[tuple[int, int], bool]`
(`rlinf/data/datasets/recap/cfg_model.py`). Rewriting that column to all-`True`
removes the negative frames, so nothing is routed to the unconditional branch
beyond the unchanged `unconditional_prob: 0.1` dropout and the conditional
branch — the one inference uses at guidance 1.0 — trains on every frame
regardless of advantage. **That is behaviour cloning on the same data, with
every other knob identical.**

`/workspace/mk_allpos.py` writes it; verified `true_rate 0.3000 -> 1.0000` over
1,561,874 rows with columns and dtype unchanged. `relabel_advantages.py` cannot
do this — it enforces `0 < positive_quantile < 1`.

One detail worth noting: the `libero10_task0_sft` split was **already**
all-positive (`true_rate` 1.0000 before and after), so only the rollout split
changes — which is the right thing, since that split carries 99.5% of the
sampling weight (§3.5).

Config: `cfg_a100b_bc_allpos.yaml`, ~25 h.

- **BC ~= 0.56** -> the advantage weighting earned nothing in this setup. A
  strong negative result, and more informative than 0.56 on its own.
- **BC much lower** -> the CFG mechanism earned the difference, and only then is
  "reproduced RECAP" a claim about RECAP rather than about training on Task-0
  data.

**The general lesson:** a control that cannot fail is not a control. Ours was
chosen because the right comparator's checkpoint was unavailable, and the
unavailability should have been recorded as a limitation rather than quietly
substituted for.

---

---

## 4. What's in this branch

| Path | Purpose |
| --- | --- |
| `examples/offline_rl/config/cfg_a100b_upstream_ep1.yaml` | **The decisive run** (§3.8): upstream recipe with the VLM unfrozen, 0.56 on Task 0 |
| `examples/offline_rl/config/cfg_a100b_constlr.yaml` | Same run with `lr_scheduler: constant` (§3.9): removes the lr-decay confound from the apparent plateau |
| `examples/offline_rl/config/cfg_a100b_bc_allpos.yaml` | **The missing baseline** (§3.10): same run with every frame labelled positive, i.e. behaviour cloning on the same data |
| `examples/embodiment/config/frankasim_sac_cnn_async_fix.yaml` | SAC training (annotated with all measurements) |
| `examples/embodiment/config/frankasim_sac_cnn_eval.yaml` | SAC evaluation |
| `examples/offline_rl/config/recap_compute_advantages_task0.yaml` | RECAP Step 3 |
| `examples/offline_rl/config/cfg_rl_openpi_2gpu.yaml` | RECAP Step 4, epoch 1 |
| `examples/offline_rl/config/cfg_rl_openpi_2gpu_ep2.yaml` | RECAP Step 4, epoch 2 resume (§3.6) |
| `examples/offline_rl/config/cfg_unfreeze_n*_probe.yaml` | partial-unfreeze memory probes (§3.7) |
| `examples/offline_rl/config/cfg_A_n18_origFalse.yaml` | partial unfreeze with `use_orig_params: False` — OOMs (§3.7) |
| `examples/offline_rl/config/cfg_upstream_unfreeze_probe.yaml` | upstream recipe + `no_shard` — OOMs at init (§3.7) |
| `examples/offline_rl/config/cfg_B_upstream_fullshard.yaml` | upstream recipe + `full_shard` — **runs and saves**, 208 s/step (§3.7) |
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
