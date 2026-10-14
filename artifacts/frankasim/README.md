# Franka-Sim artifacts (PPO and SAC)

The machine these were produced on — 2x RTX 5090, 31.36 GiB each — was lost.
These are the only surviving outputs. The findings are in `RECAP_NOTES.md`
sections 2.1 to 2.3; the configurations, each annotated with its own
measurements, are in `examples/embodiment/config/frankasim_*.yaml`.

## `sac_cnn_global_step_8189_full_weights.pt`

61,835,335 bytes, 212 zip entries, verified readable.

The reproduced vision-based SAC policy: CNN + asynchronous SAC on
`frankasim_pickcube_vision`, trained with
`examples/embodiment/config/frankasim_sac_cnn_async_fix.yaml`.

```
during training        success_once 95.3%, final return ~68
independent eval       92.2% and 98.4% over two runs of 64 episodes
                       (frankasim_sac_cnn_eval.yaml, 69 s each)
```

The two eval numbers differ by 6 points on env-seed variation alone, which is
why section 2.2 says to report a range or set `use_fixed_reset_state_ids: True`.

This run only worked after two fixes that have nothing to do with the
algorithm:

- `MUJOCO_GL=egl`. Unset, rendering falls back to CPU software and
  `env/env_interact_step` takes 114.9 s instead of 3.4 s — **0% success after
  30 hours**, and an ETA of ~500 h rather than 3.2 h.
- env and rollout on separate GPUs. Collocated, they deadlock silently on a
  container without `CAP_SYS_PTRACE`.

Loading it needs the `cnn_policy` model type and the vision env; see
`frankasim_sac_cnn_eval.yaml`.

## `frankasim_raw_logs.tar.gz`

4,808,117 bytes compressed, 19 files, ~130 MB raw. Everything measured on that
machine:

| | |
| --- | --- |
| `sac_async_fix.log` | 89 MB — the full SAC training run behind section 2.2 |
| `envbench_{1,2,4,8,16}env.log` | the env-parallelism sweep behind section 2.3, ~6 MB each |
| `envbench_sweep.log`, `mem_1env.txt`, `mem_8env.txt` | the collated throughput and peak-memory numbers |
| `ep2.log`, `eval_ep2_g1.0.log`, `eval_ep2_g1.5.log` | RECAP epoch 2 (the frozen-VLM 0.16 arm) |
| `RLinf/examples/.../frankasim_*.yaml` | the configs as they ran, now also tracked in the repo |

`mem_1env.txt` and `mem_8env.txt` are separate files on purpose: a driver that
redirects a child with `>` cannot append to the same file with `>>`, because the
child's descriptor keeps its own offset. That silently lost the peak-GPU line
from every point of the first sweep, and it had to be re-measured.

## Not preserved

The PPO MLP policy weights. Section 2.1 was a control run on state
observations, its numbers are recorded, and it retrains in minutes.
