# RECAP operational scripts

The scripts that made the RECAP runs in `RECAP_NOTES.md` survivable. They live
here because three rented machines were lost mid-project and each time these had
to be rewritten from memory; the bugs they encode were each paid for once.

All paths are the rented-machine layout (`/workspace/RLinf`, `/workspace/datasets`,
`/workspace/models`) — adjust or symlink.

| file | what it does |
| --- | --- |
| `janitor5.sh` | Prunes checkpoints to the newest N. `janitor5.sh <exp-substring> [keep]` |
| `chain3.sh` | Chains eval repeats -> BC control -> prune -> constant-lr run, so the GPUs never wait on a human |
| `mk_allpos.py` | Writes the all-positive advantage parquet for the behaviour-cloning control (§3.10) |
| `inspect_advantages.py` | Prints row counts, columns and `advantage` true-rate of every advantage parquet |
| `loss.py` | Windowed means + cumulative differentials of `train/loss`, **alongside `train/learning_rate`** (§3.9) |

## The bugs these encode

- **Discover checkpoint directories, never glob them.** Run dirs are named after
  the *config*, not `experiment_name`; a janitor globbing the experiment name
  matched nothing for four hours in silence. `janitor5.sh` also logs the dirs it
  is watching at startup.
- **"Complete" needs all four files AND three minutes unmodified.** DCP writes
  the two `.distcp` shards before `full_weights.pt`, so file presence alone
  matches a half-written checkpoint.
- **`sort -n` on the step suffix**, or `global_step_999` outranks `global_step_3000`.
- **Liveness by heartbeat file, never `pgrep`.** A `pgrep` pattern naming the
  script also matches the ssh command line doing the check — this reported a
  dead janitor alive for three hours, and `pkill -f` once killed the invoking
  shell. Bracket patterns (`[j]anitor5`) where pgrep is unavoidable.
- **Redirect stdout to the log BEFORE `set -u`.** With the redirect below the
  venv activation, an unset `$PYTHONPATH` killed `chain3.sh`'s predecessor
  instantly and left no log — indistinguishable from never having started.
- **Verify the artifact, not the exit code.** These read the zip central
  directory of `full_weights.pt`; `$?` after a pipe is the last command's status,
  which once reported success for a dead run.
- **Read loss next to the learning rate.** A cosine schedule with
  `total_training_steps == max_steps` and `min_lr: 0` puts the LR at ~3e-12 by the
  final step, so the loss flattens whether or not the data is exhausted.
