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
| `chain4.sh` | **Resumable** successor to `chain3.sh`: refuses to start if training is already running, resumes each phase from its own newest checkpoint, and skips a phase whose result TSV already has rows — so it is safe to launch blindly |
| `supervise_chain4.sh` + `recap-chain.supervisor.conf` | Makes the box self-healing. Install the conf into `/etc/supervisor/conf.d/`, `supervisorctl reread && supervisorctl update`; `autostart=true` relaunches `chain4.sh` when the container starts |
| `mk_allpos.py` | Writes the all-positive advantage parquet for the BC-on-everything control (§3.10) |
| `mk_succonly.py` | Writes the success-only advantage parquet for the **filtered-BC** arm (§3.12) |
| `find_successful_episodes.py` | How success was established: grouping the rollout split by episode, `return` at the last frame is exactly `-300.0` (2097) or `0.0` (1999), and 1999/4096 = **48.80%** — the number the tutorial reports as its baseline |
| `chain5.sh` | Phase C replacement: waits for `chain4.sh`, then the filtered-BC run + eval |
| `dl_rlt.sh`, `rlt_prep.sh` | RLT (ManiSkill) prep: download, artifact verification, norm stats, env smoke test |
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
- **Never edit a shell script that is mid-execution.** Bash reads a script
  lazily from the file and keeps a byte offset, so rewriting it corrupts what the
  running instance reads next. To change a queued phase, use the script's own
  skip-guard instead — here `constlr_results.tsv` was pre-seeded so `chain4.sh`
  skipped its constant-lr phase, and `chain5.sh` did the replacement work.
- **A supervisor and a hand-launched script will both start the same job.** The
  new wrapper `exec`-ed `chain5.sh` while a hand-launched one was already waiting;
  both would have cleared the same wait and put two trainings on the same two
  GPUs. It was caught only because two start banners appeared in one log. The
  wrapper now refuses if any chain script is already running.
- **`pgrep` self-matching has a second form.** Bracketing the pgrep pattern
  (`[c]hain5`) is not enough if a later stage of the same pipeline greps for an
  unbracketed literal — `pgrep -af "[c]hain5.sh" | grep -c "bash /workspace/chain5.sh"`
  returned 4 because the invoking command line contains that literal. Count with
  `ps -eo args --no-headers | awk '$NF=="/path/script.sh"'` instead.
- **Nothing restarts itself on a rented box.** This instance stopped and came
  back three times, idling 4.5 h, 22 min and 2 h 16 m each time because a human
  had to notice. `ping` succeeding while the SSH port refuses is the signature of
  a stopped container: the host's port mapping is still there with nothing behind
  it. The fix is a supervisor program with `autostart=true` plus a chain script
  that is safe to launch blindly — not a human checking.
- **Read loss next to the learning rate.** A cosine schedule with
  `total_training_steps == max_steps` and `min_lr: 0` puts the LR at ~3e-12 by the
  final step, so the loss flattens whether or not the data is exhausted.
