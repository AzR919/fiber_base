# fiber_base — notes for Claude

See `README.md` for what the code does. This file covers the working setup.

## Machines
I work on both **macOS** and **Windows (via WSL)**. Figure out which one you're on (e.g. path starts with `/Users/azr/...` → Mac; `C:\Users\azaan\...` or `/mnt/c/...` → Windows) and use the matching commands below.

| | macOS | Windows |
|---|---|---|
| Local repo | `/Users/azr/Documents/SFU/Lab/code/fiber_base` | `C:\Users\azaan\Documents\SFU\Lab\code_base\fiber_base` (`/mnt/c/...` in WSL) |
| SSH lives in | native macOS ssh | WSL (`wsl ...`) |

**Sync (from 23 Nov):** the user no longer runs the `misc/sync_fiber*` watchers. They open the SSH socket and
Claude does every transfer explicitly with `misc/cluster.sh` (Windows: `wsl bash misc/cluster.sh <command>`). The
local repo is the hub: code goes local → cluster, results / logs / plots come cluster → local, and anything one
cluster needs from the other goes through the local buffer.

## Remote clusters (nibi is the main cluster from 28 Nov; fir is secondary)
- **From 28 Nov nibi is the main cluster** (its GPU queue was about a third of fir's). Submit new work on nibi. Jobs
  already queued on fir stay there (no duplicates on nibi). If nibi keeps finishing work sooner, move more over.
- Both clusters have the same layout: code `/home/azr/projects/def-maxwl/azr/code/fiber_base`, data
  `/home/azr/projects/def-maxwl/azr/data/...`, venv `/home/azr/projects/def-maxwl/azr/misc/menv` (`~/project/...`
  is the same place). The filesystems are separate: data, results and the venv exist per cluster.
- SSH uses a ControlMaster socket per cluster in `~/.ssh/sockets/` (2FA happens when the socket is opened).
- **Before any remote command, verify the socket is alive** (`wsl bash misc/cluster.sh check`). If it's down, tell
  the user and give them the command to reopen it; don't retry, never trigger a fresh login.

| | macOS | Windows |
|---|---|---|
| Check sockets | `ssh -O check nibi` / `fir` | `wsl bash misc/cluster.sh check` |
| Reopen (2FA) | `ssh -fN nibi` / `fir` | `wsl ssh -fN nibi` / `wsl ssh -fN fir` |
| Run remote cmd | `ssh nibi 'command'` | `wsl bash -c "ssh -o BatchMode=yes nibi 'command'"` (or a script file) |
| Close socket | `ssh -O exit nibi` | `wsl ssh -O exit nibi` |

- Modules if needed: `module load gcc arrow/19.0.1 cuda rust/1.95.0` (check versions per cluster).
- More handy commands (salloc/srun) are in `misc/misc.txt`.

### `misc/cluster.sh` (sync helper)
| Command | Does |
|---|---|
| `check` | which sockets are alive |
| `push-code <c>` | local code → cluster (no delete; skips results, logs, plots, wandb, ignore, .git) |
| `pull <c>` | cluster `results/ logs/ plots/` → local (no delete: local is the union of both clusters) |
| `push-results <c> <dir>...` | checkpoints a job on `<c>` needs, from the local buffer |
| `ls-data <c>`, `diff-data <a> <b>` | data listings (`ignore/manifests/`) and what `<b>` is missing |
| `copy-data <a> <b> <path>...` | data via the local buffer, one path at a time; big paths removed locally after arriving |
| `buffer` | local buffer sizes and free disk |

**Routine:** before submitting on a cluster, `push-code` to it. Each cluster keeps its own `results/`; copy a
checkpoint across (`push-results`) only when a job there needs it (current main models and what the next days
need). After jobs finish, `pull` from that cluster: the analysis and reports are put together locally from both.
Keep the local buffer small: results, logs and plots (a few GB) stay; CRAMs (~30 GB each) and ENCODE bigWigs
(~1 GB each) only pass through, and `copy-data` refuses anything that would leave less than 20 GB free.
Both clusters have the same data since 24 Nov (fir's older `regions/` kept as `DATA_FIBER/regions_fir_oct`).
On 28 Nov fir's sealed `regions/` (from `2026-11-27_32`), the HCT116 / MCF-7 replicates and the CTCF peak files
were copied to nibi; nibi's previous `regions/` is kept as `DATA_FIBER/regions_nibi_pre_sealed`. Upload from local
to a cluster runs at only ~0.3 MB/s, so `copy-data` is only practical for small things.
**Every Fiber-seq CRAM is in the original archive on nibi** `data/misc/google-filtered-crams.tar` (837 GB, members
named `PS<id>-fire-v0.1-filtered.cram[.crai]`; `misc/fiber_cram_rename.py` maps PS IDs to our names). Extract
single members with `tar -xOf <tar> <member> > <new name>` (seeks, fast) instead of copying from fir; nibi's project
space is near quota (4.3 / 4.6 TiB on 28 Nov), so extract only what a job needs. The three XI CRAMs (`HG008_3E4`,
`GM25455_LCL`, `HEK293T_200U`) were extracted this way on 28 Nov (`2026-11-28_19`).

## Slurm jobs
- Training: `slurm_batch_command.sh` → `main.py`. Eval of a saved checkpoint: `slurm_batch_command_eval.sh` → `eval.py`.
- Logs go to `logs/<job-name>.out/.err`; checkpoints to `results/<timestamp>_train_<prefix>_<data>_<model>_<train>/`; eval output to `results/<timestamp>_eval_<training run name>/` (`_eval_BASELINE_<data>` for baseline-only).
- **Job name scheme:** `YYYY-MM-DD_NN_<train|eval>_description` — dashes in the date, underscores after it, `NN` = two-digit index for that day starting at `00`, then `train` or `eval` so the kind of run shows in the prefix when scrolling `logs/`. Examples: `2026-10-14_00_train_no_fire`, `2026-10-14_01_eval_no_fire`. One-off jobs that are neither (plots, analysis) use the description directly.
- Every new run gets the next index for the day (count across both training and eval scripts — check `logs/` and the other script). Update `--job-name` in the script before submitting.
  Indices count across **both clusters**: check the local `logs/` (after `pull`) and `sacct` on every cluster that is up.
- The same scripts and venv run on both clusters. The scripts' CPU / memory headers fit nibi; on fir, override
  them on the sbatch command line so the job isn't billed for a larger slice: `--cpus-per-task=3` for a
  `2g.20gb` (training, eval: 64 GB is within fir's 72 GB), `--cpus-per-task=1 --mem=35G` for a `1g.10gb`
  (molecules). DataLoader workers follow `SLURM_CPUS_PER_TASK`. `sbatch --test-only` checks a request without
  submitting.
- GPUs are H100 MIG slices (`1g.10gb`, `2g.20gb`, `3g.40gb`); pick the smallest that fits.
- **Time limit ≤ 3 h** (jobs up to 3 h can use the fastest partition, `gpubase_bygpu_b1`), and close to the
  expected run time, not padded far beyond it. Longer training continues in a chained job with `--resume latest`
  (README, "Checkpoints and resuming").
- **CPUs / memory in proportion to the slice** (billing takes the largest of GPU, CPU, memory): on nibi a `1g.10gb`
  covers 2 CPUs / 37 GB, a `2g.20gb` 4 CPUs / 74 GB, a `3g.40gb` 6 CPUs / 111 GB; on fir (48 cores, 1152 GB per
  4-GPU node) 1 CPU / 35 GB, 3 CPUs / 72 GB, 5 CPUs / 108 GB.
- **CPUs with a GPU job:** requesting more CPUs than the slice comes with bills the job for a larger slice. Keep `--cpus-per-task` at or below (and set DataLoader workers to fit):

  | GPU slice | Fir | Nibi |
  |---|---|---|
  | `1g.10gb` | 1 | 2 |
  | `2g.20gb` | 3 | 4 |
  | `3g.40gb` | 5 | 6 |
  | Full H100 | 12 | 14 |

## Conventions
- Finished work goes in `reports/TODONE_YY_MM.md` (one file per month, e.g. `TODONE_26_11.md`), **newest on top**:
  add each new item at the top under its date (what was done, the result in a line, where the details are). Open
  work stays in `reports/TODO.md` / `TODO_extended.md`.
- Configs are numbered (`dataNN_`, `modelNN_`, `evalNN_`); add new ones with the next number instead of editing ones that have already been used in a run. Each config starts with a header comment saying what it is.
- Retired configs go in `configs/old_configs/`.
- Weekly result write-ups go in `reports/`. Their figures go in `plots/<name>/` (gitignored; name matches the
  `plots_code/` folder, e.g. `plots/26_oct_30/`), linked from the report as `../plots/...`, never in `reports/`.
- `tester.py`, `misc/` and `plots_code/` are one-off scripts, not part of the pipeline. When changing pipeline code (renaming functions, changing signatures or config formats), don't update, test or preserve compatibility with them, and don't let them shape the design. Only touch them when explicitly asked to work on one.
