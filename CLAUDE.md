# fiber_base — notes for Claude

See `README.md` for what the code does. This file covers the working setup.

## Machines
I work on both **macOS** and **Windows (via WSL)**. Figure out which one you're on (e.g. path starts with `/Users/azr/...` → Mac; `C:\Users\azaan\...` or `/mnt/c/...` → Windows) and use the matching commands below.

| | macOS | Windows |
|---|---|---|
| Local repo | `/Users/azr/Documents/SFU/Lab/code/fiber_base` | `C:\Users\azaan\Documents\SFU\Lab\code_base\fiber_base` (`/mnt/c/...` in WSL) |
| SSH lives in | native macOS ssh | WSL (`wsl ...`) |
| Sync tool | `misc/sync_fiber.sh` (rsync + watchdog, local → remote on save) | `misc/sync_fiber.bat` (rsync pull from remote, then Mutagen two-way-safe) |
| Stop sync | `misc/sync_fiber_stop.sh` | `misc/sync_fiber_stop.bat` |

Sync excludes `wandb/`, `results/`, `ignore/`, `__pycache__/`, `.git/` (the Mac watcher also skips `logs/` and `plots/`). So: edit files locally and they reach nibi automatically; results and checkpoints live only on nibi.
Run the stop script before closing the laptop (avoids fail2ban bans from dangling connections).

## Remote cluster (nibi; `fir` also configured)
- SSH uses a ControlMaster socket in `~/.ssh/sockets/` (2FA happens when the socket is opened).
- **Before any remote command, verify the socket is alive.** If it's missing or the connection fails, tell the user the socket is down and give them the command to reopen it — don't retry.

| | macOS | Windows |
|---|---|---|
| Check socket | `ssh -O check nibi` (or `ls ~/.ssh/sockets/`) | `wsl ls ~/.ssh/sockets/` |
| Reopen (2FA) | `ssh -fN nibi` | `wsl ssh -fN nibi` |
| Run remote cmd | `ssh nibi 'command'` | `wsl bash -c "ssh nibi 'command'"` |
| Close socket | `ssh -O exit nibi` | `wsl ssh -O exit nibi` |

- Code on nibi: `~/project/def-maxwl/azr/code/fiber_base` (same as `/home/azr/projects/def-maxwl/azr/code/fiber_base`, which the slurm scripts use)
- Venv: `source /project/def-maxwl/azr/misc/menv/bin/activate`
- Modules if needed: `module load gcc arrow/19.0.1 cuda rust/1.95.0`
- More handy commands (salloc/srun, mutagen) are in `misc/misc.txt`.

## Slurm jobs
- Training: `slurm_batch_command.sh` → `main.py`. Eval of a saved checkpoint: `slurm_batch_command_eval.sh` → `eval.py`.
- Logs go to `logs/<job-name>.out/.err`; checkpoints to `results/<timestamp>_<prefix>_<data>_<model>_<train>/`.
- **Job name scheme:** `YYYY-MM-DD_NN_description` — dashes in the date, underscores after it, `NN` = two-digit index for that day starting at `00`. Example: `2026-10-14_00_no_fire`.
- Every new run gets the next index for the day (count across both training and eval scripts — check `logs/` and the other script). Update `--job-name` in the script before submitting.
- GPUs are H100 MIG slices (`1g.10gb`, `2g.20gb`, `3g.40gb`); pick the smallest that fits.

## Conventions
- Configs are numbered (`dataNN_`, `modelNN_`, `evalNN_`); add new ones with the next number instead of editing ones that have already been used in a run. Each config starts with a header comment saying what it is.
- Retired configs go in `configs/old_configs/`.
- Weekly result write-ups go in `reports/`.
