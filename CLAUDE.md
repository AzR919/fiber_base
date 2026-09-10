## Remote cluster (nibi)
- SSH via WSL ControlMaster. Socket at `~/.ssh/sockets/`. Check with: `wsl ls ~/.ssh/sockets/`
- Before any remote command, verify socket exists. If missing or connection fails, tell the user: "SSH socket to nibi is down — run `wsl ssh -fN nibi` to reopen it (requires 2FA)"
- Run remote commands with: `wsl bash -c "ssh nibi 'command'"`
- Code on nibi: `/project/def-maxwl/azr/code/fiber_base`
- Venv: `source /project/def-maxwl/azr/misc/menv/bin/activate`
- Running mutagen as well so the files are syncd between local and remote
- Commands used to run these are in misc/misc.txt
- Slurm job name scheme: YYYY_MM_DD_[two digit index for current day]_[description]. Example: 2026-09-08_00_test_fiber_coverage_vis_utils. Every new run should get an incremented two digit index, so make that change in slurm_batch_command.sh before running.
