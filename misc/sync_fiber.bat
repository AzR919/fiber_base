@echo off
setlocal enabledelayedexpansion

:: Parse args: first non-flag arg is TARGET, --no-pull skips the initial rsync
set TARGET=
set NO_PULL=0
for %%A in (%*) do (
    if "%%A"=="--no-pull" (
        set NO_PULL=1
    ) else (
        if "!TARGET!"=="" set TARGET=%%A
    )
)
if "!TARGET!"=="" set TARGET=nibi

set SESSION=fiber-!TARGET!
set REMOTE=!TARGET!:/project/def-maxwl/azr/code/fiber_base
set LOCAL=/mnt/c/Users/azaan/Documents/SFU/Lab/code_base/fiber_base

:: Terminate any existing Mutagen sessions before opening SSH
echo [sync] Terminating existing Mutagen sessions...
wsl ~/bin/mutagen sync terminate --all

:: Open SSH socket (2FA happens here if not already alive)
wsl ssh -fN !TARGET!
if !ERRORLEVEL! neq 0 (
    echo [sync] ERROR: SSH connection to '!TARGET!' failed.
    exit /b 1
)

:: Remote is ground truth: pull remote -> local before starting Mutagen (skip with --no-pull)
if !NO_PULL!==0 (
    echo [sync] Pulling from remote (remote is ground truth)...
    wsl rsync -avz --delete ^
      --exclude="wandb/" --exclude="results/" --exclude="ignore/" ^
      --exclude="__pycache__/" --exclude="*.pyc" --exclude="*.pyo" --exclude=".git/" ^
      !REMOTE!/ !LOCAL!/
) else (
    echo [sync] Skipping initial pull ^(--no-pull^).
)

:: Start Mutagen sync
wsl ~/bin/mutagen sync create --name !SESSION! ^
  --sync-mode two-way-safe ^
  --ignore "wandb/" ^
  --ignore "results/" ^
  --ignore "ignore/" ^
  --ignore "__pycache__/" ^
  --ignore "*.pyc" ^
  --ignore "*.pyo" ^
  --ignore ".git/" ^
  "!LOCAL!" ^
  "!REMOTE!"

echo.
echo [sync] Mutagen session "!SESSION!" started. Run "wsl ~/bin/mutagen sync list" to check status.
