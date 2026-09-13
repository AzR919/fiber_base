@echo off
setlocal

:: Target cluster: pass as first arg (default: nibi)
if "%~1"=="" (set TARGET=nibi) else (set TARGET=%~1)

set SESSION=fiber_%TARGET%
set REMOTE=%TARGET%:/project/def-maxwl/azr/code/fiber_base
set LOCAL=/mnt/c/Users/azaan/Documents/SFU/Lab/code_base/fiber_base

:: Open SSH socket (2FA happens here if not already alive)
wsl ssh -fN %TARGET% 2>nul

:: Check if a session for this target already exists
wsl bash -c "~/bin/mutagen sync list 2>/dev/null | grep -q '%SESSION%'" && (
    echo [sync] Session "%SESSION%" already running. Use "wsl ~/bin/mutagen sync list" to check status.
    exit /b 0
)

:: Start Mutagen sync
wsl ~/bin/mutagen sync create --name %SESSION% ^
  --sync-mode two-way-safe ^
  --ignore "wandb/" ^
  --ignore "results/" ^
  --ignore "ignore/" ^
  --ignore "__pycache__/" ^
  --ignore "*.pyc" ^
  --ignore "*.pyo" ^
  --ignore ".git/" ^
  "%LOCAL%" ^
  "%REMOTE%"

echo.
echo [sync] Mutagen session "%SESSION%" started. Run "wsl ~/bin/mutagen sync list" to check status.
