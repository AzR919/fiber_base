@echo off
:: Stop all fiber sync processes and close SSH connections.
:: Run this before closing your laptop to prevent fail2ban bans.

:: Terminate all Mutagen sync sessions
echo [sync] Terminating Mutagen sessions...
wsl ~/bin/mutagen sync terminate --all 2>nul && echo [sync] Mutagen sessions terminated. || echo [sync] No Mutagen sessions running.

:: Close SSH ControlMaster sockets
wsl ssh -O exit nibi 2>nul && echo [sync] SSH socket closed: nibi. || echo [sync] No nibi socket open.
wsl ssh -O exit fir  2>nul && echo [sync] SSH socket closed: fir.  || echo [sync] No fir socket open.

echo [sync] Done.
