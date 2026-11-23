@echo off
rem Starts the idtrackerai validator (see start-validator.ps1). Close this window to stop it.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-validator.ps1"
pause
