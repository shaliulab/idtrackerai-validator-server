@echo off
rem Double click to install (or update) the idtrackerai validator for this Windows user.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
