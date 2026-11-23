@echo off
rem Runs "rclone config" with the validator's rclone (Drive exchange set up: see the README's table).
set "ENV=%~dp0env"
set "PATH=%ENV%;%ENV%\Library\bin;%ENV%\Scripts;%PATH%"
rclone config
echo.
echo Remotes now configured:
rclone listremotes
pause
