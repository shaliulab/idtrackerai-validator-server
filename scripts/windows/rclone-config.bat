@echo off
rem Runs "rclone config" with the validator's rclone (Drive exchange set up: see the README's table).
set "ENV=%~dp0env"
set "PATH=%ENV%;%ENV%\Library\mingw-w64\bin;%ENV%\Library\usr\bin;%ENV%\Library\bin;%ENV%\Scripts;%ENV%\bin;%PATH%"
rem Same HOME as start-validator.ps1 (rclone may look for its config under HOME when it is set)
if not defined HOME set "HOME=%USERPROFILE%"
set "RCLONE=%ENV%\Library\bin\rclone.exe"
if not exist "%RCLONE%" set "RCLONE=rclone"
"%RCLONE%" config
echo.
echo Remotes now configured:
"%RCLONE%" listremotes
pause
