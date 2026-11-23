# Starts the idtrackerai validator installed by install.ps1 (settings in settings.json next to this
# file) and opens it in the browser. Close this window (or press Ctrl+C) to stop it.
# Messages also go to logs\server-<time>.log.

$ErrorActionPreference = "Stop"
$InstallDir = $PSScriptRoot
$Settings = Get-Content (Join-Path $InstallDir "settings.json") -Raw | ConvertFrom-Json
$Env = Join-Path $InstallDir "env"

# rclone, ffmpeg and the environment's DLLs first in PATH, as `conda activate` would (the server also
# finds them without it, oscompat.add_env_to_path)
$env:PATH = @($Env, (Join-Path $Env "Library\mingw-w64\bin"), (Join-Path $Env "Library\usr\bin"),
              (Join-Path $Env "Library\bin"), (Join-Path $Env "Scripts"), (Join-Path $Env "bin"), $env:PATH) -join ";"
foreach ($tool in "rclone", "ffmpeg") {
    $toolPath = Join-Path $Env "Library\bin\$tool.exe"
    if ((Test-Path $toolPath) -and -not (Test-Path "env:$($tool.ToUpper())")) { Set-Item -Path "env:$($tool.ToUpper())" -Value $toolPath }
}
$env:FLYHOSTEL_VIDEOS = $Settings.data_dir
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
# Windows has no HOME / USER: code written for Linux reads them (os.environ["HOME"]). Set them for
# the processes started from here only, when unset (never machine-wide).
if (-not $env:HOME) { $env:HOME = $env:USERPROFILE }
if (-not $env:USER) { $env:USER = $env:USERNAME }
foreach ($name in "VALIDATOR_USERNAME", "VALIDATOR_PASSWORD", "VALIDATOR_EXPERIMENT", "CVAT_HOST",
                  "CVAT_USERNAME", "CVAT_PASSWORD") {
    $value = $Settings.$name
    if ($value) { Set-Item -Path "env:$name" -Value $value }
}

# The commit shown in the app bar (no .git here): "idtrackerai-validator 1.3.0 (abc1234), built ..."
$VersionFile = Join-Path $InstallDir "VERSION.txt"
if (Test-Path $VersionFile) {
    if ((Get-Content $VersionFile -Raw) -match "\(([^)]+)\)") { $env:VALIDATOR_COMMIT = $Matches[1] }
}

New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir "logs") | Out-Null
$Log = Join-Path $InstallDir ("logs\server-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
Write-Host "idtrackerai validator: experiments in $($env:FLYHOSTEL_VIDEOS), log in $Log"
Write-Host "Close this window to stop the server."

# The app wipes and recreates ./frames: run it from the install folder
Set-Location $InstallDir
# Python logs to stderr: with "Stop", Windows PowerShell 5.1 would end the script at its first line
$ErrorActionPreference = "Continue"
& (Join-Path $Env "Scripts\start-idtrackerai-validator-server.exe") --port $Settings.port 2>&1 |
    ForEach-Object { "$_" } | Tee-Object -FilePath $Log
