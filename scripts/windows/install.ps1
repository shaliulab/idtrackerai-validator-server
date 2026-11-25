# Installs the idtrackerai validator on Windows, for the current user (no administrator rights).
#
# Run install.bat (double click) from the unzipped bundle: this folder holds install.ps1 and wheels\
# (the lab's packages, built on Linux with
#   docker build --target windows-bundle --output dist/windows .
# so no GitHub account, ssh key, git or Node.js is needed here).
#
# What it does, in %LOCALAPPDATA%\idtrackerai-validator (override with -InstallDir):
#   1. Miniforge (conda-forge's conda) in miniforge3\, unless already there
#   2. a conda environment in env\ with Python 3.10, rclone and ffmpeg (with libx264)
#   3. torch / torchvision CPU builds, then the validator and its dependencies from PyPI + wheels\
#   4. start-validator.ps1 + settings.json, and a "FlyHostel validator" shortcut on the desktop
# Running it again (e.g. from a newer bundle) updates the installation and keeps the settings.
# -Update: run by the app's "Check for updates" (updater.py, server stopped first): asks nothing (the
# experiments folder comes from settings.json) and leaves the desktop shortcut as it is.
# Logs: logs\install-<time>.log.

param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "idtrackerai-validator"),
    [string]$DataDir = "",
    [int]$Port = 5000,
    [string]$TorchIndex = "https://download.pytorch.org/whl/cpu",
    [string]$TorchVersion = "2.13.0",
    [string]$TorchvisionVersion = "0.28.0",
    [switch]$NoShortcut,
    [switch]$Update
)
if ($Update) { $NoShortcut = $true }

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is ~10x slower with the progress bar
$Bundle = $PSScriptRoot
# Windows has no HOME / USER: code written for Linux reads them (os.environ["HOME"]). Set them for
# the processes started from here only, when unset (never machine-wide).
if (-not $env:HOME) { $env:HOME = $env:USERPROFILE }
if (-not $env:USER) { $env:USER = $env:USERNAME }
# Python's console output in UTF-8 whatever the Windows locale (e.g. Spanish: cp850 / cp1252)
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$Wheels = Join-Path $Bundle "wheels"

New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir "logs") | Out-Null
$Log = Join-Path $InstallDir ("logs\install-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
Start-Transcript -Path $Log | Out-Null

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }

# Runs a program, its output (stdout and stderr) shown and kept in the install log; throws when it
# fails. stderr is merged as text: with "Stop", Windows PowerShell 5.1 would end at the first line a
# program writes to stderr (pip's warnings), and Start-Transcript only logs what goes through Out-Host.
function Run($exe, [string[]]$arguments) {
    Write-Host "$exe $($arguments -join ' ')"
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $exe @arguments 2>&1 | ForEach-Object { "$_" } | Out-Host
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $saved
    }
    if ($code -ne 0) { throw "failed (exit code $code): $exe $($arguments -join ' ')" }
}

# What `conda activate` puts first in PATH: conda-forge's rclone.exe / ffmpeg.exe and the DLLs of the
# environment are in Library\bin, which is not on PATH otherwise.
function EnvPath($envDir) {
    return @($envDir, (Join-Path $envDir "Library\mingw-w64\bin"), (Join-Path $envDir "Library\usr\bin"),
             (Join-Path $envDir "Library\bin"), (Join-Path $envDir "Scripts"), (Join-Path $envDir "bin")) -join ";"
}

# Full path of <name>.exe in the environment (Library\bin for conda-forge), else on PATH
function FindTool($envDir, $name) {
    foreach ($dir in (EnvPath $envDir).Split(";")) {
        $path = Join-Path $dir "$name.exe"
        if (Test-Path $path) { return $path }
    }
    $cmd = Get-Command "$name.exe" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "$name.exe not found in $envDir (Library\bin) nor on PATH: the conda environment is incomplete, run install.bat again"
}

try {
    if (-not (Test-Path (Join-Path $Wheels "idtrackerai_validator_server-*.whl"))) {
        throw "wheels\idtrackerai_validator_server-*.whl not found next to install.ps1: unzip the whole bundle first."
    }
    if ($InstallDir -match " ") {
        Write-Warning "The install folder has a space in its path ($InstallDir); conda advises against it."
    }

    # 1. Miniforge
    $Conda = Join-Path $InstallDir "miniforge3\Scripts\conda.exe"
    if (-not (Test-Path $Conda)) {
        Step "Downloading Miniforge (conda-forge's conda, ~90 MB)"
        $Installer = Join-Path $env:TEMP "Miniforge3-Windows-x86_64.exe"
        Invoke-WebRequest -UseBasicParsing -OutFile $Installer `
            -Uri "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
        Step "Installing Miniforge in $InstallDir\miniforge3 (a few minutes)"
        $p = Start-Process -FilePath $Installer -Wait -PassThru -ArgumentList @(
            "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=0", "/NoRegistry=1", "/NoShortcuts=1",
            "/S", "/D=$(Join-Path $InstallDir 'miniforge3')")
        if ($p.ExitCode -ne 0 -or -not (Test-Path $Conda)) { throw "the Miniforge installer failed (exit code $($p.ExitCode))" }
        Remove-Item $Installer -ErrorAction SilentlyContinue
    } else {
        Step "Miniforge already installed in $InstallDir\miniforge3"
    }

    # 2. The environment: Python, rclone, ffmpeg (the gpl build has libx264)
    $Env = Join-Path $InstallDir "env"
    $Python = Join-Path $Env "python.exe"
    if (-not (Test-Path $Python)) {
        Step "Creating the environment (Python 3.10, rclone, ffmpeg)"
        Run $Conda @("create", "--yes", "--prefix", $Env, "--override-channels", "-c", "conda-forge",
                     "python=3.10", "pip", "rclone", "ffmpeg=*=gpl*")
    } else {
        Step "Updating rclone and ffmpeg in the environment"
        Run $Conda @("install", "--yes", "--prefix", $Env, "--override-channels", "-c", "conda-forge",
                     "rclone", "ffmpeg=*=gpl*")
    }

    $env:PATH = (EnvPath $Env) + ";" + $env:PATH

    # 3. Python packages
    Step "Installing torch / torchvision ($TorchIndex)"
    Run $Python @("-m", "pip", "install", "--disable-pip-version-check", "--index-url", $TorchIndex,
                  "torch==$TorchVersion", "torchvision==$TorchvisionVersion")

    Step "Installing the validator and its dependencies (PyPI + the bundle's wheels)"
    # the bundle's wheel by path: by name, pip keeps an older version already installed ("Requirement
    # already satisfied") and would not install the new version's new requirements
    $Viewer = @(Get-ChildItem -Path $Wheels -Filter "idtrackerai_validator_server-*.whl")[0].FullName
    Run $Python @("-m", "pip", "install", "--disable-pip-version-check", "--find-links", $Wheels, $Viewer)
    # The lab's packages keep their version numbers between commits: install the bundle's copies even
    # when pip thinks the same version is already there.
    $LabWheels = @(Get-ChildItem -Path $Wheels -Filter "*.whl" | ForEach-Object { $_.FullName })
    Run $Python (@("-m", "pip", "install", "--disable-pip-version-check", "--force-reinstall", "--no-deps") + $LabWheels)

    # idtrackerai requires opencv-python, the rest the headless build; both unpack into the same cv2\
    # folder. Keep only the headless one.
    $cv = (& $Python -c "import importlib.metadata as m; print(m.version('opencv-python-headless'))").Trim()
    Run $Python @("-m", "pip", "uninstall", "--yes", "opencv-python", "opencv-python-headless")
    Run $Python @("-m", "pip", "install", "--disable-pip-version-check", "--no-deps", "opencv-python-headless==$cv")

    Step "Checking the installation"
    Run $Python @("-c", "import cv2, flyhostel, idtrackerai, imgstore, idtrackerai_validator_server; print('ok, OpenCV', cv2.__version__)")
    # The same check without HOME / USER (Windows has none; set above only for this window)
    $home_, $user_ = $env:HOME, $env:USER
    Remove-Item Env:HOME, Env:USER -ErrorAction SilentlyContinue
    try {
        Run $Python @("-c", "import imgstore, flyhostel.data.df, idtrackerai, idtrackerai_validator_server.box_annotations, idtrackerai_validator_server.gdrive_exchange; print('ok without HOME / USER')")
    } finally {
        $env:HOME, $env:USER = $home_, $user_
    }
    # rclone and ffmpeg by full path (conda-forge: Library\bin), as the server finds them
    $Rclone = FindTool $Env "rclone"
    $Ffmpeg = FindTool $Env "ffmpeg"
    Run $Rclone @("version")
    Run $Ffmpeg @("-hide_banner", "-version")
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $encoders = & $Ffmpeg -hide_banner -encoders 2>&1 | ForEach-Object { "$_" }
    $ErrorActionPreference = $saved
    if (-not ($encoders -match "libx264")) { Write-Warning "ffmpeg has no libx264: publishing minimal bundles will fail" }

    # 4. Settings, start script, shortcut
    # settings.json: data_dir, port, and optionally VALIDATOR_USERNAME / VALIDATOR_PASSWORD /
    # VALIDATOR_EXPERIMENT / CVAT_* (added by hand, kept when installing again)
    $Settings = Join-Path $InstallDir "settings.json"
    $current = [pscustomobject]@{}
    if (Test-Path $Settings) {
        $current = Get-Content $Settings -Raw | ConvertFrom-Json
        if (-not $DataDir) { $DataDir = $current.data_dir }
        if (-not $PSBoundParameters.ContainsKey("Port") -and $current.port) { $Port = [int]$current.port }
    }
    if (-not $DataDir -and $Update) { throw "settings.json has no data_dir: run install.bat by hand once" }
    if (-not $DataDir) {
        $default = Join-Path $env:USERPROFILE "flyhostel_videos"
        Write-Host ""
        $answer = Read-Host "Folder for the experiments (FLYHOSTEL_VIDEOS) [Enter = $default]"
        $DataDir = if ($answer.Trim()) { $answer.Trim().Trim('"') } else { $default }
    }
    New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
    $current | Add-Member -Force -NotePropertyName data_dir -NotePropertyValue $DataDir
    $current | Add-Member -Force -NotePropertyName port -NotePropertyValue $Port
    $current | ConvertTo-Json | Set-Content -Encoding UTF8 $Settings
    Copy-Item -Force (Join-Path $Bundle "start-validator.ps1") (Join-Path $InstallDir "start-validator.ps1")
    foreach ($file in "start-validator.bat", "rclone-config.bat", "VERSION.txt") {
        if (Test-Path (Join-Path $Bundle $file)) { Copy-Item -Force (Join-Path $Bundle $file) (Join-Path $InstallDir $file) }
    }

    if (-not $NoShortcut) {
        $shell = New-Object -ComObject WScript.Shell
        $link = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "FlyHostel validator.lnk"))
        $link.TargetPath = Join-Path $InstallDir "start-validator.bat"
        $link.WorkingDirectory = $InstallDir
        $link.Description = "Start the idtrackerai validator and open it in the browser"
        $link.Save()
    }

    $version = (& $Python -c "import importlib.metadata as m; print(m.version('idtrackerai_validator_server'))").Trim()
    Step "Done: idtrackerai validator $version"
    Write-Host "Experiments folder: $DataDir"
    Write-Host "Start it with the 'FlyHostel validator' shortcut on the desktop (or $InstallDir\start-validator.bat);"
    Write-Host "it opens http://localhost:$Port in the browser."
    Write-Host "Drive exchange: set up rclone with $InstallDir\rclone-config.bat"
    Write-Host "Install log: $Log"
} catch {
    Write-Host ""
    Write-Host "INSTALL FAILED: $_" -ForegroundColor Red
    Write-Host "Send this log to the person who gave you the installer: $Log" -ForegroundColor Red
    Stop-Transcript | Out-Null
    exit 1
}
Stop-Transcript | Out-Null
