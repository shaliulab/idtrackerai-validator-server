# Windows install: test checklist

The bundle (`idtrackerai-validator-windows-<version>-<commit>.zip`, made on Linux with
`scripts/make_windows_bundle.sh`) installs the validator for one Windows user without
administrator rights: Miniforge, a conda environment with Python 3.10, rclone and ffmpeg,
torch (CPU) and the validator. Nothing of it was run on Windows before this test: tick each
box, and on any failure collect the logs of section 7.

## 1. Prerequisites

- [ ] Windows 10 or 11, 64-bit; about 6 GB free on the system drive (Miniforge ~0.5 GB, the
      environment ~3 GB, pip's cache) plus room for the experiments (a minimal bundle is
      ~100 MB, a whole experiment tens of GB).
- [ ] Internet access to github.com (Miniforge's installer), conda.anaconda.org (conda-forge),
      pypi.org / files.pythonhosted.org and download.pytorch.org. A proxy that blocks one of them
      stops the install.
- [ ] No GitHub account, ssh key, git or Node.js: the lab's packages (flyhostel, idtrackerai,
      imgstore, ethoscopy) and the web interface are inside the bundle (`wheels\`). They are only
      needed to install from source (`pip install .` of a clone, as in the README's Linux
      install, where pip clones flyhostel's private submodules over ssh).
- [ ] For the Drive exchange test: a Google account the lab's exchange folder is shared with
      (Editor), and on the lab server one experiment published to that folder (Drive exchange
      tab, kind minimal).

## 2. Install

1. [ ] Copy the zip to the Windows computer. Right click it → Properties: if there is an
       "Unblock" box, tick it → OK. Right click → Extract All… → Extract.
2. [ ] In the extracted folder, double click `install.bat`. If Windows shows "Windows protected
       your PC": More info → Run anyway.
3. [ ] When asked "Folder for the experiments", press Enter (`C:\Users\<you>\flyhostel_videos`)
       or type another folder.
4. [ ] Wait (10–20 min, mostly downloads). Success looks like:
       ```
       ok, OpenCV 3.4.18
       rclone v1.7x.x
       ==> Done: idtrackerai validator 1.2.4
       Experiments folder: C:\Users\<you>\flyhostel_videos
       ```
       and no yellow "ffmpeg has no libx264" warning. A "FlyHostel validator" shortcut is on the
       desktop. On failure the window shows `INSTALL FAILED: …` and the log's path.

Installed in `%LOCALAPPDATA%\idtrackerai-validator\` (`miniforge3\`, `env\`, `logs\`,
`settings.json`, `start-validator.bat`).

## 3. Start

1. [ ] Double click "FlyHostel validator" on the desktop. A console window prints
       `idtrackerai validator serving at http://localhost:5000` within ~30 s, and the browser
       opens that address (else open it by hand).
2. [ ] The page shows the app bar with the version tag (the bundle's commit, as in `VERSION.txt`). The server listens on
       this computer only (127.0.0.1): no firewall prompt is expected; if one appears, Cancel is
       fine.
3. [ ] To stop: close the console window. Port 5000 taken? Set `"port": 5001` in
       `settings.json` and start again.

## 4. Drive exchange (set up rclone, download an experiment)

1. [ ] Double click `%LOCALAPPDATA%\idtrackerai-validator\rclone-config.bat` (paste the path in
       Explorer's address bar). Answer as in the README's "Drive exchange" table (new remote
       `gdrive`, storage `drive`, scope `1`, `root_folder_id` = the folder id of the shared
       link, browser sign in with the Google account). At the end it lists `gdrive:`.
2. [ ] In the app, Drive exchange tab: shared folder `gdrive:`, your name → Save. Expected: a
       "connected" badge.
3. [ ] "Experiments on Drive" lists the published experiment → Download. The console at the
       bottom shows the job; expected: done in under a few minutes for a minimal bundle, then the
       experiment appears in the experiment picker and the folder holds it under
       `flyhostel_videos\FlyHostel<N>\…`.
4. [ ] Open it (Open button, or the picker).

## 5. Group viewer and Annotate

1. [ ] Group viewer: the frame image appears, with the flies' identities drawn on them; the
       slider, ←/→ and the play button move through the frames; the blobs table shows each
       fly's row. (A minimal bundle only holds the scenes' frames: frames outside them are blank,
       as expected.)
2. [ ] Annotate tab: the Scenes to check list is filled; clicking a scene shows the frame with
       boxes and label buttons. Change a box's label (select it, press a digit), press Enter:
       the frame gets a ✓ in the list. Reload the page (F5): the change is still there.
3. [ ] Drive exchange tab → "Upload my annotations" → job done. On the lab server, Drive
       exchange tab → the experiment's Annotators table lists your name → Check → the plan
       lists the frame(s) you changed.
4. [ ] Optional: "Check for updates" (pull) → job done.

## 6. Optional

- [ ] Login: add `"VALIDATOR_USERNAME": "test", "VALIDATOR_PASSWORD": "test123"` to
      `settings.json` (JSON: commas between entries), restart: the browser shows the login
      page; signing in opens the app.
- [ ] Update: unzip a newer bundle and run its `install.bat`: it reuses Miniforge and the
      environment, installs the new code and keeps `settings.json`.
- [ ] Uninstall: delete `%LOCALAPPDATA%\idtrackerai-validator` and the desktop shortcut (the
      experiments folder is not touched).
- [ ] Docker Desktop instead (needs WSL2 and Docker Desktop installed; a 2.5 GB image): download
      the repository as zip from GitHub (Code → Download ZIP), unzip, in PowerShell in that
      folder: `$env:FLYHOSTEL_VIDEOS = "C:\Users\<you>\flyhostel_videos"; docker compose up -d
      --build` (~5–10 min), open http://localhost:5000, rclone via
      `docker compose exec validator rclone config`. Same checks as 4–5.

## 7. If something fails, send

- `%LOCALAPPDATA%\idtrackerai-validator\logs\install-*.log` (install) and
  `logs\server-*.log` (server; the newest one)
- what was on screen (a screenshot of the console window / the page), and the browser's
  console (F12 → Console) if the page misbehaves
- `"%LOCALAPPDATA%\idtrackerai-validator\env\python.exe" -m pip list > pip-list.txt`
- Windows version (Win+R → `winver`) and whether the user name / install path has spaces or
  non-ASCII characters
