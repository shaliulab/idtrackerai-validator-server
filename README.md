![build](https://github.com/shaliulab/idtrackerai-validator-server/actions/workflows/ci.yml/badge.svg)

# Purpose

Validate results of experiments in the flyhostel platform.

The application serves both the API and the web interface from a single port,
so no separate frontend server is needed.

# How to install in Linux

Tested on Python 3.10. You need git, and Node.js with npm to build the web interface
(install them from your package manager or from https://nodejs.org if `npm --version`
fails).

## Download

```
mkdir -p $HOME/opt
cd $HOME/opt
git clone https://github.com/shaliulab/idtrackerai-validator-server
cd idtrackerai-validator-server
```

No `git submodule update` is needed: `pip install` fetches what it needs at the
commits this version pins.

## Make a conda/mamba environment

Recommended, so the validator's packages do not mix with others. Make it once:

```
conda create --name validator python=3.10
```

Then activate it in every new terminal before installing, updating or starting
the server (the prompt then starts with `(validator)`):

```
conda activate validator
```

## Install

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install .
```

This gives the Group viewer (with sleep bouts, interactions, incubator readings and
contours), the Annotate tab, the Drive exchange tab and the analysis tabs that need
nothing else (Ethograms, PE, Pose, QC, Metadata overview, Data management, Monitor, ...).
Optional extras:

| Install | Adds |
|---|---|
| `pip install ".[sleap]"` | the SLEAP labels tab (sleap-io) |
| `pip install ".[full]"` | every extra (today the same as `[sleap]`) |

`pip install ".[annotator]"`, a former command, still works and installs the default.

What `pip install` does on its own:

- the web interface: when `idtrackerai_validator_server/frontend/` has none, it is
  built with npm from the `idtrackerai-validator-client` submodule, or, when the
  submodule is not checked out, from a clone of its pinned commit (kept in `build/`).
  With one already there it installs that one as it is, unless pip built it from an
  older pinned commit (then it is rebuilt).
- flyhostel (with ethoscopy and idtrackerai) and imgstore, at the commits the
  submodules point at, downloaded from GitHub as archives of public repositories: no
  GitHub account or ssh key is needed. When the `flyhostel` submodule is checked out,
  they are installed from it instead, local changes included.

idtrackerai needs torch and torchvision. The first `pip install` gets the CPU builds,
which are enough to serve the app; without it pip installs the default (CUDA) builds,
several GB. Skip it if your environment already has torch, for example a CUDA build.

Not Python packages, so not installed by pip:

- `rclone` (https://rclone.org/install/) for the Drive exchange tab (and the Dropbox
  checks of the Data management tab); see "Drive exchange" below.
- `ffmpeg` (with libx264) to publish minimal bundles in the Drive exchange tab and for
  the sleep bout / PE videos.
- The lab tools run in their own conda environments (`FLYHOSTEL_VALIDATION_ENV` for
  Prepare scenes / Integrate, `YOLOV7_CVAT` for Propose, roi-proposer, idtrackerai v4,
  the metadata downloader, SLEAP): see the environment variables in `CLAUDE.md`.

Use `pip install -e .` instead of `pip install .` if you intend to edit the Python code.
To work on the interface, check out its submodule
(`git submodule update --init idtrackerai-validator-client`) and rebuild it with
`./build_frontend.sh` (which runs `git submodule update --init --recursive`
first).

## Drive exchange

People who annotate experiments shared through Google Drive (the **Drive exchange**
tab) install as above and set up rclone:

The publisher shares the Google Drive exchange folder with your Google account as
**Editor** and sends you its link, `https://drive.google.com/drive/folders/<FOLDER_ID>`.
The folder id is not a secret (the sharing gives the access), and you never need the
publisher's OAuth client id or secret: you sign in with your own Google account.

Install the current rclone (https://rclone.org/install/, e.g.
`curl https://rclone.org/install.sh | sudo bash`; not the Linux distribution's package,
usually too old to sign in to Google) and add the remote with `rclone config`:

| Prompt | Answer |
|---|---|
| `n/s/q>` | `n` (new remote) |
| `name>` | `gdrive` |
| `Storage>` | `drive` (Google Drive) |
| `client_id>`, `client_secret>` | Enter (empty: rclone's own client) |
| `scope>` | `1` (full access; `drive.file` would hide the other people's files) |
| `root_folder_id>` | the `<FOLDER_ID>` of the link |
| `service_account_file>` | Enter |
| `Edit advanced config?` | `n` |
| `Use web browser to automatically authenticate?` | `y`, and sign in with the Google account the folder is shared with |
| `Configure this as a Shared Drive?` | `n` |
| `Keep this remote?` | `y`, then `q` |

Without a browser (e.g. over ssh) answer `n` to the browser question and run the
`rclone authorize "drive"` it prints on a computer that has one (same rclone version),
then paste the result back. `rclone lsd gdrive:` should list the published experiments.
Never share the output of `rclone config` / `rclone config show`: it holds the sign-in
token, which opens the whole Drive of the account.

Start the server with an empty folder for the experiments:

```
mkdir -p $HOME/flyhostel_videos
FLYHOSTEL_VIDEOS=$HOME/flyhostel_videos start-idtrackerai-validator-server
```

Open http://localhost:5000. In the Drive exchange tab set the shared folder to `gdrive:`
(the remote name and a colon, nothing after it: `root_folder_id` already points at the
folder; with a shortcut to the folder in My Drive instead, `gdrive:FlyHostelExchange`)
and your name, Save (the badge says connected), download an experiment, annotate it in
the Annotate tab and upload your annotations. Experiments
published as a minimal bundle hold only the frames of the scenes to check (a few hundred
MB instead of the whole recording). The tab's
"How to set up" and the "How to use" tab explain the rest.

## Install a specific version

Releases are tagged (see https://github.com/shaliulab/idtrackerai-validator-server/releases).
To install one, check out its tag and install as above:

```
git checkout v1.3.0
pip install .
```

## Update

```
conda activate validator
cd $HOME/opt/idtrackerai-validator-server
git pull
pip install .
```

If you checked out the submodules, update them too (`git submodule update --init
--recursive`, then `./build_frontend.sh`): `pip install` installs from a checked-out
submodule rather than the pinned commit, and keeps an interface built by
`build_frontend.sh` as it is.

To go to a specific version instead of the latest, replace `git pull` with
`git fetch --tags` followed by `git checkout v1.3.0` (the version you want).

Then restart the server so it runs the new version (see "Stop and restart" under
"Run" below), and reload the page in the browser.

# Install with Docker

An alternative to the conda install above: one image with the server, the web
interface, rclone and ffmpeg (the default install: Group viewer, Annotate, Drive
exchange and the analysis tabs that need nothing else). You need Docker with BuildKit
(Docker 23 or newer; on Windows and macOS, Docker Desktop). No GitHub account or ssh
key is needed: the build fetches flyhostel and, of its submodules, only the public ones
it installs (ethoscopy, idtrackerai, imgstore), at the commits `setup.py` pins.

## Build

```
git clone https://github.com/shaliulab/idtrackerai-validator-server
cd idtrackerai-validator-server
docker build \
    --build-arg UID=$(id -u) --build-arg GID=$(id -g) \
    --build-arg VALIDATOR_COMMIT=$(git rev-parse --short HEAD) \
    -t idtrackerai-validator .
```

The build takes about 5 minutes and the image is 2.5 GB (torch's CPU build is
0.7 GB of it). The image builds the web interface from the client commit `setup.py`
pins; local checkouts of the submodules are not used.

`UID` / `GID` are the user the server runs as: files it writes into your data folder
(annotations, caches) belong to that user. Build arguments:

| argument | default | meaning |
|---|---|---|
| `TORCH_INDEX` | `https://download.pytorch.org/whl/cpu` | torch / torchvision index; `https://download.pytorch.org/whl/cu128` for a GPU image (several GB larger) |
| `TORCH_VERSION`, `TORCHVISION_VERSION` | 2.13.0, 0.28.0 | |
| `RCLONE_VERSION` | 1.71.2 | |
| `UID`, `GID` | 1000 | |
| `VALIDATOR_COMMIT`, `VALIDATOR_COMMIT_DATE` | empty | shown in the app bar (the image has no `.git`) |

## Run

```
docker run -d --name validator -p 5000:5000 \
    -v /path/to/flyhostel_videos:/data \
    -v $HOME/.config/rclone:/home/validator/.config/rclone \
    -v validator-config:/home/validator/.config \
    -v validator-cache:/home/validator/.cache \
    -e VALIDATOR_USERNAME=choose-a-user -e VALIDATOR_PASSWORD=choose-a-password \
    --restart unless-stopped idtrackerai-validator
```

Open http://localhost:5000. Or, with `docker-compose.yml`:

```
export FLYHOSTEL_VIDEOS=/path/to/flyhostel_videos
docker compose up -d --build
```

The image's stages also make the Windows install bundle (see "Install on Windows").

| mount | holds |
|---|---|
| `/data` | `$FLYHOSTEL_VIDEOS`: `index.txt` and the experiment folders (the Annotate tab writes `flyhostel/validation/validator_annotations.db` in them, the PE tab `pe_annotations.db` at the top). Must be writable for annotating; read-only works to browse |
| `/home/validator/.config/rclone` | your `rclone.conf` (Drive exchange, Dropbox checks): set the remote up on the host with `rclone config`, or inside with `docker exec -it validator rclone config` |
| `/home/validator/.config` | the login cookie's secret key, Drive exchange settings, validation queue |
| `/home/validator/.cache` | QC and continuity caches |

The container listens on all its interfaces, so set `VALIDATOR_USERNAME` and
`VALIDATOR_PASSWORD` (or publish the port on loopback only: `-p 127.0.0.1:5000:5000`).
Every environment variable of `CLAUDE.md` can be passed with `-e`, e.g.
`VALIDATOR_EXPERIMENT`, `CVAT_HOST` / `CVAT_USERNAME` / `CVAT_PASSWORD`. For https
mount the certificate of `scripts/make_tls_certificates.sh` and point at it:
`-v ~/.config/idtrackerai-validator/tls:/tls:ro -e VALIDATOR_CERTFILE=/tls/cert.pem
-e VALIDATOR_KEYFILE=/tls/key.pem`.

Experiment folders that are symlinks (e.g. `FlyHostel1 -> /media/…/DATA1/FlyHostel1`)
point outside `/data` inside the container: mount each target at the same path too
(`-v /media/…/DATA1:/media/…/DATA1`).

GPU: nothing in the default install needs one. To use it anyway (torch in
idtrackerai's segmentation), build with a CUDA `TORCH_INDEX`, install the NVIDIA
Container Toolkit and add `--gpus all` (compose: the commented `deploy` block).

Update: `git pull`, build again, then `docker rm -f validator` and run it again (the
volumes keep the settings).

What does not work in the image: the features that run the lab tools from their own
conda environments, which the image does not include: Prepare scenes, Integrate and
Write validated tables in the Annotate tab (`FLYHOSTEL_VALIDATION_ENV`), Propose
(YOLO + solver; `YOLOV7_CVAT`, needs a GPU), the Data collector's roi-proposer,
preprocess-sample (idtrackerai v4), Crop arenas and its fetch from cv2 (ssh), the
metadata downloads of the Overview (Google token), the SLEAP labels tab and training,
the Monitor tab (nextflow processes of the host), the PE video's browser engine
(Chrome), CVAT launch, the Ethoscope tab's download from cv1 and the Data management
tab's backup drives unless mounted. These are tabs for the lab's server; annotators
need none of them. Each could be added later as an image of its own (or a larger
variant of this one) if needed.

# Install on Windows

For annotators: a zip with an installer that needs no administrator rights, GitHub
account, git or Node.js (the lab's packages and the web interface are inside it).

Make the zip on a Linux computer with Docker (~2 minutes, nothing from GitHub but the
public repositories):

```
scripts/make_windows_bundle.sh      # -> dist/idtrackerai-validator-windows-<version>-<commit>.zip
```

On Windows: unzip it, double click `install.bat` and answer where the experiments go.
It installs, in `%LOCALAPPDATA%\idtrackerai-validator`, Miniforge, a conda environment
with Python 3.10, rclone and ffmpeg, torch (CPU) and the validator (10–20 minutes,
mostly downloads), and puts a "FlyHostel validator" shortcut on the desktop, which
starts the server and opens http://localhost:5000. `rclone-config.bat` there sets up the
Drive exchange remote (see "Drive exchange" above). Running a newer bundle's
`install.bat` updates it and keeps the settings (`settings.json`: experiments folder,
port, optionally `VALIDATOR_USERNAME` / `VALIDATOR_PASSWORD`). Logs are in its `logs\`.
`scripts/windows/TESTING.md` is the step-by-step test.

Docker Desktop (WSL2) runs the image of "Install with Docker" on Windows too.

# Setup

Point the application at your data directory:

```
echo "export FLYHOSTEL_VIDEOS='/path/to/flyhostel_data/videos'" >> ~/.bashrc
export FLYHOSTEL_VIDEOS='/path/to/flyhostel_data/videos'
```

# Structure database

The application assumes the experiment folders are saved in a database as follows

`$FLYHOSTEL_VIDEOS/FlyHostelN/GROUPSIZEX/FOLDER`

example:

`'/path/to/flyhostel_data/videos/FlyHostel1/3X/2026-08-19_14-00-00'`

Inside `2026-08-19_14-00-00` the application expects a collection of .mp4 files named

`000001.mp4`
`000002.mp4`
`000003.mp4`

and so on

and a sqlite file called

`FlyHostel1_3X_2026-08-19_14-00-00.db`

Once you have a database structured like that, run:

```
cd $FLYHOSTEL_VIDEOS
find -maxdepth 4 -mindepth 4 -regex .*FlyHostel.*db -not -name index.db > index.txt
```

# Run

There are two ways to run the server: in a terminal, or as a background service
that keeps running after you close the terminal or log out, and starts again
when the computer restarts. Use the service on a computer other people connect to.

## In a terminal

```
conda activate validator
start-idtrackerai-validator-server --port 5000
```

The application is then available at http://localhost:5000 (a browser window
opens; add `--no-browser` to skip it). The server runs as long as this terminal
stays open, and prints its messages there.

- **Stop**: press `Ctrl+C` in that terminal.
- **Restart** (e.g. after an update): press `Ctrl+C`, then run the same
  `start-idtrackerai-validator-server ...` command again.

## As a background service

Set it up once (the `FLYHOSTEL_VIDEOS` of "Setup" above must be set in this
terminal):

```
conda activate validator
cd $HOME/opt/idtrackerai-validator-server
mkdir -p ~/.config/systemd/user/idtrackerai-viewer.service.d
sed -e "s|^ExecStart=[^ ]*|ExecStart=$(which start-idtrackerai-validator-server)|" \
    -e "s|^Environment=FLYHOSTEL_VIDEOS=.*|Environment=FLYHOSTEL_VIDEOS=$FLYHOSTEL_VIDEOS|" \
    -e "s|^Environment=PE_ANNOTATIONS_DB=.*|Environment=PE_ANNOTATIONS_DB=$FLYHOSTEL_VIDEOS/pe_annotations.db|" \
    scripts/idtrackerai-viewer.service > ~/.config/systemd/user/idtrackerai-viewer.service
```

The service accepts connections from the network (`--host 0.0.0.0`), so give it
a login. Replace `choose-a-user` and `choose-a-password` before running this:

```
cat > ~/.config/systemd/user/idtrackerai-viewer.service.d/auth.conf <<'EOF'
[Service]
Environment=VALIDATOR_USERNAME=choose-a-user
Environment=VALIDATOR_PASSWORD=choose-a-password
EOF
chmod 600 ~/.config/systemd/user/idtrackerai-viewer.service.d/auth.conf
```

Then start it, and have it start again whenever the computer restarts:

```
systemctl --user daemon-reload
systemctl --user enable --now idtrackerai-viewer
loginctl enable-linger $USER
```

`loginctl enable-linger` keeps the service running after you log out. If it asks
for a password you do not have, ask an administrator to run
`sudo loginctl enable-linger <your user name>`.

The application is then available at http://localhost:5000 on this computer, and
at `http://<this computer's name or IP>:5000` from others.

## Stop and restart

| what | in a terminal | as a service |
| --- | --- | --- |
| restart (after an update) | `Ctrl+C`, then start it again | `systemctl --user restart idtrackerai-viewer` |
| stop | `Ctrl+C` | `systemctl --user stop idtrackerai-viewer` |
| start | `start-idtrackerai-validator-server --port 5000` | `systemctl --user start idtrackerai-viewer` |
| is it running? | look at the terminal | `systemctl --user status idtrackerai-viewer` (`q` to quit) |
| messages / errors | printed in the terminal | `journalctl --user -u idtrackerai-viewer -f` (`Ctrl+C` to quit) |

Restarting cuts off what the server is doing: let saves and running jobs (e.g.
an integration in the Annotate tab) finish first. After a restart, reload the
page in the browser.

Options:

| flag | default | meaning |
| --- | --- | --- |
| `--port` | 5000 (`$BACKEND_PORT`) | port to listen on |
| `--host` | 127.0.0.1 (`$BACKEND_HOST`) | interface to bind; `0.0.0.0` accepts connections from the network |
| `--no-browser` | off | do not open a browser window on startup |
| `--threads` | 8 | worker threads |
| `--certfile`, `--keyfile` | `$VALIDATOR_CERTFILE`, `$VALIDATOR_KEYFILE` | TLS certificate and key: serve over https instead of http |
| `--debug` | off | Flask development server with reloader; loopback only |

## Login and https

By default there is no login: anyone who can reach the port can use the app. On
the loopback interface (the default `--host`) that is only this machine. Before
using `--host 0.0.0.0`, turn on the login, and preferably https:

```
export VALIDATOR_USERNAME=<user>
export VALIDATOR_PASSWORD=<password>
```

When both are set, every page and API call needs a sign in. The form at `/login`
keeps you signed in with a cookie, and scripts can send the same credentials as
HTTP Basic auth.

For the background service, that is the `auth.conf` above.

To serve over https, make a certificate and start the server with it:

```
scripts/make_tls_certificates.sh
start-idtrackerai-validator-server --host 0.0.0.0 --no-browser \
    --certfile ~/.config/idtrackerai-validator/tls/cert.pem \
    --keyfile ~/.config/idtrackerai-validator/tls/key.pem
```

For the background service, add the certificate instead, then restart it:

```
cat > ~/.config/systemd/user/idtrackerai-viewer.service.d/tls.conf <<EOF
[Service]
Environment=VALIDATOR_CERTFILE=$HOME/.config/idtrackerai-validator/tls/cert.pem
Environment=VALIDATOR_KEYFILE=$HOME/.config/idtrackerai-validator/tls/key.pem
EOF
systemctl --user daemon-reload
systemctl --user restart idtrackerai-viewer
```

The address is then `https://…:5000` instead of `http://`.

The script makes a small certificate authority (`ca.pem`) and a server certificate
signed by it. Each client computer trusts `ca.pem` once, and then browsers show no
warning. The login page links it as `/ca.crt` with instructions.
`scripts/install_ca_linux_chrome.sh` does it for Chrome on Linux.

An SSH tunnel (below) is the alternative on a shared machine.

## Running on a remote machine

Start the server on the remote host (in a terminal there, or as a service),
then forward the single port from your laptop:

```
ssh -L 5000:localhost:5000 remote-host
```

Open http://localhost:5000 locally. Only one port needs forwarding — the API and
the interface share it.

# Development

Working on the client with hot reload, instead of rebuilding each time:

```
cd idtrackerai-validator-client
npm start
```

This starts the Vite dev server on port 3000 and proxies API calls to the backend
on port 5000 (`REACT_APP_BACKEND_PORT` to change it), so run the Python server alongside it. Forward both ports if you
are working remotely.

The compiled interface in `idtrackerai_validator_server/frontend/` is a build
artifact and is not tracked by git. To produce distributable artifacts:

```
./build_frontend.sh
python setup.py sdist bdist_wheel
unzip -l dist/*.whl | grep index.html    # confirm the interface is included
```

A wheel built this way contains the compiled interface, so installing it needs
Python only — no Node.js.
