![build](https://github.com/shaliulab/idtrackerai-validator-server/actions/workflows/ci.yml/badge.svg)

# Purpose

Validate results of experiments in the flyhostel platform.

The application serves both the API and the web interface from a single port,
so no separate frontend server is needed.

# How to install in Linux

Tested on Python 3.10. Building the web interface needs Node.js and npm: install
them from your package manager or from https://nodejs.org if `npm --version` fails.

## Download

```
mkdir -p $HOME/opt
cd $HOME/opt
git clone https://github.com/shaliulab/idtrackerai-validator-server
cd idtrackerai-validator-server
```

## Get the submodules

The web interface (`idtrackerai-validator-client`) and `flyhostel` are git
submodules, and `flyhostel` has submodules of its own. Some of these use ssh URLs
(`git@github.com:…`), and some are private lab repositories.

**With access to the shaliulab repositories and an ssh key on GitHub** (lab members):

```
git submodule update --init --recursive
```

**Without them**, fetch only the public submodules the validator needs, over https:

```
git config --global url."https://github.com/".insteadOf "git@github.com:"
git submodule update --init idtrackerai-validator-client flyhostel
git -C flyhostel submodule update --init libraries/ethoscopy libraries/idtrackerai libraries/imgstore
```

## Make a conda/mamba environment (optional)

```
conda create --name validator python=3.10
conda activate validator
```

## Build the web interface and install

```
cd $HOME/opt/idtrackerai-validator-server
./build_frontend.sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install .
```

`build_frontend.sh` compiles the React client in `idtrackerai-validator-client/`
and copies it into `idtrackerai_validator_server/frontend/`, where the server
looks for it. It runs `git submodule update --init --recursive` first, so without
access to the private submodules it stops there. In that case build the interface
during the install instead:

```
rm -rf idtrackerai_validator_server/frontend
pip install .
```

`pip install .` builds the interface only when `idtrackerai_validator_server/frontend/`
has none; with one already there it installs that one as it is.

idtrackerai needs torch and torchvision at runtime. The first `pip install` gets
the CPU builds, which are enough to serve the app. Skip it if your environment
already has torch, for example a CUDA build.

Use `pip install -e .` instead of `pip install .` if you intend to edit the Python
code. An editable install does **not** build the client, so `./build_frontend.sh`
is still required.

## Install a specific version

Releases are tagged (see https://github.com/shaliulab/idtrackerai-validator-server/releases).
To install one, check out its tag before getting the submodules:

```
git checkout v1.2.1
```

and then follow the steps above from "Get the submodules".

## Update

Get the new code **before** building the interface again: otherwise the old
interface is rebuilt and installed.

```
cd $HOME/opt/idtrackerai-validator-server
git pull                       # or: git fetch --tags && git checkout <version>
git submodule update --init --recursive    # or the public-only commands above
./build_frontend.sh            # or: rm -rf idtrackerai_validator_server/frontend
pip install .
```

Then restart the server.

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

```
start-idtrackerai-validator-server --port 5000 --no-browser > server.log 2>&1
```

The application is then available at http://localhost:5000

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

To serve over https, make a certificate and start the server with it:

```
scripts/make_tls_certificates.sh
start-idtrackerai-validator-server --host 0.0.0.0 --no-browser \
    --certfile ~/.config/idtrackerai-validator/tls/cert.pem \
    --keyfile ~/.config/idtrackerai-validator/tls/key.pem
```

The script makes a small certificate authority (`ca.pem`) and a server certificate
signed by it. Each client computer trusts `ca.pem` once, and then browsers show no
warning. The login page links it as `/ca.crt` with instructions.
`scripts/install_ca_linux_chrome.sh` does it for Chrome on Linux.

An SSH tunnel (below) is the alternative on a shared machine.

## Running on a remote machine

Start the server on the remote host, then forward the single port from your
laptop:

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
