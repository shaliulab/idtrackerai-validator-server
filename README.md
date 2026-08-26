![build](https://github.com/shaliulab/idtrackerai-validator-server/actions/workflows/ci.yml/badge.svg)

# Purpose

Validate results of experiments in the flyhostel platform.

The application serves both the API and the web interface from a single port,
so no separate frontend server is needed.

# How to install in Linux

## Download

The client is a git submodule, so clone recursively:

```
mkdir -p $HOME/opt
cd $HOME/opt
git clone --recurse-submodules https://github.com/shaliulab/idtrackerai-validator-server
```

If you already cloned without `--recurse-submodules`:

```
cd $HOME/opt/idtrackerai-validator-server
git submodule update --init --recursive
```

## Make a conda/mamba environment (optional)

```
conda create --name validator python=3.7.12
conda activate validator
```

## Build the web interface

The interface is written in React and has to be compiled once before the server
can serve it. This requires Node.js and npm — install them from your package
manager or from https://nodejs.org if `npm --version` fails.

```
cd $HOME/opt/idtrackerai-validator-server
./build_frontend.sh
```

This runs `npm ci && npm run build` in `idtrackerai-validator-client/` and copies
the result into `idtrackerai_validator_server/frontend/`, where the Flask app
looks for it. Re-run it whenever you change the client.

## Install the Python package

Tested on Python 3.7.12.

```
cd $HOME/opt/idtrackerai-validator-server
pip install "./idtrackerai[gpu]"
pip install ./flyhostel
pip install .
```

The `[gpu]` extra pulls in torch and torchvision, which idtrackerai needs at
runtime. Quote it — the brackets are shell glob characters.

Use `pip install -e .` instead of `pip install .` if you intend to edit the
Python code. Note that an editable install does **not** build the client, so
`./build_frontend.sh` is still required.

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
| `--port` | 5000 | port to listen on |
| `--host` | 127.0.0.1 | interface to bind; `0.0.0.0` accepts connections from the network |
| `--no-browser` | off | do not open a browser window on startup |
| `--threads` | 8 | waitress worker threads |
| `--debug` | off | Flask development server with reloader; loopback only |

`--host 0.0.0.0` makes the application reachable by anyone who can reach the
port. There is no authentication, so prefer an SSH tunnel (below) on a shared
machine.

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

This starts the CRA dev server on port 3000 and proxies API calls to the backend
on port 5000, so run the Python server alongside it. Forward both ports if you
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