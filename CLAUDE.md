# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a full-stack web application for visualizing and validating animal tracking data from FlyHostel experiments (fly behavior research at Liu Lab @ VIB-KU Leuven). It displays identity assignments produced by YOLOv7+idtrackerai, allowing researchers to review and correct them frame-by-frame.

- **Backend**: Python Flask API (`idtrackerai_validator_server/`)
- **Frontend**: React 18 app (`idtrackerai-validator-client/`, git submodule)

## Running the Server

**Backend (default port 5000):**
```bash
VALIDATOR_EXPERIMENT=FlyHostel4/6X/2023-08-31_13-00-00 ./start_server.sh
# or on port 5001:
VALIDATOR_EXPERIMENT=FlyHostel4/6X/2023-08-31_13-00-00 ./start_server_5001.sh
```

Or directly:
```bash
VALIDATOR_EXPERIMENT=FlyHostel4/6X/2023-08-31_13-00-00 BACKEND_PORT=5000 \
  python idtrackerai_validator_server/app.py
```

**Frontend:**
```bash
cd idtrackerai-validator-client
npm start   # dev server on port 3000
npm run build
```

## Required Environment Variables

| Variable | Required | Description |
|---|---|---|
| `VALIDATOR_EXPERIMENT` | Yes | Experiment path, e.g. `FlyHostel4/6X/2023-08-31_13-00-00` or underscore form `FlyHostel4_6X_2023-08-31_13-00-00` |
| `FLYHOSTEL_VIDEOS` | Yes | Root directory containing all experiment folders and `index.txt` |
| `BACKEND_PORT` | No | Flask port (default: 5000) |
| `USE_VAL` | No | `"True"` or `"False"` — override auto-detection of validated DB tables |

## Installation

```bash
pip install -e .
```

The `idtrackerai` and `flyhostel` packages are external dependencies not published to PyPI — they must be installed from source separately.

## Architecture

### Backend (`idtrackerai_validator_server/`)

- **`app.py`**: Flask application — all routes and request handling. The `VALIDATOR_EXPERIMENT` env var selects the experiment at startup. Uses a global thread `Lock` for concurrent frame access.
- **`main.py`**: Entry point — launches `app.py` via subprocess using the conda env's Python. The `start-idtrackerai-validator-server` console script calls this.
- **`backend.py`**: Loads experiments (`load_experiment`), reads idtrackerai config from SQLite, runs segmentation via `idtrackerai._process_frame`, and provides frame annotation/drawing utilities.
- **`database.py`**: `DatabaseManager` dynamically creates Flask-SQLAlchemy models at runtime. Table names have a `_VAL` suffix when the experiment has been validated (e.g., `ROI_0_VAL`, `IDENTITY_VAL`). The suffix is auto-detected by `check_if_validated` or set via `USE_VAL`.
- **`constants.py`**: Feature flags (`INCLUDE_POSE`, `WITH_FRAGMENTS`) and defaults (`first_chunk=50`, `FRAMES_DIR`).
- **`utils.py`**: Loads rejection CSVs from `$FLYHOSTEL_VIDEOS/<experiment>/interactions/`.
- **`pose_reader.py`**: `BehaviorReader`/`GroupBehaviorReader` for reading per-animal MP4 clips. Not currently used by the main app.
- **`databasev1.py`**: Entirely commented out (deprecated).

### Database Schema (SQLite)

Tables accessed (with or without `_VAL` suffix based on validation status):
- `ROI_0` — centroid positions per frame (`frame_number`, `x`, `y`, `in_frame_index`, `area`, `fragment`)
- `IDENTITY` — identity assignments (`frame_number`, `in_frame_index`, `identity`, `local_identity`)
- `CONCATENATION` — chunk-level identity linking
- `METADATA` — experiment metadata (`field`/`value` key-value store: `framerate`, `chunksize`, `date_time`, `idtrackerai_conf`, `ethoscope_metadata`)
- `AI` — frames where AI (YOLOv7 or idtrackerai) intervened

### API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/frame/<frame_number>` | Returns JPEG frame saved to `./frames/`; runs segmentation and updates `contours` |
| `GET /api/tracking/<frame_number>` | Returns centroid positions, identities, ZT time |
| `GET /api/preprocess/<frame_number>` | Returns contours from last `process_frame` call |
| `GET /api/framerate` | Returns recording framerate |
| `GET /api/next_error/<n>`, `prev_error/<n>` | Navigate to frames where `identity=0` |
| `GET /api/next_ok/<n>`, `prev_ok/<n>` | Navigate to frames where all identities are non-zero |
| `GET /api/next_rejection/<n>`, `prev_rejection/<n>` | Navigate to flagged rejection frames |
| `GET /api/next_ai/<n>`, `prev_ai/<n>` | Navigate to frames with AI intervention |

### Frontend (`idtrackerai-validator-client/src/`)

- **`constants.js`**: `BACKEND_SERVER` hostname must match the server (currently hardcoded to `"cv3"`). `BACKEND_PORT` can be overridden via `REACT_APP_BACKEND_PORT` env var.
- **`App.js`**: Root component — fetches frame image, tracking data, and contours in parallel on each frame change. Manages playback loop.
- **`FrameWithSquare.js`**: Canvas rendering (react-konva) of the video frame with overlaid tracking data, contours, and pose keypoints.
- **`buttons.js`**: Navigation buttons (prev/next by 1, 10, 30, chunk; prev/next error/ok/AI/rejection; play/pause).
- **`slider.js`**: Scrubber bar for frame navigation.
- **`queue.js`**: `RequestQueue` serializes image fetches (max 1 simultaneous) to avoid out-of-order frame display.

## Key Configuration Notes

- **`BACKEND_SERVER` in `constants.js`** must be updated when deploying to a different host.
- **`first_chunk` in `constants.py`** controls which video chunk loads at startup (default: 50).
- `INCLUDE_POSE=False` in `constants.py` — pose support is disabled and not fully implemented (raises `NotImplementedError` if enabled).
- Frames are cached to `./frames/` (relative to working directory) and wiped at each server start.
