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
| `VALIDATOR_EXPERIMENT` | Yes | Experiment path, e.g. `FlyHostel4/6X/2023-08-31_13-00-00` or underscore form `FlyHostel4_6X_2023-08-31_13-00-00`; an arena of a multi-arena recording: `FlyHostel1/4X/2026-10-13_17-00-00/arenas/1` or `FlyHostel1_4X_2026-10-13_17-00-00_arena1` |
| `FLYHOSTEL_VIDEOS` | Yes | Root directory containing all experiment folders and `index.txt` |
| `BACKEND_PORT` | No | Flask port (default: 5000) |
| `FLYHOSTEL_METADATA` | No | CSV exports of the metadata sheets, for not-analyzed experiments in the Overview (default: `/flyhostel_data/metadata`) |
| `DOWNLOAD_FLYHOSTEL_METADATA` | No | `download_flyhostel_metadata` executable for the Overview's metadata update (default: `~vibflysleep/mambaforge/envs/rapids-23.04/bin/…`) |
| `GOOGLEAPI_CONFIG` | No | Config listing the metadata sheets (`SHEETS`) (default: `~/.config/googleapi/config.yaml`) |
| `USE_VAL` | No | `"True"` or `"False"` — force the tracking source for every experiment. Unset: each loaded experiment starts on validated tables (`_VAL`) when it has >1 fly and those tables exist, else raw tracking (`app.default_use_val`); the "Validated tracking" box switches it until the next experiment load |
| `CVAT_HOST`, `CVAT_USERNAME`, `CVAT_PASSWORD` | No | CVAT server (port 8080) for the Group viewer's CVAT buttons; without them the buttons stay grayed out |
| `FLYHOSTEL_BACKUP` | No | Backup the Data management tab compares experiments with (default: `/flyhostel_data/backup`, same `FlyHostel<N>/<X>X/<date>` layout) |
| `DATA_MANAGEMENT_CACHE` | No | Where the last Data management scan is kept (default: `~/.cache/idtrackerai-validator/data_management.json`) |
| `CVAT_PUBLIC_URL` | No | CVAT URL opened by the browser (default: `http://<CVAT_HOST>:8080`, or the host name the browser used to reach the server when `CVAT_HOST` is local) |

## Installation

```bash
pip install -e .
```

The `idtrackerai` and `flyhostel` packages are external dependencies not published to PyPI — they must be installed from source separately.

## Architecture

### Backend (`idtrackerai_validator_server/`)

- **`app.py`**: Flask application — all routes and request handling. The `VALIDATOR_EXPERIMENT` env var selects the experiment at startup. Uses a global thread `Lock` for concurrent frame access. `_open_experiment` (startup and `/api/load`) also opens experiments without database (not analyzed yet): `db_manager` stays `None`, the SQLAlchemy URI is `:memory:` (sqlite would create an empty `.db`, which the index would then list), and `STORE_INFO` holds the store's frame range, `index.db` (frame times) and the metadata sheet rows (`overview.sheet_flies`: `reference_hour` for ZT, sex/genotype for `/api/animal_metadata`). Frame, contours (`<date>.conf`), frame range, framerate and `/api/tracking?source=blobs` work; database-only routes answer 503/400.
- **`main.py`**: Entry point — launches `app.py` via subprocess using the conda env's Python. The `start-idtrackerai-validator-server` console script calls this.
- **`backend.py`**: Loads experiments (`load_experiment`; `load_experiment_without_database` from `metadata.yaml`, `<date>.conf` and the folder name's start hour, starting at `first_chunk` or the last chunk recorded; imgstore writes `index.db` when the folder has none), reads idtrackerai config from SQLite, runs segmentation via `idtrackerai._process_frame`, and provides frame annotation/drawing utilities.
- **`database.py`**: `DatabaseManager` dynamically creates Flask-SQLAlchemy models at runtime. Table names have a `_VAL` suffix when the experiment has been validated (e.g., `ROI_0_VAL`, `IDENTITY_VAL`). The suffix is auto-detected by `check_if_validated` or set via `USE_VAL`.
- **`experiments.py`**: Experiment names ↔ folders: `to_relpath` (folder relative to `$FLYHOSTEL_VIDEOS`, the form of `SELECTED_EXPERIMENT`), `to_flat` (flyhostel's name), `experiment_folder` (top-level `FlyHostel<N>/<X>X/<date>`), built on `flyhostel.utils.experiment`. Multi-arena recordings keep each arena as its own imgstore + database in `<experiment>/arenas/<n>/` (`…_arena<n>`); use these helpers instead of `replace("/", "_")`. `INDEX_COMMAND` lists the databases of experiment folders and of their arenas (other depth-4 folders pruned).
- **`constants.py`**: Feature flags (`INCLUDE_POSE`, `WITH_FRAGMENTS`) and defaults (`first_chunk=50`, `FRAMES_DIR`).
- **`ethogram.py`**: `register_ethogram(app, ...)` adds `/api/ethogram/*` routes (merged from the behavior-viewer project). Serves pre-rendered plotly ethograms and HLS movies from `<basedir>/motionmapper/<NN>/` (`<fly>.json`, `<fly>.feather`, `movie/movie.m3u8`). The figure x axis is seconds since movie start; frame = `first_chunk*chunksize + t*framerate`, with `first_chunk` read from the feather. A fly with the feather but no JSON gets its figure drawn on first `/figure` request (`_ensure_figure`: `draw_ethogram` on the first 24 h, `t0 = first_chunk*chunksize`, ~15 s; `/api/ethogram/flies` gives `drawn`). Fly ids may carry an arena (`…_arena1__00`). Replaces the behavior-viewer project (its user systemd service, port 5300, is disabled).
- **`overview.py`**: `register_overview(app)` adds `/api/overview`: reads `METADATA` (`ethoscope_metadata`, chunks) of every database in `index.txt` (thread pool, cached per file until its mtime changes) for the Overview tab. Fly columns are discovered, not hard-coded; junk columns (`" "`, `_`, `03-02`) and `NONE`/`nan` values are dropped. `merge_light_cycle` gathers the light cycle (LD/DD/LL), which older sheets wrote in `condition`, `experiment` or at the end of `vial`, into `light_cycle` (in the response only, databases untouched), with `light_cycle_source` per fly and `light_cycle_flags` per experiment (listed under "Non-standard light cycle encoding" in the Overview tab). `scan_not_analyzed` adds the experiments of the metadata sheets (latest `*metadata<N> - <sheet>.csv` in `$FLYHOSTEL_METADATA` of each sheet under `SHEETS` in `$GOOGLEAPI_CONFIG`) whose folder exists but has no `*db` other than `index.db`; rows with an `arena` (basedir `FlyHostel<N>/1X/<date>`, the arena's NX) map to `FlyHostel<N>/*X/<date>/arenas/<arena>` (`_sheet_experiment`), so unanalyzed arenas are listed one by one; `rebuild_index` runs the index command.
- **`metadata_sync.py`**: `register_metadata_sync(app)`: refresh an experiment's `ethoscope_metadata` from the Google sheets in two steps (preview job with checks, then apply), and `predownload` for many experiments (`download_metadata_batch.py`, standalone, reuses googleapi's own filtering so the CSVs equal the CLI's; ~30 s for all). Arenas (`…/arenas/<n>`) download with `--arena <n>` (batch: `arena` per experiment) and are checked against the arena's number of animals (its conf) and `arena` column. Runs `$DOWNLOAD_FLYHOSTEL_METADATA` (default: the `rapids-23.04` env's `download_flyhostel_metadata`, which needs a valid token in `~/.config/googleapi/`; 300 s timeout).
- **`jobs.py`**: In-memory registry of background jobs (daemon threads) with per-item progress, polled by the frontend console. A job may carry a JSON `result` for the client. The last 50 finished jobs are kept.
- **`cvat.py`**: `register_cvat(app, ...)` adds `/api/cvat`: CVAT projects named after the loaded experiment (`FlyHostel1_3X_<date>`, optionally with a suffix such as `_long`; exact name first) and every frame of their tasks as `frame_number -> [task, job, task frame]` (image `<experiment>_<chunk>_<frame0>_<block>.png` = frame `frame0 + block`; the exact-name project's, then the newest task wins). Cached 5 min per experiment (~5 s to build). `build_index`, `configured`, `public_base` and `frame_url` are also used outside Flask by flyhostel's `integrate-human-annotations` to print the CVAT links of frames that still have problems.
- **`data_management.py`**: `register_data_management(app, ...)` adds `/api/data_management*`: every `FlyHostel<N>/<X>X/<date>/` folder of `$FLYHOSTEL_VIDEOS` walked (symlinks inside not followed) and compared with its copy in `$FLYHOSTEL_BACKUP`. `is_critical` mirrors what flyhostel's backup (`flyhostel/scripts/main.py` → `FlyHostelGroup.backup`, `FlyHostelLoader.backup`) copies, raw chunks for chunks < `BACKUP_CHUNKS` (400), pose only for those tracked in `ROI_0` (`tracked_chunks`: first/last chunk, separate MIN and MAX queries so both use the index); a critical file is backed up when its copy has the same size and is not older. Status `ok` / `outdated` / `incomplete` / `none`; `problems` lists tracked chunks (`tracking_gaps`) or raw chunks (`raw_gaps`) that are not consecutive, shown as "⚠ gaps"; `local_only_bytes` = everything without an identical copy. Scans run as a job (one thread per FlyHostel, minutes on the HDDs), kept in `$DATA_MANAGEMENT_CACHE`. Delete rescans first and refuses unless `ok`, the experiment is not loaded and its folder does not resolve into the backup; it then removes the folder and its `index.txt` line.
- **`blobs_collection.py`**: `read_blobs` gives `/api/tracking?source=blobs` its rows from idtrackerai's ListOfBlobs of the frame's chunk (`<basedir>/idtrackerai/session_<chunk:06d>/`, most advanced of `tracking/blobs_collection_no_gaps.npy`, `tracking/blobs_collection.npy`, `preprocessing/blobs_collection{_interpolated,,_segmented}.npy`), one row per final centroid, identity 0 when unassigned. Unpickling takes ~4 s, so the last 3 files are cached and reloaded when their mtime changes (ongoing runs rewrite them).
- **`utils.py`**: Loads rejection CSVs from `$FLYHOSTEL_VIDEOS/<experiment>/interactions/`, and per-fly asleep frames (`load_sleep_data`, cached in `SLEEP_CACHE`, cleared on experiment switch).
- **`pose_reader.py`**: 
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
| `GET /api/list` | `experiments` (index.txt, flat names) and `not_analyzed` (metadata-sheet experiments with a folder but no database, as in the Overview); the picker shows them in two groups |
| `POST /api/load`, `GET /api/current_experiment` | Load an experiment (with or without database; without, `first_frame` is in the first chunk idtrackerai has a session for, else `first_chunk`); both return `experiment`, `has_database`, `has_blobs` |
| `GET /api/frame/<frame_number>` | JPEG of the frame (encoded in memory; requests may overlap, only `cap.get_image` holds the lock) |
| `GET /api/tracking/<frame_number>[?source=blobs]` | Returns centroid positions, identities, ZT time. `source=blobs` reads them from the chunk's `blobs_collection*.npy` instead of the database (no pose) and adds `blobs` (`chunk`, `path`, `mtime` or `error`). Selector: "Tracking" in the app bar (Database / blobs_collection / None = recording only; options the experiment lacks are disabled; each load starts on Database, else blobs_collection, else None) |
| `GET /api/preprocess/<frame_number>` | Contours of that frame, segmented on demand and cached (only requested by the client when `PRINT_CONTOUR`) |
| `GET /api/framerate` | Returns recording framerate |
| `GET /api/version` | Commit of the root repository (`commit`, `date`, `dirty`), read once at server start; shown in the app bar's top right corner |
| `GET /api/next_error/<n>`, `prev_error/<n>` | Navigate to frames where `identity=0` |
| `GET /api/next_ok/<n>`, `prev_ok/<n>` | Navigate to frames where all identities are non-zero |
| `GET /api/next_rejection/<n>`, `prev_rejection/<n>` | Navigate to flagged rejection frames |
| `GET /api/next_ai/<n>`, `prev_ai/<n>` | Navigate to frames with AI intervention |
| `GET /api/sleep/<prev|next>/<identity>/<n>` | Start of the previous / next sleep bout of a fly (an ongoing bout is skipped) |
| `GET /api/ethogram/flies` | Flies of the loaded experiment, with `available` = ethogram JSON exists |
| `GET /api/ethogram/<fly>/figure`, `/properties`, `/movie/<file>` | Plotly figure JSON; framerate/chunksize/first_chunk/has_movie; HLS playlist and segments |
| `GET /api/ethogram/<fly>/locate?frame_number=N` or `?zt=S` | Converts between frame number (since chunk 0), ZT seconds (since ZT0 of day 1, flyhostel convention) and movie time |
| `GET /api/ethogram/<fly>/movie_frame?frame_number=N` or `?zt=S` | JPEG of the fly's movie at that time |
| `GET /api/ethogram/<fly>/sleep_bout?frame_number=N` | Sleep bout ongoing at N, else the next one; 404 if none left |
| `GET /api/ethogram/<fly>/sleep_bout/prev`, `/next?frame_number=N` | Start of the previous / next sleep bout; an ongoing bout is skipped in both directions (same logic as `/api/sleep/<dir>/...`, see `utils.find_sleep_bout`) |
| `GET /api/ethogram/<fly>/sleep_bout/video?frame_number=N` | MP4 of that bout cut from the fly's movie (whole 10 s HLS segments, stream copy), named `<fly>_sleep_bout_t0-<ZT s>_t1-<ZT s>.mp4` |
| `POST /api/ethogram/<fly>/sleep_bouts/videos` | Starts a job making one MP4 per sleep bout (or only the current/next one with `{"frame_number": N}`); `{"save": true}` writes them to `<basedir>/flyhostel/videos/<fly>/`, otherwise the result (mp4 or zip) is offered at `/api/jobs/<id>/download` |
| `GET /api/ethogram/<fly>/saved_videos` | Sleep bout videos in `<basedir>/flyhostel/videos/<fly>/`, parsed from their `_t0-<ZT>_t1-<ZT>` names and placed on the movie's time axis (one jump button each in the Ethograms tab) |
| `GET /api/pe/human_annotations`, `DELETE` same | Counts / deletes the loaded experiment's human-made rows in `pe_annotations.db` (`pe_annotations`, `gt_segments`, `gt_windows`; auto segmentation is kept). `DELETE` needs `{"experiment": <loaded experiment>}` and first dumps the rows to `<db dir>/pe_annotations_deleted/<experiment>_<timestamp>.json`. Button: "Delete human annotations…" in the PE tab (`DeletePEAnnotations.js`) |
| `GET /api/overview[?refresh=1]` | Every experiment of `index.txt`: path fields (flyhostel, group size, date), `has_val`, `duration_h`, `flies` (one dict per `ethoscope_metadata` row), plus `fly_columns` ordered by coverage. Also the experiments of the metadata sheets with a folder but no database (`analyzed: false`, `sheets`, flies from the sheet rows), and `warnings` |
| `POST /api/overview/reindex` | Job rewriting `$FLYHOSTEL_VIDEOS/index.txt` (`experiments.INDEX_COMMAND`: the depth-4 `find` as before plus `arenas/<n>/*db`, replaced only if find listed something; one at a time, 409 otherwise); the count is in `items[0].count`. Button: "Rebuild index" in the Overview tab |
| `POST /api/metadata/preview` | `{"experiment"}`: job running `download_flyhostel_metadata` for that FlyHostel number + date; `result` = new rows, the database row of the same identity (`previous`), `changed_columns`, `checks` (rows = animals, flyhostel_number/date/time, identity, genotype, light cycle). Nothing written |
| `POST /api/metadata/predownload`, `GET /api/metadata/predownloaded` | `{"experiments": [...]}`: job downloading each sheet once (`download_metadata_batch.py`, run with the downloader env's python) and keeping every experiment's CSV in memory; previews then use it (`"fresh": true` in the preview downloads again). Result / GET: per experiment `status` (same, minor, filled, changes, error, failed), `key_changes`, `key_filled`, `errors`. Button: "Predownload sheet metadata" above the experiments table, which shows a Sheet column |
| `POST /api/metadata/apply` | `{"experiment", "job_id", "force"}`: writes that preview to `METADATA.ethoscope_metadata` (and `metadata.csv` if present), old values backed up in `<experiment>/metadata_backups/`; failed checks need `force`; each preview can be written once. Dialog: `MetadataUpdate.js`, opened from the Overview tab |
| `GET /api/jobs/<id>`, `/api/jobs/<id>/download` | Progress of a background job (one item per video, progress 0–1); its result file, downloadable once |
| `GET /api/data_management` | Every experiment folder with its last scan (`status`, `local_bytes`, `backup_bytes`, `local_only_bytes` and `local_only` per top-level entry, `missing` / `changed` critical files, `in_dropbox_index` from `$FLYHOSTEL_VIDEOS/<FlyHostel>_dropbox_index.txt`), plus `disks` (free space per drive) and `loaded` |
| `POST /api/data_management/scan` | `{"experiments": [...]}` (default all): job rescanning them; one at a time (409) |
| `POST /api/data_management/delete` | `{"experiment", "confirm": <same name>}`: job deleting that experiment folder if its backup is `ok` (rescanned first); result `freed`, `index_lines_removed`. Dialog in `DataManagementTab.js` |
| `GET /api/cvat[?refresh=1]` | `configured`, `base_url` (for the browser), `projects` (`id`, `name`, `tasks`), `frames` (frames selected for CVAT validation → `[task, job, task frame]`). Buttons: "CVAT project" / "Frame in CVAT" in the Group viewer (`CvatLinks.js`) |

### Frontend (`idtrackerai-validator-client/src/`)

- **`index.css`**: theme tokens as CSS variables (`<html data-theme="light|dark">`, set by `App.js`) and the shared classes (`.card`, `.field`, `.toolbar`, `.icon-btn`, …). `THEMES` in `App.js` mirrors them for components still taking a `theme` prop.
- **`constants.js`**: `BACKEND_SERVER` hostname must match the server (currently hardcoded to `"cv3"`). `BACKEND_PORT` can be overridden via `REACT_APP_BACKEND_PORT` env var.
- **`App.js`**: Root component — owns the `FramePlayer` (seeks on every `frameNumber` change, runs playback while `isPlaying`). `experimentInfo` (`has_database`, `has_blobs` from `/api/current_experiment`) sets the tracking source and grays out the PE and Ethograms tabs (and Ctrl+2/3) for experiments without database. The Group viewer frame is sized to fill the height left above the console (at most half the window width) until the splitter beside it is dragged (width saved as a share of the window in `fh-frame-width-share`; double click returns to auto-fit); the side panel fills the available height.
- **`FrameWithSquare.js`**: Canvas rendering (react-konva) of the video frame with overlaid tracking data, contours, and pose keypoints. The hover magnifier's border takes the identity colour of the fly nearest the cursor, whose identity is shown in its top-left corner.
- **App bar** (`App.js`): `VersionTag` (server commit, `*` = uncommitted changes) and `Logos`, two images loaded from `public/logos/logo1.png` and `logo2.png` (hidden while missing; see `LOGOS`).
- **`layout.js`**: hooks `useFillHeight` (height down to the console), `useElementSize` (ResizeObserver), `useStoredNumber` (localStorage-backed number).
- **`Splitter.js`**: Drag handle used to resize panels (frame | side panel, console).
- **`CvatLinks.js`**: Group viewer Navigation card: "CVAT project" (opens `/projects/<id>`) and "Frame in CVAT" (opens `/tasks/<t>/jobs/<j>?frame=<f>`, grayed out unless the displayed frame is one of the CVAT-selected frames, looked up client side in the `/api/cvat` index fetched once per experiment).
- **`buttons.js`**: Navigation buttons (prev/next by 1, 10, 30, chunk; prev/next error/ok/AI/rejection; play/pause).
- **`slider.js`**: Scrubber bar for frame navigation.
- **`EthogramViewer.js`**: "Ethograms" tab (Ctrl+3), lazy-loaded because it bundles plotly. One fly at a time via dropdown / ◀ ▶ / ↑↓; ←/→ pan; syncs with the HLS movie; "Open frame in viewer" jumps the idtrackerai tab to the cursor frame. Its `request` prop ({fly, frameNumber}) opens a fly at a frame: used by the blobs table "etho" button.
- **`PEValidator.js`**: PE tab. Fills the window down to the console: a controls card, the media card (video | pose | traces; square panels sized from its height, auto ≈60% of the room, draggable, `fh-pe-media-height`) and the bouts table; "Current media" is a resizable side panel (`fh-pe-side-width`). Styles: `.pe-*` in `index.css`. "Find PE near" a ZT (s) or frame number selects the closest PE bout (conversion via `/api/ethogram/<fly>/locate`); bursts are ordered by score or, with "sort chronologically", by first frame (audit mode keeps the audit order).
- **`OverviewTab.js`**: "Overview" tab (Ctrl+4), dashboard of all experiments from `/api/overview` (fetched when first shown). Counting is client side on fly rows; clicking a bar toggles a filter (each chart ignores its own filter); a "Find experiments" query (genotype, living with another genotype / no one / only its own, optional exact number of flies of each of the two genotypes, date range; `applyQuery`, stored in `fh-overview-query`) applied before the bar filters, a derived `cohoused_with` property (other flies' genotypes), experiments/flies toggle, cross-tab, experiments table whose "Open" loads the experiment (`openExperiment` in `App.js`), CSV export. Chart choice in `fh-overview-*` localStorage. Styles: `.ov-*` in `index.css`.
- **`DataManagementTab.js`**: "Data management" tab (Ctrl+6): experiment folders with size on disk / in backup / not in backup, backup status (▸ lists missing or changed critical files), Dropbox index, free space per drive; "Scan all" / ↻ start scan jobs; "Delete" (only for `ok`, not loaded) opens a dialog that needs the experiment name typed.
- **`HelpTab.js`**: "How to use" tab (Ctrl+5), static user guide with a contents sidebar and a shortcut table. Keep it in sync when buttons, shortcuts or workflows change.
- **`blobs_table.js`**: `BlobsTable` (viewer's per-fly table; `readOnly` drops the editable frame/chunk inputs and the sleep/etho columns) and `FlyRow`, the read-only row of one fly at a given frame (fetched from `/api/tracking/<frame>`), shown in the PE tab (selected bout's start) and the Ethograms tab (cursor).
- **`MessageConsole.js`**: Console docked at the bottom of the app. Its height is dragged from its top edge (`fh-console-height`). `ConsoleProvider` (in `index.js`) exposes `useConsole()` → `log(level, text)` and `trackJob(job)` (polls `/api/jobs/<id>`, shows a progress bar per item, downloads the result).
- **`framePlayer.js`**: `FramePlayer` loads image + tracking (+ contours) per frame as one bundle and caches them. `seek(n)`: the latest request wins, frames are shown in request order (fast scrolling skips frames, never shows them out of order). `play()`: wall-clock playback that prefetches one frame ahead and drops frames when the server is slower than the requested speed. Mouse: wheel scrubs (pauses playback), middle click plays/pauses.

## Key Configuration Notes

- **`BACKEND_SERVER` in `constants.js`** must be updated when deploying to a different host.
- **`first_chunk` in `constants.py`** controls which video chunk loads at startup (default: 50).
- `INCLUDE_POSE=False` in `constants.py` — pose support is disabled and not fully implemented (raises `NotImplementedError` if enabled).
- `./frames/` (`FRAMES_DIR`) is still wiped at server start but no longer written: frames are encoded in memory.
- Sleep-bout video helpers live in flyhostel (`flyhostel/data/pose/loaders/sleep_videos.py`, also `FlyHostelLoader.record_all_sleep_bouts()`); `ethogram.py` and `utils.py` import them. The server env imports flyhostel from wherever it is installed, so that copy must contain the module.
