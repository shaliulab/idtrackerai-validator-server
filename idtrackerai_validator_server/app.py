import hmac
import io
import os
from threading import Lock, Timer
import shutil
import argparse
from collections import OrderedDict
import re
import traceback
import logging
from flask import Flask, Response, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from flask import g
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
import cv2
from sqlalchemy import func, create_engine
from sqlalchemy.orm import Session
from flask import session
import h5py
from pathlib import Path

from flyhostel.data.human_validation.utils import check_if_validated
from flyhostel.utils import (
    get_identities,
    get_number_of_animals,
    get_square_width,
    get_square_height,    
)


from idtrackerai_validator_server.constants import (
    WITH_FRAGMENTS, first_chunk, FRAMES_DIR, INCLUDE_POSE
)
from idtrackerai_validator_server.database import DatabaseManager
from idtrackerai_validator_server.experiments import to_flat, to_relpath
from idtrackerai_validator_server.backend import (
    load_experiment,
    generate_database_filename,
    process_frame,
    list_experiments,
    has_database,
    load_experiment_without_database,
    frame_time_from_index,
)
from idtrackerai_validator_server.pe_validation import register_pe_validation
from idtrackerai_validator_server.ethogram import register_ethogram
from idtrackerai_validator_server.deg_labels import register_deg_labels
from idtrackerai_validator_server.jobs import register_jobs
from idtrackerai_validator_server.overview import register_overview, scan_not_analyzed, sheet_flies
from idtrackerai_validator_server.metadata_sync import register_metadata_sync
from idtrackerai_validator_server.validation_csv import register_validation_csv
from idtrackerai_validator_server.cvat import register_cvat
from idtrackerai_validator_server.data_management import register_data_management
from idtrackerai_validator_server.restore import register_restore
from idtrackerai_validator_server.dropbox_files import register_dropbox_files
from idtrackerai_validator_server.qc import register_qc
from idtrackerai_validator_server.pose import register_pose, get_h5_file, close_h5_files
from idtrackerai_validator_server.monitor import register_monitor
from idtrackerai_validator_server.resources import register_resources
from idtrackerai_validator_server.environment import IncubatorEnvironment
from idtrackerai_validator_server.blobs_collection import (
    read_blobs, clear_cache as clear_blobs_cache, session_chunks, has_blobs_collections,
)
from idtrackerai_validator_server import rois as roi_sources
from idtrackerai_validator_server.utils import load_rejections, sleep_bouts, find_sleep_bout, SLEEP_CACHE as _sleep_cache

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)
logging.getLogger("idtrackerai_validator_server.backend").setLevel(logging.DEBUG)
logging.getLogger("imgstore").setLevel(logging.WARNING)
logging.getLogger("watchdog.observers").setLevel(logging.WARNING)

# Folder relative to $FLYHOSTEL_VIDEOS (FlyHostel1/4X/<date>/arenas/1 for an arena)
SELECTED_EXPERIMENT_ = os.environ.get("VALIDATOR_EXPERIMENT", None)
SELECTED_EXPERIMENT = to_relpath(SELECTED_EXPERIMENT_) if SELECTED_EXPERIMENT_ else None

# USE_VAL="True"/"False" forces the tracking source for every experiment; unset,
# each loaded experiment starts from default_use_val() and the "Validated
# tracking" box switches it until the next experiment is loaded.
USE_VAL = os.environ.get("USE_VAL", None)
if USE_VAL is not None:
    USE_VAL = USE_VAL == "True"


def default_use_val(experiment):
    """Validated tracking (the *_VAL tables) for groups, raw tracking for single
    flies; never the validated tables when the experiment has none."""
    if USE_VAL is not None:
        return USE_VAL
    try:
        group = get_number_of_animals(to_flat(experiment)) > 1
    except Exception as error:
        logger.warning("Could not count the animals of %s: %s", experiment, error)
        group = True
    if not group:
        return False
    validated = check_if_validated(generate_database_filename(experiment)) == "_VAL"
    if not validated:
        logger.warning("%s has no validated tables, showing raw tracking", experiment)
    return validated

lock = Lock()

# Initialize application with CORS settings
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

app = Flask(__name__, static_folder=None)


app.config['SECRET_KEY'] = 'FLYHOSTEL_1234'
CORS(app)

# HTTP Basic Auth on every route when VALIDATOR_USERNAME and VALIDATOR_PASSWORD are set
AUTH_USERNAME = os.environ.get("VALIDATOR_USERNAME")
AUTH_PASSWORD = os.environ.get("VALIDATOR_PASSWORD")


@app.before_request
def require_login():
    if not (AUTH_USERNAME and AUTH_PASSWORD):
        return None
    auth = request.authorization
    if (
        auth is not None
        and hmac.compare_digest((auth.username or "").encode(), AUTH_USERNAME.encode())
        and hmac.compare_digest((auth.password or "").encode(), AUTH_PASSWORD.encode())
    ):
        return None
    return Response(
        "Login required", 401,
        {"WWW-Authenticate": 'Basic realm="idtrackerai validator", charset="UTF-8"'},
    )

register_pe_validation(app, get_selected_experiment=lambda: SELECTED_EXPERIMENT)
register_ethogram(app, get_selected_experiment=lambda: SELECTED_EXPERIMENT)
register_deg_labels(app)
register_jobs(app)
register_overview(app)
register_metadata_sync(app)
register_validation_csv(app, get_selected_experiment=lambda: SELECTED_EXPERIMENT)
register_cvat(app, get_selected_experiment=lambda: SELECTED_EXPERIMENT)
register_data_management(app, get_selected_experiment=lambda: SELECTED_EXPERIMENT)
register_restore(app)
register_dropbox_files(app)
register_monitor(app)
register_resources(app)


def _qc_state():
    if db_manager is None:
        return None
    return {"experiment": SELECTED_EXPERIMENT, "dbfile": db_manager.dbfile,
            "table": db_manager.tables["IDENTITY"].__tablename__,
            "n_animals": NUMBER_OF_ANIMALS, "chunksize": int(CHUNKSIZE), "framerate": float(FRAMERATE)}


register_qc(app, get_state=_qc_state)


def _fly_centroid(frame_number, identity):
    """(x, y) of the fly with that identity at that frame (with 1 fly, its only row)."""
    tables = db_manager.tables
    rows = tables["ROI_0"].query.filter_by(frame_number=frame_number).all()
    if NUMBER_OF_ANIMALS == 1:
        return (rows[0].x, rows[0].y) if rows else None
    ids = {r.in_frame_index: r.identity for r in tables["IDENTITY"].query.filter_by(frame_number=frame_number).all()}
    return next(((r.x, r.y) for r in rows if ids.get(r.in_frame_index) == identity), None)


def _pose_state():
    if db_manager is None:
        return None
    return {"experiment": SELECTED_EXPERIMENT, "chunksize": int(CHUNKSIZE), "framerate": float(FRAMERATE),
            "centroid": _fly_centroid}


register_pose(app, get_state=_pose_state)

# Clean up previous frames
if os.path.exists(FRAMES_DIR):
    shutil.rmtree(FRAMES_DIR)

# Use a placeholder URI until an experiment is loaded via POST /api/load (and for
# experiments without database: sqlite would create an empty one)
if SELECTED_EXPERIMENT is not None and has_database(SELECTED_EXPERIMENT):
    database_file = generate_database_filename(SELECTED_EXPERIMENT)
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{database_file}"
    app.config['SQLALCHEMY_BINDS'] = {SELECTED_EXPERIMENT: f"sqlite:///{database_file}"}
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///:memory:"
    app.config['SQLALCHEMY_BINDS'] = {}

db = SQLAlchemy(app)

cap = None
frame = None
contours = []
offset = None
CHUNKSIZE = None
FRAMERATE = None
IDTRACKERAI_CONFIG = None
db_manager = None
# Experiments without database (not analyzed yet, e.g. idtrackerai still running):
# {frame_min, frame_max, index_db, chunk, flies (metadata sheet rows)}; None otherwise
STORE_INFO = None
HAS_BLOBS = False
NUMBER_OF_ANIMALS = None
# Incubator temperature / humidity of the loaded experiment
ENVIRONMENT = IncubatorEnvironment()


def _number_of_animals(experiment, flies=None):
    """Animals tracked: the NX of the name, except in an arena (its idtrackerai conf or
    database; else its rows in the metadata sheets, `flies`)."""
    try:
        return int(get_number_of_animals(to_flat(experiment)))
    except Exception as error:
        logger.warning("Could not count the animals of %s: %s", experiment, error)
        if flies:
            return len(flies)
        return int(re.search(r"(\d+)X", experiment).group(1))


def _open_experiment(experiment):
    """Open `experiment` (slash form) and set the globals. Without database, only its
    recording (and the blobs_collections idtrackerai has written) can be shown."""
    global SELECTED_EXPERIMENT, cap, db_manager, offset, CHUNKSIZE, FRAMERATE, IDTRACKERAI_CONFIG
    global STORE_INFO, HAS_BLOBS, NUMBER_OF_ANIMALS
    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], experiment)
    if has_database(experiment):
        manager = DatabaseManager(app, db, with_fragments=WITH_FRAGMENTS, experiment=experiment, use_val=default_use_val(experiment))
        logger.info("Validation status: %s", manager.use_val)
        _, new_cap, experiment_metadata, config = load_experiment(experiment, first_chunk, manager)
        if experiment_metadata is None:
            raise RuntimeError(f"Failed to load experiment metadata for {experiment}")
        store_info = None
        blobs = has_blobs_collections(basedir)
    else:
        manager = None
        chunks = session_chunks(basedir)
        flies = sheet_flies(os.environ["FLYHOSTEL_VIDEOS"], experiment)
        reference_hour = next((f["reference_hour"] for f in flies if f.get("reference_hour")), None)
        if reference_hour is None:
            logger.warning("No reference_hour for %s in the metadata sheets: no ZT", experiment)
        # Start where idtrackerai started
        chunk = first_chunk if first_chunk in chunks or not chunks else chunks[0]
        new_cap, experiment_metadata, config, store_info = load_experiment_without_database(experiment, chunk, reference_hour)
        store_info["flies"] = flies
        blobs = bool(chunks)

    SELECTED_EXPERIMENT = experiment
    db_manager = manager
    cap = new_cap
    offset, CHUNKSIZE, FRAMERATE = experiment_metadata
    IDTRACKERAI_CONFIG = config
    STORE_INFO = store_info
    HAS_BLOBS = blobs
    NUMBER_OF_ANIMALS = _number_of_animals(experiment, store_info["flies"] if store_info else None)
    ENVIRONMENT.open(basedir)


def _experiment_info():
    basedir = None if SELECTED_EXPERIMENT is None else os.path.join(os.environ["FLYHOSTEL_VIDEOS"], SELECTED_EXPERIMENT)
    return {"experiment": SELECTED_EXPERIMENT, "has_database": db_manager is not None, "has_blobs": HAS_BLOBS,
            # the folder, and where it really is (FlyHostel<N> links point to other drives)
            "basedir": basedir, "realpath": basedir and os.path.realpath(basedir)}


if SELECTED_EXPERIMENT is not None:
    with app.app_context():
        _open_experiment(SELECTED_EXPERIMENT)
# Contours of the last frames segmented by /api/preprocess
CONTOURS_CACHE_SIZE = 500
_contours_cache = OrderedDict()
_contours_lock = Lock()

# Bodypart indices to keep (figure this out from step 1)
BODYPARTS_TO_IGNORE = [12, 13, 14, 15, 16, 17]  # ← UPDATE THIS
BODYPARTS_TO_KEEP = [i for i in range(18) if i not in BODYPARTS_TO_IGNORE]

# Map bodypart index to name
BODYPART_NAMES = {
    0: "proboscis",
    1: "thorax",
    2: "abdomen",
    3: "fLL",
    4: "mLL",
    5: "rLL",
    6: "fRL",
    7: "mRL",
    8: "rRL",
    9: "head",
    10: 'lW',
    11: 'rW',
    
}

def get_pose_from_h5(fly_id_str, frame_number, experiment, chunksize):
    """
    Extract pose from H5 file, properly handling chunk offsets.
    
    CRITICAL: The experiment doesn't start at frame 0 of the H5 file.
    It starts at first_chunk * chunksize.
    
    Args:
        fly_id_str: e.g., "FlyHostel1_1X_2026-06-13_17-00-00__01"
        frame_number: frame number in the EXPERIMENT (0-indexed, where 0 is first_chunk*chunksize)
        experiment: experiment path
        chunksize: frames per chunk (from METADATA table)
        first_chunk: first chunk number of this experiment (from constants or METADATA)
    
    Returns:
        Dict {bodypart_name: [x, y]} (coords RELATIVE to centroid) or None
    
    Example:
        If first_chunk=5, chunksize=10000:
        - Experiment frame 0 = Video frame 50000 (chunk 5, position 0)
        - Experiment frame 15000 = Video frame 65000 (chunk 6, position 5000)
    """
    try:
        # Step 6: Open the H5 file and validate bounds
        h5_file = get_h5_file(fly_id_str, experiment)

        if h5_file is None:
            logger.debug(f"Could not open H5 file for {fly_id_str}")
            return None
        
        first_chunk=int(os.path.basename(h5_file["files"][0].decode()).split(".")[0])

        # Step 5: Calculate the index WITHIN this H5 file
        # Each chunk spans [chunk_id * chunksize, (chunk_id + 1) * chunksize)
        # So for a frame in chunk N, the index within that chunk is:
        h5_frame_index = frame_number - (first_chunk * chunksize)

        # tracks shape: (1, 2, 18, total_frames_in_chunk)
        # Validate the index
        if h5_frame_index >= h5_file['tracks'].shape[3]:
            logger.error(
                f"H5 frame index {h5_frame_index} out of bounds "
                f"(max: {h5_file['tracks'].shape[3]}) for {fly_id_str}"
            )
            return None
        
        # Step 7: Extract the coordinates for this frame
        xy_coords = h5_file['tracks'][0, :, :, h5_frame_index]  # shape: (2, 18)
        
        # Step 8: Build and return the pose dict
        pose_dict = {}
        for bp_idx in BODYPARTS_TO_KEEP:
            name = BODYPART_NAMES.get(bp_idx, f"bp_{bp_idx}")
            x = float(xy_coords[0, bp_idx])
            y = float(xy_coords[1, bp_idx])
            pose_dict[name] = [x, y]
        
        return pose_dict
    
    except Exception as e:
        logger.error(f"Error extracting pose for {fly_id_str} frame {frame_number}: {e}")
        logger.error(traceback.print_exc())
        return None
    

def _experiment_required():
    return jsonify({"error": "No experiment loaded. POST to /api/load first."}), 503

@app.route("/api/list", methods=["GET"])
def list():
    """experiments: with a database (index.txt, flat names); not_analyzed: experiments
    of the metadata sheets whose folder has no database (as in the Overview)."""
    try:
        result = list_experiments()
    except Exception as error:
        logger.error("Error listing experiments: %s", error)
        result = {"experiments": [SELECTED_EXPERIMENT] if SELECTED_EXPERIMENT else []}
    try:
        records, _ = scan_not_analyzed(os.environ["FLYHOSTEL_VIDEOS"], set())
        result["not_analyzed"] = sorted(to_flat(r["experiment"]) for r in records)
    except Exception as error:
        logger.error("Error listing not-analyzed experiments: %s", error)
        result["not_analyzed"] = []
    return jsonify(result)

@app.route("/api/load", methods=["POST"])
def load():
    global frame, contours

    data = request.get_json()
    if not data or "experiment" not in data:
        return jsonify({"error": "experiment field required"}), 400

    try:
        new_experiment = to_relpath(data["experiment"])
    except Exception:
        return jsonify({"error": f"Not an experiment: {data['experiment']}"}), 400

    if not os.path.exists(os.path.join(os.environ["FLYHOSTEL_VIDEOS"], new_experiment, "metadata.yaml")):
        return jsonify({"error": f"Experiment not found: {new_experiment}"}), 404

    # Without database, an in-memory one: sqlite would create an empty file
    if has_database(new_experiment):
        uri = f"sqlite:///{generate_database_filename(new_experiment)}"
    else:
        uri = "sqlite:///:memory:"

    lock.acquire()
    try:
        db.session.remove()

        app.config["SQLALCHEMY_DATABASE_URI"] = uri
        app.config["SQLALCHEMY_BINDS"] = {new_experiment: uri}

        try:
            engines_dict = db._app_engines[app]
            for eng in engines_dict.values():
                eng.dispose()
            engines_dict.clear()
            new_engine = create_engine(uri)
            engines_dict[None] = new_engine
            engines_dict[new_experiment] = new_engine
        except (AttributeError, KeyError):
            pass

        _open_experiment(new_experiment)
        frame = None
        contours = []
        with _contours_lock:
            _contours_cache.clear()
        _sleep_cache.clear()
        clear_blobs_cache()
        close_h5_files()
        logger.info("Switched to experiment %s", SELECTED_EXPERIMENT)

    except Exception as error:
        print(traceback.print_exc())

        logger.error("Error loading experiment %s: %s", new_experiment, error)
        return jsonify({"error": str(error)}), 500
    finally:
        lock.release()

    first_frame = (STORE_INFO["chunk"] if STORE_INFO else first_chunk) * CHUNKSIZE
    return jsonify({"message": "success", "first_frame": first_frame,
                    "use_val": bool(db_manager is not None and db_manager.use_val), **_experiment_info()})


def row2dict(row):
    d = {}
    for column in row.__table__.columns:
        d[column.name] = getattr(row, column.name)
    return d


@app.route('/api/frame_range', methods=['GET'])
def get_frame_range():
    if STORE_INFO is not None:
        return jsonify({"min_frame": STORE_INFO["frame_min"], "max_frame": STORE_INFO["frame_max"]})
    if db_manager is None:
        return _experiment_required()
    tables = db_manager.tables
    try:
        # tables["IDENTITY"] is already IDENTITY_VAL or IDENTITY,
        # depending on db_manager.use_val — so this respects app.py's choice.
        min_frame, max_frame = db.session.query(
            func.min(tables["IDENTITY"].frame_number),
            func.max(tables["IDENTITY"].frame_number),
        ).one()
        return jsonify({"min_frame": min_frame, "max_frame": max_frame})
    except Exception as error:
        logger.error("Error computing frame range: %s", error)
        return jsonify({"message": str(error)}), 500

@app.route("/api/framerate", methods=['GET'])
def get_framerate():
    if STORE_INFO is not None:
        return str(FRAMERATE)
    if db_manager is None:
        return _experiment_required()
    tables = db_manager.tables
    framerate=tables["METADATA"].query.filter_by(field="framerate").first().value
    return framerate


def _git_version():
    """Commit of the root repository this server runs from, read once at start-up
    (later checkouts do not change the code already running)."""
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def git(*args):
        return subprocess.run(
            ["git", "-C", root, *args], capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    try:
        return {
            "commit": git("rev-parse", "--short", "HEAD"),
            "date": git("log", "-1", "--format=%cs"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except Exception as error:
        logging.warning("Could not read the git commit of %s: %s", root, error)
        return {"commit": None, "date": None, "dirty": False}


GIT_VERSION = _git_version()


@app.route("/api/version", methods=['GET'])
def get_version():
    return jsonify(GIT_VERSION)


@app.route("/api/animal_metadata", methods=['GET'])
def get_animal_metadata():
    if STORE_INFO is not None:
        # Not analyzed: the metadata sheet rows
        result = {}
        flies = STORE_INFO["flies"]
        for i, fly in enumerate(flies):
            identity = fly.get("identity") if len(flies) > 1 else None
            try:
                key = str(int(float(identity))) if identity is not None else str(i if len(flies) == 1 else i + 1)
            except ValueError:
                key = str(i + 1)
            result[key] = {k: fly[k] for k in ("sex", "genotype") if k in fly}
        return jsonify(result)
    if db_manager is None:
        return _experiment_required()
    tables = db_manager.tables
    try:
        row = tables["METADATA"].query.filter_by(field="ethoscope_metadata").first()
        if row is None:
            return jsonify({})
        df = pd.read_csv(io.StringIO(row.value), index_col=0)

        experiment = to_flat(SELECTED_EXPERIMENT)
        try:
            identities = get_identities(experiment)
        except Exception:
            identities = list(range(1, len(df) + 1))

        def safe(v):
            try:
                return '' if pd.isna(v) else str(v)
            except (TypeError, ValueError):
                return str(v) if v is not None else ''

        # Match rows to flies on the `identity` column: the rows are not necessarily
        # in identity order. Without that column (or with a single animal, whose
        # identity is 0 here but usually 1 in the metadata), fall back to row order.
        by_identity = {}
        if 'identity' in df.columns and len(identities) > 1:
            for _, r in df.iterrows():
                try:
                    by_identity[int(r['identity'])] = r
                except (TypeError, ValueError):
                    pass
            if len(by_identity) != len(df):
                logger.warning("ethoscope_metadata of %s: identity column missing or duplicated, "
                               "matching flies by row order", experiment)
                by_identity = {}

        result = {}
        for i, identity in enumerate(identities):
            if by_identity:
                r = by_identity.get(int(identity))
            else:
                r = df.iloc[i] if i < len(df) else None
            if r is not None:
                entry = {}
                if 'sex' in df.columns:
                    entry['sex'] = safe(r.get('sex'))
                if 'genotype' in df.columns:
                    entry['genotype'] = safe(r.get('genotype'))
                result[str(identity)] = entry
        return jsonify(result)
    except Exception as e:
        logger.error("Error fetching animal metadata: %s", e)
        return jsonify({})


@app.route('/api/sleep/<direction>/<int:identity>/<int:frame_number>', methods=['GET'])
def navigate_sleep(direction, identity, frame_number):
    """Return the start of the next / previous sleep bout of `identity`.

    direction: "prev" | "next". An ongoing bout is skipped in both directions:
    next goes to the bout after it, prev to the bout before it.
    Returns {"frame_number": N} or {"frame_number": null} when none found.
    """
    if db_manager is None:
        return _experiment_required()
    if direction not in ('prev', 'next'):
        return jsonify({"error": "direction must be prev or next"}), 400

    experiment = to_flat(SELECTED_EXPERIMENT)
    bout = find_sleep_bout(sleep_bouts(experiment, identity, FRAMERATE), frame_number, direction)
    return jsonify({"frame_number": int(bout[0]) if bout else None})


def _read_frame(frame_number):
    """(image or None, actual frame number). Only cap.get_image needs the lock:
    requests overlap (the client prefetches), so images stay in local variables."""
    with lock:
        try:
            image, (frame_number, _) = cap.get_image(frame_number)
            return image, frame_number
        except (ValueError, AssertionError) as error:
            app.logger.error(f"Can't fetch frame {frame_number}: {error}")
            return None, frame_number


@app.route('/api/frame/<int:frame_number>', methods=['GET'])
def get_frame(frame_number):
    """JPEG of `frame_number` (a blank image if it cannot be read)."""
    global frame

    if cap is None:
        return jsonify({'error': 'Cap could not be loaded'}), 404

    image, frame_number = _read_frame(frame_number)
    if image is None:
        image = np.ones((1000, 1000), np.uint8) * 255 if frame is None else np.ones_like(frame, np.uint8) * 255
    else:
        frame = image

    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 50])
    if not ok:
        return jsonify({'error': f'Could not encode frame {frame_number}'}), 500
    response = send_file(io.BytesIO(jpeg.tobytes()), mimetype="image/jpeg")
    response.headers["X-Frame-Number"] = str(frame_number)
    return response


def _roi_sources():
    return roi_sources.list_sources(
        os.path.join(os.environ["FLYHOSTEL_VIDEOS"], SELECTED_EXPERIMENT), SELECTED_EXPERIMENT,
        IDTRACKERAI_CONFIG if db_manager is not None else None,
    )


@app.route('/api/rois', methods=['GET'])
def get_rois():
    """ROIs and segmentation settings of the loaded experiment: its database's idtrackerai
    config, <date>.toml, landmarks.toml and <date>.conf (those present), read-only."""
    if SELECTED_EXPERIMENT is None:
        return _experiment_required()
    return jsonify({"experiment": SELECTED_EXPERIMENT,
                    "sources": [roi_sources.public(s) for s in _roi_sources()]})


@app.route('/api/preprocess/<int:frame_number>', methods=['GET'])
def get_preprocess(frame_number):
    """Contours of `frame_number`, from idtrackerai's segmentation (computed on demand, cached).
    Segmentation takes ~40% of a frame's cost, so /api/frame does not do it.
    ?source=<id of /api/rois> segments with that source's settings (contours and areas)."""
    global contours

    if cap is None:
        return jsonify({'error': 'Cap could not be loaded'}), 404
    source_id = request.args.get("source")
    source = None
    if source_id:
        source = next((s for s in _roi_sources() if s["id"] == source_id), None)
        if source is None or "error" in source:
            return jsonify({"contours": [], "error": (source or {}).get("error", f"No source {source_id}")}), 404
    # an edited file invalidates its contours
    key = ((source_id, source.get("mtime")) if source else None, frame_number)
    with _contours_lock:
        cached = _contours_cache.get(key)
    if cached is None:
        image, _ = _read_frame(frame_number)
        if image is None:
            return jsonify({"contours": [], "error": f"Frame {frame_number} cannot be read"}), 404
        try:
            if source is None:
                cached = process_frame(image, session.get("idtrackerai_config", IDTRACKERAI_CONFIG)), None
            else:
                cached = roi_sources.segment(image, source)
        except Exception as error:
            logger.error(error)
            cached = [], None
        with _contours_lock:
            _contours_cache[key] = cached
            while len(_contours_cache) > CONTOURS_CACHE_SIZE:
                _contours_cache.popitem(last=False)
    contours, areas = cached
    return jsonify({"contours": contours, "areas": areas, "source": source_id})


def get_pose(db_manager, frame_number):
    identities=get_identities(to_flat(SELECTED_EXPERIMENT))
    pose={}
    for identity in identities:
        pose[str(identity)]=db_manager.get_pose_for_animal(identity, frame_number)
    return pose

def project_to_absolute(pose, centroids):
    centroids_coords={}
    for centroid in centroids:
        centroids_coords[str(centroid["identity"])]=(centroid["x"], centroid["y"])
    
    pose_abs={}
    for identity in centroids_coords:
        pose_={}
        if identity not in pose:
            continue
        for bodypart in pose[identity]:
            if any((coord is None for coord in pose[identity][bodypart])):
                pose_[bodypart]=(None, None)
            else:
                pose_[bodypart]=np.round(np.array(pose[identity][bodypart])-50 + centroids_coords[identity]).tolist()
        
        pose_abs[identity]=pose_
    return pose_abs


def _zt_fields(frame_time):
    """t = seconds since ZT0 and its HH:MM:SS. frame_time is ms since the o'clock hour
    the recording started in; offset is the seconds between ZT0 and that hour."""
    if frame_time is None or offset is None:
        return {"t": None, "ZT": None}
    t = frame_time / 1000 + offset
    hours = str(int(t // 3600)).zfill(2)
    minutes = str(int((t % 3600) // 60)).zfill(2)
    seconds = str(int(t % 60)).zfill(2)
    return {"t": t, "ZT": f"{hours}:{minutes}:{seconds}"}


@app.route('/api/tracking/<int:frame_number>', methods=['GET'])
def get_tracking(frame_number):
    if SELECTED_EXPERIMENT is None or cap is None:
        return _experiment_required()

    number_of_animals = NUMBER_OF_ANIMALS
    logger.debug("Loading tracking data for %s", SELECTED_EXPERIMENT)
    tables = db_manager.tables if db_manager is not None else None

    # source=blobs: rows of idtrackerai's blobs_collection*.npy (e.g. of an ongoing run)
    # instead of ROI_0/IDENTITY
    source = request.args.get("source", "db")
    blobs_info = None

    out = []
    number_of_animals_found = 0
    try:
        chunksize = CHUNKSIZE

        # frame_time (ms since the marked time) lives in STORE_INDEX, keyed by frame_number;
        # without database, in imgstore's index.db
        if tables is None:
            frame_time = frame_time_from_index(STORE_INFO["index_db"], frame_number)
        else:
            try:
                store_row = tables["STORE_INDEX"].query.filter_by(frame_number=frame_number).first()
            except Exception as error:
                app.logger.error(error)
                store_row = None
            frame_time = store_row.frame_time if store_row is not None else None
        zt_fields = {**_zt_fields(frame_time), **ENVIRONMENT.at(frame_time)}

        rows = []
        if source == "blobs":
            basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], SELECTED_EXPERIMENT)
            blob_rows, blobs_info = read_blobs(basedir, frame_number, chunksize)
            for row in blob_rows:
                out.append({"frame_number": frame_number, **zt_fields, **row, "chunksize": chunksize})
            number_of_animals_found = len(out)
        elif tables is not None:
            rows = tables["ROI_0"].query.filter_by(frame_number=frame_number).all()
            identity_table = tables["IDENTITY"].query.filter_by(frame_number=frame_number).all()

        for row in rows:
            identity = None
            local_identity = None
            hit = False
            for id_row in identity_table:
                if row.in_frame_index == id_row.in_frame_index:
                    identity = id_row.identity
                    local_identity = id_row.local_identity
                    hit = True
 
            if not hit:
                identity = None
 
            if row.modified is None:
                modified = 0
            else:
                modified = row.modified
            
            data = {
                "frame_number": frame_number,
                **zt_fields,
                "x": row.x,
                "y": row.y,
                "in_frame_index": row.in_frame_index,
                "fragment": getattr(row, "fragment", -1),
                "area": row.area,
                "identity": identity,
                "local_identity": local_identity,
                "modified": modified,
                "chunksize": chunksize,
            }
 
            number_of_animals_found += 1
            out.append(data)
        
        out = sorted(out, key=lambda x: x["identity"] if x["identity"] is not None else -1)
    except Exception as error:
        app.logger.error(error)
    
    if number_of_animals_found == 0:
        app.logger.warning("No animals found for frame %s", frame_number)
    else:
        app.logger.info("Number of animals found = %s", number_of_animals_found)
 
    # ===== NEW: FETCH POSE DATA FOR EACH ANIMAL FROM H5 FILES =====
    
    pose_absolute = {}
    include_pose = INCLUDE_POSE and request.args.get("pose", "1") != "0" and source != "blobs" and tables is not None
    
    if include_pose:
        try:
            # Get list of fly identities
            experiment=to_flat(SELECTED_EXPERIMENT)
            square_width=get_square_width(experiment)
            square_height=get_square_height(experiment)


            identities = get_identities(experiment)
            identity_to_fly_id = {i: fly_id for i, fly_id in enumerate(identities)}
           
            for animal in out:
                if animal['identity'] is not None and animal['identity'] in identity_to_fly_id.values():
                    try:
                        fly_id_str=str(animal['identity']).zfill(2)
                        pose_relative = get_pose_from_h5(
                            fly_id_str,
                            frame_number,
                            experiment,
                            chunksize
                        )
                        
                        if pose_relative:
                            # Convert from relative (centroid-relative) to absolute coordinates
                            pose_absolute_animal = {}
                            # Pose is relative to top-left of 200x200 square centered at centroid
                            # Top-left corner is at (centroid_x - 100, centroid_y - 100)
                            square_top_left_x = animal['x'] - square_width//2
                            square_top_left_y = animal['y'] - square_height//2
                            
                            for bodypart_name, (rel_x, rel_y) in pose_relative.items():
                                if rel_x is not None and rel_y is not None:
                                    # The H5 coordinates are relative to centroid
                                    abs_x = round(square_top_left_x  + rel_x, 2)
                                    abs_y = round(square_top_left_y  + rel_y, 2)
                                    pose_absolute_animal[bodypart_name] = [None if np.isnan(abs_x) else abs_x, None if np.isnan(abs_y) else abs_y]
                                else:
                                    pose_absolute_animal[bodypart_name] = [None, None]
                            
                            pose_absolute[str(animal['identity'])] = pose_absolute_animal
                    except Exception as e:
                        logger.error(f"Failed to load pose for identity {animal['identity']}: {e}")
                        logger.error(traceback.print_exc())
        except Exception as e:
            logger.error(f"Error in pose processing: {e}")


     
    data = {
        "tracking_data": out,
        "number_of_animals": number_of_animals,
        "pose": pose_absolute
    }
    if blobs_info is not None:
        data["blobs"] = blobs_info

    response=jsonify(data)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/api/prev_rejection/<int:frame_number>', methods=['GET'])
def get_prev_rejection(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_rejection(frame_number, "previous")


@app.route('/api/next_rejection/<int:frame_number>', methods=['GET'])
def get_next_rejection(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_rejection(frame_number, "next")


@app.route('/api/prev_error/<int:frame_number>', methods=['GET'])
def get_prev_error(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_error(frame_number, "previous")


@app.route('/api/next_error/<int:frame_number>', methods=['GET'])
def get_next_error(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_error(frame_number, "next")


@app.route('/api/prev_ok/<int:frame_number>', methods=['GET'])
def get_prev_ok(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_ok(frame_number, "previous")


@app.route('/api/next_ok/<int:frame_number>', methods=['GET'])
def get_next_ok(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_ok(frame_number, "next")

@app.route('/api/prev_ai/<int:frame_number>', methods=['GET'])
def get_prev_ai(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_ai(frame_number, "previous")


@app.route('/api/next_ai/<int:frame_number>', methods=['GET'])
def get_next_ai(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_ai(frame_number, "next")

@app.route('/api/pe/flies', methods=['GET'])
def get_flies():
    if SELECTED_EXPERIMENT is None:
        return jsonify([])
    else:
        experiment=to_flat(SELECTED_EXPERIMENT)
        logger.warning(experiment)
        identities=get_identities(experiment)
        flies = [
            f"{experiment}__{str(identity).zfill(2)}"
            for identity in identities
        ]
        return jsonify(flies)

@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    Timer(0.5, lambda: os._exit(0)).start()
    return jsonify({"message": "Shutting down gracefully..."})


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path.startswith("api/"):
        return jsonify({"error": "unknown endpoint"}), 404

    candidate = os.path.join(FRONTEND_DIR, path)
    if path and os.path.isfile(candidate):
        # Hashed filenames — safe to cache hard.
        return send_from_directory(FRONTEND_DIR, path, max_age=31536000)

    # index.html must never be cached, or a rebuilt bundle won't be picked up.
    return send_from_directory(FRONTEND_DIR, "index.html", max_age=0)

@app.route("/api/use_val", methods=["GET"])
def get_use_val():
    return jsonify({
        "use_val": bool(db_manager.use_val) if db_manager is not None else bool(USE_VAL)
    })

@app.route("/api/current_experiment", methods=["GET"])
def current_experiment():
    return jsonify(_experiment_info())


@app.route("/api/use_val", methods=["POST"])
def set_use_val():
    """Switch the loaded experiment's tracking source; the next /api/load starts
    again from default_use_val()."""
    global db_manager
    data = request.get_json() or {}
    use_val = bool(data.get("use_val"))

    if SELECTED_EXPERIMENT is None:
        return jsonify({"error": "no experiment loaded"}), 400
    if db_manager is None:
        return jsonify({"error": f"{SELECTED_EXPERIMENT} has no database"}), 400

    with lock:
        db_manager = DatabaseManager(
            app, db, with_fragments=WITH_FRAGMENTS,
            experiment=SELECTED_EXPERIMENT, use_val=use_val,
        )
    logger.info("Validation status: %s", db_manager.use_val)
    return jsonify({"use_val": bool(db_manager.use_val)})

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    
    logger.debug("Gracefully shutting down...")
    close_h5_files()  # ← ADD THIS
    db.session.close()
    logger.debug("DB connection closed. Bye!")
 


def get_first_non_zero_frame(sql_session: Session, frame_number: int, direction=True):
    global SELECTED_EXPERIMENT
    tables = db_manager.tables

    if direction == "next":
        filter_condition = tables["IDENTITY"].frame_number > frame_number
    elif direction == "previous":
        filter_condition = tables["IDENTITY"].frame_number < frame_number
    else:
        raise Exception(f"direction must be either next or previous. direction={direction}")

    subquery = (
        sql_session.query(
            tables["IDENTITY"].frame_number,
            func.min(tables["IDENTITY"].identity).label("min_identity")
        )
        .filter(filter_condition)
        .group_by(tables["IDENTITY"].frame_number)
        .subquery()
    )

    if direction=="previous":
        result = (
            sql_session.query(subquery)
            .filter(subquery.c.min_identity != 0)
            .order_by(-subquery.c.frame_number)
            .first()
        )
    
    elif direction=="next":
        result = (
            sql_session.query(subquery)
            .filter(subquery.c.min_identity != 0)
            .order_by(subquery.c.frame_number)
            .first()
        )

    return result.frame_number if result else None

def get_ok(frame_number, direction):
    frame_number= get_first_non_zero_frame(db.session, frame_number, direction)
    logger.debug("get_ok %s", frame_number)
    return jsonify({"frame_number": frame_number})


def get_rejection(frame_number, direction):

    global SELECTED_EXPERIMENT
    experiment=to_flat(SELECTED_EXPERIMENT)

    fn=frame_number
    
    try:
        rejections, features=load_rejections(experiment)

        frames_with_rejection=rejections["first_frame"].values
        diff=frames_with_rejection-frame_number

        if direction=="next":
            frames_with_rejection=frames_with_rejection[diff>0]
            diff=diff[diff>0]
            index=np.argmin(diff)
            fn=frames_with_rejection[index]

        elif direction=="previous":
            frames_with_rejection=frames_with_rejection[diff<0]
            diff=diff[diff<0]
            index=np.argmin(-diff)
            fn=frames_with_rejection[index]
    except KeyError:
        logger.warning("Cannot find %s rejection", direction)
    except FileNotFoundError as error:
        logger.warning(error)


    return jsonify({"frame_number": int(fn)})


def get_error(frame_number, direction):

    global SELECTED_EXPERIMENT
    tables = db_manager.tables

    if direction=="next":
        query=tables["IDENTITY"].query.filter(tables["IDENTITY"].frame_number>frame_number, tables["IDENTITY"].identity==0)
        hit=query.first()
    elif direction=="previous":
        query=tables["IDENTITY"].query.filter(tables["IDENTITY"].frame_number<frame_number, tables["IDENTITY"].identity==0)
        hit=query.order_by(-tables["IDENTITY"].id).first()
    else:
        raise Exception(f"direction must be either next or previous. direction={direction}")

    if hit:
        frame_number=hit.frame_number
    else:
        frame_number=None

    return jsonify({"frame_number": frame_number})


def get_ai(frame_number, direction):
    global SELECTED_EXPERIMENT
    tables = db_manager.tables

    if direction=="next":
        query=tables["AI"].query.filter(tables["AI"].frame_number>frame_number)
        hit=query.order_by(tables["AI"].frame_number).first()
    elif direction=="previous":
        query=tables["AI"].query.filter(tables["AI"].frame_number<frame_number)
        hit=query.order_by(-tables["AI"].frame_number).first()
    else:
        raise Exception(f"direction must be either next or previous. direction={direction}")

    if hit:
        frame_number=hit.frame_number
        ai=hit.ai
    else:
        frame_number=None
        ai=None

    return jsonify({"frame_number": frame_number, "ai": ai})


def get_parser():

    ap=argparse.ArgumentParser()
    ap.add_argument("--port", default=5000, type=int)
    ap.add_argument("--host",default="0.0.0.0")
    return ap

if __name__ == "__main__":

    ap=get_parser()
    args=ap.parse_args()
    app.run(port=args.port, host=args.host, debug=True)  # or set debug=False for production