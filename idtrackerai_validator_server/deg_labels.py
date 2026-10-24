"""
deg_labels.py  —  deepethogram (DEG) labels made in the Ethograms tab.

The behavior classifier (RF) is trained on the human labels of the DEG projects
($DEG_PROJECTS/DEG/FlyHostel_deepethogram for 150 fps recordings, ..._47fps otherwise,
as flyhostel's DEGLoader.get_deg_data_dir). A video folder holds one chunk of one fly:

    <project>/DATA/<experiment>_<chunk:06d>_<local identity:03d>/<chunk:06d>.mp4
                                                                /<chunk:06d>_labels.csv

with one row per frame of the chunk and one column per class of the project: 1 present,
0 absent, -1 not labelled. The local identity is the fly's idtrackerai identity in that
chunk (flyhostel's get_local_identities_from_experiment, also used by DEGLoader to read
the labels back), the video a copy of <basedir>/flyhostel/single_animal/<lid:03d>/<chunk>.mp4.

The Ethograms tab works in 1 s bins of movie time (ethogram.py), so labelling the
seconds [start, end) writes the `framerate` rows of each of them. A label is one of
PRESETS (the RF classes, as flyhostel.data.deg.apply_semantic_rules reads them back) or
any set of the project's columns; the given columns are set to 1, the others to 0.
Clearing sets the rows back to -1. Missing folders and label files are made with
deepethogram (deg_add_video.py, run with $DEG_PYTHON). Every write is logged to
$DEG_LABELS_LOG.
"""
import json
import logging
import os
import re
import subprocess
import threading
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from flask import jsonify, request

from flyhostel.utils import get_basedir, get_local_identities_from_experiment

from idtrackerai_validator_server.ethogram import FLY_PATTERN, _properties, track

logger = logging.getLogger(__name__)

DEG_ROOT = os.path.join(os.environ.get("DEG_PROJECTS", "/flyhostel_data/fiftyone/FlyBehaviors"), "DEG")
DEG_PYTHON = os.environ.get("DEG_PYTHON", os.path.expanduser("~/mambaforge/envs/rapids-23.04/bin/python"))
ADD_VIDEO_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deg_add_video.py")
LOG_PATH = os.environ.get("DEG_LABELS_LOG", os.path.expanduser("~/.cache/idtrackerai-validator/deg_labels.jsonl"))
# Longest range labelled at once (s): every chunk it touches may need a new video folder (~5 s each)
MAX_RANGE = 3600

# RF class -> DEG columns set to 1
PRESETS = {
    "inactive": ["inactive"],
    "inactive+micromovement": ["inactive", "micromovement"],
    "inactive+pe": ["inactive", "pe"],
    "feed": ["feed"],
    "groom": ["groom"],
    "walk": ["walk"],
    "background": ["background"],
}
# codes of the label track: PRESETS, then any other combination, then seconds whose frames disagree
CLASSES = list(PRESETS) + ["other", "mixed"]
OTHER, MIXED = CLASSES.index("other"), CLASSES.index("mixed")

_locks = {}
_locks_lock = threading.Lock()


def _lock(key):
    with _locks_lock:
        return _locks.setdefault(key, threading.Lock())


def project_dir(fly):
    name = "FlyHostel_deepethogram" if round(_properties(fly)["framerate"]) == 150 else "FlyHostel_deepethogram_47fps"
    return os.path.join(DEG_ROOT, name)


@lru_cache(maxsize=4)
def _class_names(project, mtime):
    import yaml
    with open(os.path.join(project, "project_config.yaml")) as handle:
        return list(yaml.safe_load(handle)["project"]["class_names"])


def class_names(project):
    path = os.path.join(project, "project_config.yaml")
    return _class_names(project, os.path.getmtime(path))


@lru_cache(maxsize=4096)
def _local_identities(experiment, chunk, chunksize):
    """identity -> local identity of every fly in the chunk."""
    table = get_local_identities_from_experiment(experiment, chunk * chunksize)
    return {int(i): int(l) for i, l in zip(table["identity"], table["local_identity"])}


def _folder_key(experiment, chunk, local_identity):
    return f"{experiment}_{chunk:06d}_{local_identity:03d}"


def _labels_path(project, key, chunk):
    return os.path.join(project, "DATA", key, f"{chunk:06d}_labels.csv")


def _fly_label_files(fly):
    """(chunk, labels file) of every existing video folder of the fly in its DEG project."""
    experiment, identity = fly.split("__")
    props = _properties(fly)
    project = project_dir(fly)
    pattern = re.compile(rf"^{re.escape(experiment)}_(\d{{6}})_(\d{{3}})$")
    files = []
    for entry in os.scandir(os.path.join(project, "DATA")):
        match = pattern.match(entry.name)
        if not match:
            continue
        chunk, local_identity = int(match[1]), int(match[2])
        if props["first_chunk"] is not None and chunk < props["first_chunk"]:
            continue
        path = _labels_path(project, entry.name, chunk)
        if not os.path.exists(path):
            continue
        try:
            if _local_identities(experiment, chunk, props["chunksize"]).get(int(identity)) != local_identity:
                continue
        except Exception as error:   # chunk without identities
            logger.debug("No identities for %s chunk %s: %s", experiment, chunk, error)
            continue
        files.append((chunk, path))
    return sorted(files)


def _row_classes(df):
    """Class code of every row of a label file: -1 not labelled, else index in CLASSES."""
    values = df.to_numpy()
    labelled = (values != -1).any(axis=1)
    codes = np.full(len(df), OTHER, dtype=np.int64)
    present = values == 1
    columns = list(df.columns)
    for code, preset in enumerate(PRESETS.values()):
        if not all(c in columns for c in preset):
            continue
        want = np.zeros(len(columns), dtype=bool)
        want[[columns.index(c) for c in preset]] = True
        codes[(present == want).all(axis=1)] = code
    codes[~labelled] = -1
    return codes


@lru_cache(maxsize=1024)
def _file_seconds(path, mtime, chunk, chunksize, start_frame, framerate):
    """(seconds, codes) of the movie seconds the label file has labelled frames in.
    A second whose labelled frames are not all of one class (or are partly unlabelled) is MIXED."""
    rows = _row_classes(pd.read_csv(path, index_col=0))
    seconds = np.floor((chunk * chunksize + np.arange(len(rows)) - start_frame) / framerate).astype(np.int64)
    keep = seconds >= 0
    seconds, rows = seconds[keep], rows[keep]
    if not len(rows):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    unique, first = np.unique(seconds, return_index=True)
    lo = np.minimum.reduceat(rows, first)
    hi = np.maximum.reduceat(rows, first)
    labelled = hi >= 0
    codes = np.where(lo == hi, lo, MIXED)[labelled]
    return unique[labelled], codes


def label_runs(fly):
    """[[start s, end s, code], ...] of the fly's DEG labels, in movie time."""
    props = _properties(fly)
    if props["first_chunk"] is None:
        raise ValueError(f"Cannot place frames in the movie of {fly}: no feather file")
    start_frame = props["first_chunk"] * props["chunksize"]
    runs = []
    files = _fly_label_files(fly)
    for chunk, path in files:
        seconds, codes = _file_seconds(path, os.path.getmtime(path), chunk, props["chunksize"],
                                       start_frame, props["framerate"])
        if not len(seconds):
            continue
        # break where the second or the code jumps
        breaks = np.flatnonzero((np.diff(seconds) != 1) | (np.diff(codes) != 0)) + 1
        for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(seconds)]):
            run = [int(seconds[a]), int(seconds[b - 1]) + 1, int(codes[a])]
            if runs and runs[-1][1] == run[0] and runs[-1][2] == run[2]:
                runs[-1][1] = run[1]   # continues across chunks
            else:
                runs.append(run)
    return runs, len(files)


def ensure_label_file(fly, chunk):
    """Labels file of the fly's chunk, with its video folder made by deepethogram if missing.
    Returns (path, created)."""
    experiment, identity = fly.split("__")
    props = _properties(fly)
    local_identity = _local_identities(experiment, chunk, props["chunksize"]).get(int(identity))
    if local_identity is None:
        raise ValueError(f"{fly} has no local identity in chunk {chunk}")
    project = project_dir(fly)
    key = _folder_key(experiment, chunk, local_identity)
    path = _labels_path(project, key, chunk)
    if os.path.exists(path):
        return path, False

    video = os.path.join(get_basedir(experiment), "flyhostel", "single_animal",
                         f"{local_identity:03d}", f"{chunk:06d}.mp4")
    if not os.path.exists(video):
        raise FileNotFoundError(f"No single-animal video of {fly} in chunk {chunk}: {video}")
    logger.info("Adding %s to %s", video, project)
    try:
        result = subprocess.run([DEG_PYTHON, ADD_VIDEO_SCRIPT, project, video, key], stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"deepethogram took too long to add {video}")
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"deepethogram could not add {video}: {tail}")
    out = json.loads(result.stdout.strip().splitlines()[-1])
    return out["labels"], out["created"]


def write_labels(fly, start, end, behaviors):
    """Label the movie seconds [start, end) with `behaviors` (DEG columns set to 1, the
    others 0), or clear them (-1) when behaviors is None. Returns one dict per file written."""
    props = _properties(fly)
    if props["first_chunk"] is None:
        raise ValueError(f"Cannot place frames in the movie of {fly}: no feather file")
    chunksize, framerate = props["chunksize"], props["framerate"]
    origin = props["first_chunk"] * chunksize
    first = origin + int(round(start * framerate))
    last = origin + int(round(end * framerate))     # exclusive

    project = project_dir(fly)
    columns = class_names(project)
    if behaviors is not None:
        unknown = [b for b in behaviors if b not in columns]
        if unknown or not behaviors:
            raise ValueError(f"Unknown behaviors {unknown}: the project's are {columns}" if unknown else "No behavior given")

    written = []
    for chunk in range(first // chunksize, (last - 1) // chunksize + 1):
        a = max(first, chunk * chunksize) - chunk * chunksize
        b = min(last, (chunk + 1) * chunksize) - chunk * chunksize
        with _lock((fly, chunk)):
            path, created = ensure_label_file(fly, chunk)
            df = pd.read_csv(path, index_col=0)
            missing = [c for c in (behaviors or []) if c not in df.columns]
            if missing:
                raise ValueError(f"{path} has no column {missing}")
            b = min(b, len(df))
            if b <= a:
                continue
            if behaviors is None:
                df.iloc[a:b, :] = -1
            else:
                df.iloc[a:b, :] = 0
                df.iloc[a:b, [df.columns.get_loc(c) for c in behaviors]] = 1
            df.to_csv(path + ".tmp")
            os.replace(path + ".tmp", path)
        written.append({"chunk": chunk, "path": path, "created": created, "rows": [a, b]})
    _log({"time": time.time(), "fly": fly, "start": start, "end": end, "behaviors": behaviors,
          "files": [{"path": w["path"], "rows": w["rows"]} for w in written]})
    return written


def _log(entry):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError as error:
        logger.warning("Could not log DEG labels to %s: %s", LOG_PATH, error)


def register_deg_labels(app):

    def _payload(fly):
        runs, n_files = label_runs(fly)
        project = project_dir(fly)
        return {
            "project": project,
            "columns": class_names(project),
            "presets": list(PRESETS.items()),   # [name, columns], in key order (jsonify sorts dicts)
            "classes": CLASSES,
            "framerate": _properties(fly)["framerate"],
            "files": n_files,
            "runs": runs,
        }

    @app.route("/api/ethogram/<fly>/deg_labels", methods=["GET"])
    def deg_labels_get(fly):
        """The fly's DEG labels as runs of movie seconds ([start, end, index in classes])."""
        if not FLY_PATTERN.match(fly):
            return jsonify({"error": f"invalid fly id {fly}"}), 400
        try:
            return jsonify(_payload(fly))
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        except Exception as error:
            logger.exception("Could not read the DEG labels of %s", fly)
            return jsonify({"error": str(error)}), 500

    @app.route("/api/ethogram/<fly>/deg_labels", methods=["POST"])
    def deg_labels_post(fly):
        """{"start", "end" (movie seconds, end exclusive), "behaviors": [DEG columns] or null to clear}"""
        if not FLY_PATTERN.match(fly):
            return jsonify({"error": f"invalid fly id {fly}"}), 400
        body = request.get_json(silent=True) or {}
        try:
            start, end = int(body["start"]), int(body["end"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "start and end (integer seconds of movie) are required"}), 400
        behaviors = body.get("behaviors")
        if start < 0 or end <= start:
            return jsonify({"error": f"Invalid range [{start}, {end})"}), 400
        if end - start > MAX_RANGE:
            return jsonify({"error": f"At most {MAX_RANGE} s can be labelled at once"}), 400
        n = len(track(fly)["codes"])
        if end > n:
            return jsonify({"error": f"The ethogram of {fly} ends at {n} s"}), 400
        try:
            written = write_labels(fly, start, end, behaviors)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 400
        except Exception as error:
            logger.exception("Could not write the DEG labels of %s", fly)
            return jsonify({"error": str(error)}), 500
        return jsonify({**_payload(fly), "written": written})
