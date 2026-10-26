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
deepethogram (deg_add_video.py, run with $DEG_PYTHON, ~30 s each): a write that needs
one runs as a job (one item per new folder, progress in the console). Every write is
logged to $DEG_LABELS_LOG.

Undo: each write keeps, per file, the rows it replaced and those it wrote (int8, in
memory, lost when the server restarts), under the `action` id the client sends (one per
label applied; Label while moving adds its seconds to the action of the label it
paints with). POST .../deg_labels/undo restores the fly's last action, the files'
rows in reverse order of writing; it refuses when the rows are no longer what was
written (changed elsewhere, e.g. in the deepethogram GUI). An undone action can be
redone (.../deg_labels/redo) until the next write of that fly.
"""
import glob
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
from idtrackerai_validator_server.jobs import start_job

logger = logging.getLogger(__name__)

DEG_ROOT = os.path.join(os.environ.get("DEG_PROJECTS", "/flyhostel_data/fiftyone/FlyBehaviors"), "DEG")
DEG_PYTHON = os.environ.get("DEG_PYTHON", os.path.expanduser("~/mambaforge/envs/rapids-23.04/bin/python"))
ADD_VIDEO_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deg_add_video.py")
PROGRESS_MARKER = "@@PROGRESS "   # deg_add_video.MARKER (the script must not be imported here)
LOG_PATH = os.environ.get("DEG_LABELS_LOG", os.path.expanduser("~/.cache/idtrackerai-validator/deg_labels.jsonl"))
# Longest range labelled at once (s): every chunk it touches may need a new video folder (~5 s each)
MAX_RANGE = 3600

# RF class -> DEG columns set to 1, in the order of the client's keys 1, 2, ...
PRESETS = {
    "walk": ["walk"],
    "groom": ["groom"],
    "feed": ["feed"],
    "inactive+micromovement": ["inactive", "micromovement"],
    "inactive": ["inactive"],
    "inactive+pe": ["inactive", "pe"],
    "background": ["background"],
}
# codes of the label track: PRESETS, then any other combination, then seconds whose frames disagree
CLASSES = list(PRESETS) + ["other", "mixed"]
OTHER, MIXED = CLASSES.index("other"), CLASSES.index("mixed")

_locks = {}
_locks_lock = threading.Lock()

# fly -> writes that can be undone, oldest first:
# {"action", "time", "start", "end", "behaviors", "files": [(chunk, path, a, b, before, after)]}
# Undone writes go to the fly's redo stack (in order of writing), which a new write empties.
UNDO_DEPTH = 500
_undo = {}
_redo = {}
_undo_lock = threading.Lock()


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


def _chunk_label_file(fly, chunk):
    """(key, labels path, local identity) of the fly's chunk; the file may not exist yet."""
    experiment, identity = fly.split("__")
    props = _properties(fly)
    local_identity = _local_identities(experiment, chunk, props["chunksize"]).get(int(identity))
    if local_identity is None:
        raise ValueError(f"{fly} has no local identity in chunk {chunk}")
    key = _folder_key(experiment, chunk, local_identity)
    return key, _labels_path(project_dir(fly), key, chunk), local_identity


def _frame_range(fly, start, end):
    """Frames [first, last) of the movie seconds [start, end)."""
    props = _properties(fly)
    if props["first_chunk"] is None:
        raise ValueError(f"Cannot place frames in the movie of {fly}: no feather file")
    origin = props["first_chunk"] * props["chunksize"]
    return origin + int(round(start * props["framerate"])), origin + int(round(end * props["framerate"]))


def _chunks(fly, start, end):
    first, last = _frame_range(fly, start, end)
    chunksize = _properties(fly)["chunksize"]
    return range(first // chunksize, (last - 1) // chunksize + 1)


def missing_folders(fly, start, end):
    """(chunk, key) of the chunks of [start, end) whose labels file has to be made first."""
    missing = []
    for chunk in _chunks(fly, start, end):
        key, path, _ = _chunk_label_file(fly, chunk)
        if not os.path.exists(path):
            missing.append((chunk, key))
    return missing


def ensure_label_file(fly, chunk, on_progress=None):
    """Labels file of the fly's chunk, with its video folder made by deepethogram if missing.
    on_progress(fraction) is called while it is made. Returns (path, created)."""
    experiment = fly.split("__")[0]
    project = project_dir(fly)
    key, path, local_identity = _chunk_label_file(fly, chunk)
    if os.path.exists(path):
        return path, False

    video = os.path.join(get_basedir(experiment), "flyhostel", "single_animal",
                         f"{local_identity:03d}", f"{chunk:06d}.mp4")
    if not os.path.exists(video):
        raise FileNotFoundError(f"No single-animal video of {fly} in chunk {chunk}: {video}")
    logger.info("Adding %s to %s", video, project)
    process = subprocess.Popen([DEG_PYTHON, ADD_VIDEO_SCRIPT, project, video, key], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # stderr is drained by a thread so that a chatty deepethogram cannot block stdout
    stderr = []
    reader = threading.Thread(target=lambda: stderr.extend(process.stderr), daemon=True)
    reader.start()
    timer = threading.Timer(600, process.kill)
    timer.start()
    last = None
    try:
        for line in process.stdout:
            if line.startswith(PROGRESS_MARKER):
                if on_progress:
                    on_progress(float(line[len(PROGRESS_MARKER):]))
            elif line.strip():
                last = line
        process.wait()
    finally:
        timer.cancel()
    reader.join(5)
    if process.returncode != 0:
        tail = "".join(stderr[-5:]).strip()
        raise RuntimeError(f"deepethogram could not add {video}: {tail or f'exit code {process.returncode}'}")
    out = json.loads(last)
    return out["labels"], out["created"]


def write_labels(fly, start, end, behaviors, action=None):
    """Label the movie seconds [start, end) with `behaviors` (DEG columns set to 1, the
    others 0), or clear them (-1) when behaviors is None; clearing makes no new folder.
    Kept for undo under `action`. Returns one dict per file written."""
    chunksize = _properties(fly)["chunksize"]
    first, last = _frame_range(fly, start, end)

    project = project_dir(fly)
    columns = class_names(project)
    if behaviors is not None:
        unknown = [b for b in behaviors if b not in columns]
        if unknown or not behaviors:
            raise ValueError(f"Unknown behaviors {unknown}: the project's are {columns}" if unknown else "No behavior given")

    written = []
    changes = []
    for chunk in _chunks(fly, start, end):
        a = max(first, chunk * chunksize) - chunk * chunksize
        b = min(last, (chunk + 1) * chunksize) - chunk * chunksize
        with _lock((fly, chunk)):
            if behaviors is None and not os.path.exists(_chunk_label_file(fly, chunk)[1]):
                continue   # nothing labelled there
            path, created = ensure_label_file(fly, chunk)
            df = pd.read_csv(path, index_col=0)
            missing = [c for c in (behaviors or []) if c not in df.columns]
            if missing:
                raise ValueError(f"{path} has no column {missing}")
            b = min(b, len(df))
            if b <= a:
                continue
            before = df.iloc[a:b, :].to_numpy(dtype=np.int8)
            if behaviors is None:
                df.iloc[a:b, :] = -1
            else:
                df.iloc[a:b, :] = 0
                df.iloc[a:b, [df.columns.get_loc(c) for c in behaviors]] = 1
            df.to_csv(path + ".tmp")
            os.replace(path + ".tmp", path)
            changes.append((chunk, path, a, b, before, df.iloc[a:b, :].to_numpy(dtype=np.int8)))
        written.append({"chunk": chunk, "path": path, "created": created, "rows": [a, b]})
    _log({"time": time.time(), "fly": fly, "start": start, "end": end, "behaviors": behaviors,
          "files": [{"path": w["path"], "rows": w["rows"]} for w in written]})
    if changes:
        with _undo_lock:
            stack = _undo.setdefault(fly, [])
            stack.append({"action": action, "time": time.time(), "start": start, "end": end,
                          "behaviors": behaviors, "files": changes})
            del stack[:-UNDO_DEPTH]
            _redo.pop(fly, None)
    return written


def _last_action(stack):
    """Index of the first write of the stack's last action (writes without action are one each)."""
    i = len(stack) - 1
    action = stack[i]["action"]
    while action is not None and i > 0 and stack[i - 1]["action"] == action:
        i -= 1
    return i


def _first_action(stack):
    """End index of the writes of the stack's first action."""
    i = 1
    action = stack[0]["action"]
    while action is not None and i < len(stack) and stack[i]["action"] == action:
        i += 1
    return i


def _summary(writes):
    return {"writes": len(writes), "start": min(w["start"] for w in writes),
            "end": max(w["end"] for w in writes), "behaviors": writes[-1]["behaviors"],
            "time": writes[-1]["time"]}


def _n_actions(stack):
    return sum(1 for j, w in enumerate(stack)
               if w["action"] is None or j == 0 or stack[j - 1]["action"] != w["action"])


def history_info(fly):
    """What undo / redo would restore: {"writes", "start", "end", "behaviors", "time",
    "actions"} of the last action written / first action undone, None when nothing."""
    with _undo_lock:
        undo, redo = _undo.get(fly), _redo.get(fly)
        return {
            "undo": {**_summary(undo[_last_action(undo):]), "actions": _n_actions(undo)} if undo else None,
            "redo": {**_summary(redo[:_first_action(redo)]), "actions": _n_actions(redo)} if redo else None,
        }


def _apply(fly, writes, forward):
    """Write back the rows of `writes`: their `after` (redo, in order) or their `before`
    (undo, in reverse). All or nothing: refused when a file's rows are not what they
    should be before that (changed elsewhere since)."""
    files = [change for w in writes for change in w["files"]]
    locks = [_lock((fly, chunk)) for chunk in sorted({change[0] for change in files})]
    for lock in locks:
        lock.acquire()
    try:
        frames = {path: pd.read_csv(path, index_col=0) for path in {change[1] for change in files}}
        current = {path: df.to_numpy(dtype=np.int8) for path, df in frames.items()}
        for _, path, a, b, before, after in (files if forward else reversed(files)):
            expected, new = (before, after) if forward else (after, before)
            if current[path].shape[1] != expected.shape[1] or not np.array_equal(current[path][a:b], expected):
                raise RuntimeError(f"{os.path.basename(path)} changed after this label was "
                                   f"{'undone' if forward else 'written'} (rows {a}-{b}): not "
                                   f"{'redone' if forward else 'undone'}")
            current[path][a:b] = new
        for path, df in frames.items():
            df.iloc[:, :] = current[path]
            df.to_csv(path + ".tmp")
            os.replace(path + ".tmp", path)
    finally:
        for lock in reversed(locks):
            lock.release()
    return sorted(frames)


def undo_last(fly):
    """Restore the rows the fly's last action replaced; it moves to the redo stack."""
    with _undo_lock:
        stack = _undo.get(fly)
        if not stack:
            raise ValueError(f"Nothing to undo for {fly}")
        i = _last_action(stack)
        writes = stack[i:]
        paths = _apply(fly, writes, forward=False)
        del stack[i:]
        _redo[fly] = writes + _redo.get(fly, [])
    info = _summary(writes)
    _log({"time": time.time(), "fly": fly, "undo": info, "files": paths})
    return info


def redo_first(fly):
    """Write again the fly's last undone action; it goes back to the undo stack."""
    with _undo_lock:
        stack = _redo.get(fly)
        if not stack:
            raise ValueError(f"Nothing to redo for {fly}")
        i = _first_action(stack)
        writes = stack[:i]
        paths = _apply(fly, writes, forward=True)
        del stack[:i]
        _undo.setdefault(fly, []).extend(writes)
    info = _summary(writes)
    _log({"time": time.time(), "fly": fly, "redo": info, "files": paths})
    return info


def _log(entry):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError as error:
        logger.warning("Could not log DEG labels to %s: %s", LOG_PATH, error)


VIDEO_FOLDER_RE = re.compile(r"^(.+)_(\d{6})_(\d{3})$")
# labels file -> [mtime, size, has labelled frames], also kept in LABELLED_CACHE
LABELLED_CACHE = os.path.join(os.path.dirname(LOG_PATH), "deg_labelled.json")
_labelled_cache = None
_labelled_lock = threading.Lock()


def _has_labels(path):
    """Does the labels file have a frame with a value other than -1 (any 0 or 1 cell)?
    Values are only -1 / 0 / 1 and the index starts each line, so ",0" / ",1" are cells."""
    with open(path, "rb") as handle:
        handle.readline()   # header
        data = handle.read()
    return b",1" in data or b",0" in data


def labelled_videos():
    """experiment (relative folder, as in the Overview) -> number of DEG video folders
    (chunk of a fly) with labelled frames, over the projects of DEG_ROOT. Each labels
    file is read once per (mtime, size) (~1000 files, ~1 GB: ~20 s cold, ~0.3 s after)."""
    global _labelled_cache
    from concurrent.futures import ThreadPoolExecutor
    from idtrackerai_validator_server.experiments import to_relpath

    files = []
    for data in sorted(glob.glob(os.path.join(DEG_ROOT, "*", "DATA"))):
        for entry in os.scandir(data):
            match = VIDEO_FOLDER_RE.match(entry.name)
            if not match or not entry.is_dir():
                continue
            path = _labels_path(os.path.dirname(data), entry.name, int(match[2]))
            try:
                stat = os.stat(path)
            except OSError:
                continue
            files.append((match[1], path, (stat.st_mtime, stat.st_size)))

    changed = []

    def check(item):
        _, path, (mtime, size) = item
        cached = _labelled_cache.get(path)
        if cached and cached[:2] == [mtime, size]:
            return cached[2]
        try:
            value = _has_labels(path)
        except OSError:
            return False
        _labelled_cache[path] = [mtime, size, value]
        changed.append(path)
        return value

    with _labelled_lock:
        if _labelled_cache is None:
            try:
                with open(LABELLED_CACHE) as handle:
                    _labelled_cache = json.load(handle)
            except (OSError, ValueError):
                _labelled_cache = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            flags = list(pool.map(check, files))
        if changed:
            try:
                os.makedirs(os.path.dirname(LABELLED_CACHE), exist_ok=True)
                tmp = LABELLED_CACHE + ".tmp"
                with open(tmp, "w") as handle:
                    json.dump(_labelled_cache, handle)
                os.replace(tmp, LABELLED_CACHE)
            except OSError as error:
                logger.warning("Could not save %s: %s", LABELLED_CACHE, error)
    counts = {}
    for (experiment, _, _), labelled in zip(files, flags):
        if not labelled:
            continue
        try:
            relpath = to_relpath(experiment)
        except Exception:   # not an experiment name
            continue
        counts[relpath] = counts.get(relpath, 0) + 1
    return counts


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
            **history_info(fly),   # undo, redo
        }

    def _write_job(fly, start, end, behaviors, missing, action):
        """Make the missing folders (one job item each, ~30 s), then write: 202 + the job,
        whose result is the response of a write without new folders."""
        def work(job):
            for i, (chunk, key) in enumerate(missing):
                job.update(i, status="running")
                with _lock((fly, chunk)):
                    ensure_label_file(fly, chunk, on_progress=lambda p, i=i: job.update(i, progress=p))
                job.update(i, status="done", progress=1.0)
            written = write_labels(fly, start, end, behaviors, action)
            job.result = {**_payload(fly), "written": written}

        names = [f"DEG video {key}" for _, key in missing]
        job = start_job(f"{fly}: DEG video{'s' if len(missing) > 1 else ''} for labelling", names, work)
        return jsonify(job.to_dict()), 202

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
        action = body.get("action")
        action = str(action) if action is not None else None
        if start < 0 or end <= start:
            return jsonify({"error": f"Invalid range [{start}, {end})"}), 400
        if end - start > MAX_RANGE:
            return jsonify({"error": f"At most {MAX_RANGE} s can be labelled at once"}), 400
        n = len(track(fly)["codes"])
        if end > n:
            return jsonify({"error": f"The ethogram of {fly} ends at {n} s"}), 400
        try:
            missing = missing_folders(fly, start, end) if behaviors is not None else []
            if missing:
                return _write_job(fly, start, end, behaviors, missing, action)
            written = write_labels(fly, start, end, behaviors, action)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 400
        except Exception as error:
            logger.exception("Could not write the DEG labels of %s", fly)
            return jsonify({"error": str(error)}), 500
        return jsonify({**_payload(fly), "written": written})

    @app.route("/api/ethogram/<fly>/deg_labels/<step>", methods=["POST"])
    def deg_labels_history(fly, step):
        """undo / redo the fly's last labelling action; returns the labels plus `undone` / `redone`."""
        if not FLY_PATTERN.match(fly):
            return jsonify({"error": f"invalid fly id {fly}"}), 400
        if step not in ("undo", "redo"):
            return jsonify({"error": f"unknown step {step}"}), 404
        try:
            done = undo_last(fly) if step == "undo" else redo_first(fly)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except RuntimeError as error:
            return jsonify({"error": str(error)}), 409
        except Exception as error:
            logger.exception("Could not %s the DEG labels of %s", step, fly)
            return jsonify({"error": str(error)}), 500
        return jsonify({**_payload(fly), {"undo": "undone", "redo": "redone"}[step]: done})
