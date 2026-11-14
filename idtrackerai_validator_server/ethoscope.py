"""Ethoscope tab: sleep of the loaded experiment's flies in the ethoscope they were in.

    GET /api/ethoscope/flies   the flies (ethoscope_metadata rows) and the ethoscope run
                               of each (`available` = at least one fly has one)
    GET /api/ethoscope/sleep   per fly the sleep trace scored from its ROI

The metadata sheets give each fly `machine_name`, `date` and `region_id` (its ROI) of
the ethoscope it lived in, plus `reference_hour` (ZT0). As scopr's `build_query`, the
run is the last one started on that date by that machine:
`$ETHOSCOPE_RESULTS/<machine_id>/<machine_name>/<date>_<time>/<date>_<time>_<machine_id>.db`.

Sleep is scored as FSLRetho (scopr's `read_single_roi`, sleepr's `sleep_annotation` with
`max_velocity_detector`), ported to numpy:
  * rows with `is_inferred` are dropped unless `has_interacted`; t (ms) is moved to ZT
    (`ms_after_ref`: hours between ZT0 and the run start, mod 24, from METADATA date_time,
    UTC), so t = seconds since ZT0 of the day the run started;
  * windows of `time_window_length` s; 60 s windows with ≤ 20 rows are dropped (sparse);
  * velocity = 10^(xy_dist_log10x1000 / 1000) / dt, set to 0 within `masking_duration` s
    after an interaction; corrected velocity = velocity · dt / `velocity_correction_coef`;
    a window is moving when the max corrected velocity of its rows but the first is > 1;
  * windows without data count as not moving; asleep = runs of not moving lasting at
    least `min_time_immobile` s.
FSLRetho does not know when the flies left the ethoscope; here rows recorded after the fly
went into the FlyHostel (`moved_at`) are dropped, and so are the windows after its last
movement (an empty tube scores as an immobile fly; sleepr's `curate_dead_animals` cuts there too).
Defaults are FSLRetho's sliders (coef 0.0048, 300 s, 10 s windows, 6 s masking).
Traces are cached per file, mtime, ROI, reference hour and parameters.

A run missing here is looked for on $ETHOSCOPE_REMOTE_HOST (cv1, same layout under
$ETHOSCOPE_REMOTE_RESULTS); POST /api/ethoscope/download copies it here (job).
"""
import calendar
import glob
import logging
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
from flask import jsonify, request

from idtrackerai_validator_server.jobs import start_job

logger = logging.getLogger(__name__)

RESULTS_DIR = os.environ.get("ETHOSCOPE_RESULTS", "/ethoscope_data/results")
DEFAULTS = {"velocity_correction_coef": 0.0048, "min_time_immobile": 300.0,
            "time_window_length": 10.0, "masking_duration": 6.0}
LIMITS = {"velocity_correction_coef": (1e-4, 0.05), "min_time_immobile": (10, 3600),
          "time_window_length": (1, 300), "masking_duration": (0, 600)}
SPARSE_WINDOW, SPARSE_MIN = 60, 20          # sleepr's curate_sparse_roi_data
RUN_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$")
DATE_TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[_ ](\d{2})[-:](\d{2})(?:[-:](\d{2}))?")
FLY_COLUMNS = ("genotype", "sex", "light_cycle", "age_min", "incubator")
# runs missing here may be on the computer the ethoscopes save to (cv1): listed over ssh and
# copied with rsync into RESULTS_DIR, at the same <machine_id>/<machine_name>/<run>/ path
REMOTE_HOST = os.environ.get("ETHOSCOPE_REMOTE_HOST", "cv1")
REMOTE_RESULTS = os.environ.get("ETHOSCOPE_REMOTE_RESULTS", "/ethoscope_data/results")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
REMOTE_TTL = 300                  # s a remote listing is kept
RECORDING_S = 600                 # a remote db modified this recently is still being written
DOWNLOAD_MARGIN = 5 * 1024 ** 3   # bytes kept free on the results drive
SAFE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

_cache = {}
_cache_lock = Lock()
CACHE_SIZE = 256


def _clean(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.upper() in ("NONE", "NAN", "NA", "NULL"):
        return None
    return s


def _number(value):
    try:
        v = float(_clean(value))
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def find_run(machine_name, date):
    """Last run of `machine_name` started on `date` (YYYY-MM-DD): (path, start 'date_time') or None."""
    if not machine_name or not date:
        return None
    date = date[:10]
    runs = []
    for folder in glob.glob(os.path.join(RESULTS_DIR, "*", glob.escape(machine_name), f"{date}_*")):
        name = os.path.basename(folder)
        if not RUN_RE.match(name):
            continue
        dbs = sorted(glob.glob(os.path.join(folder, "*.db")))
        if dbs:
            runs.append((name, dbs[-1]))
    if not runs:
        return None
    name, path = sorted(runs)[-1]
    return path, name


_remote_cache = {}                # (machine, date) -> (time listed, run dict or None)
_remote_lock = Lock()


def find_remote_runs(keys):
    """Last run of each (machine_name, date) on REMOTE_HOST, one ssh call for all keys not
    listed in the last REMOTE_TTL s: {key: {"run", "relpath" (of the db, relative to
    REMOTE_RESULTS), "size", "mtime", "recording"} or None}. Raises RuntimeError when the
    host cannot be reached."""
    now = time.time()
    out, todo = {}, []
    with _remote_lock:
        for key in keys:
            hit = _remote_cache.get(key)
            if hit and now - hit[0] < REMOTE_TTL:
                out[key] = hit[1]
            else:
                todo.append(key)
    todo = [k for k in todo if SAFE_RE.match(k[0]) and re.match(r"^\d{4}-\d{2}-\d{2}$", k[1])]
    if not todo:
        return out
    root = shlex.quote(REMOTE_RESULTS)
    script = f"cd {root} || exit 3\n" + "".join(
        f"stat -c '%n\t%s\t%Y' */{m}/{d}_*/*.db 2>/dev/null\n" for m, d in todo) + "date +%s\n"
    try:
        result = subprocess.run(SSH + [REMOTE_HOST, "bash -s"], input=script, capture_output=True,
                                text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{REMOTE_HOST} did not answer within 30 s")
    if result.returncode != 0:
        raise RuntimeError(f"ssh {REMOTE_HOST} failed ({result.returncode}): {result.stderr.strip()[-300:]}")
    lines = result.stdout.strip().splitlines()
    remote_now = float(lines.pop()) if lines and lines[-1].isdigit() else now
    found = {k: None for k in todo}
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        relpath, size, mtime = parts
        bits = relpath.split("/")
        if len(bits) != 4 or not RUN_RE.match(bits[2]):
            continue
        key = (bits[1], bits[2][:10])
        if key not in found:
            continue
        run = {"run": bits[2], "relpath": relpath, "size": int(size), "mtime": float(mtime),
               "recording": remote_now - float(mtime) < RECORDING_S}
        if found[key] is None or (run["run"], run["relpath"]) > (found[key]["run"], found[key]["relpath"]):
            found[key] = run
    with _remote_lock:
        for key, run in found.items():
            _remote_cache[key] = (now, run)
    out.update(found)
    return out


def _connect(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)


def _unix(date_time):
    """'YYYY-MM-DD_HH-MM-SS' (folder names, UTC) -> unix s, or None."""
    m = DATE_TIME_RE.search(date_time or "")
    if not m:
        return None
    y, mo, d = map(int, m.group(1).split("-"))
    return calendar.timegm((y, mo, d, int(m.group(2)), int(m.group(3)), int(m.group(4) or 0)))


def fly_entries(rows, experiment=None):
    """One entry per metadata row: identity, metadata, its ethoscope run (or why none) and
    `moved_at`: when it went into the FlyHostel (flyhostel_date + flyhostel_time, else the
    experiment folder's date), after which its ethoscope ROI is an empty tube."""
    default_move = _unix(experiment)
    flies = []
    for i, row in enumerate(rows):
        identity = _number(row.get("identity"))
        identity = int(identity) if identity is not None else (i + 1)
        machine = _clean(row.get("machine_name"))
        date = _clean(row.get("date"))
        region = _number(row.get("region_id"))
        ref = _number(row.get("reference_hour"))
        fly = {"identity": identity, "machine_name": machine, "date": date,
               "region_id": int(region) if region is not None else None,
               "reference_hour": ref, "path": None, "run": None, "error": None,
               "moved_at": _unix(f"{_clean(row.get('flyhostel_date')) or ''}_{_clean(row.get('flyhostel_time')) or ''}") or default_move}
        for c in FLY_COLUMNS:
            fly[c] = _clean(row.get(c))
        if not machine or not date or region is None:
            fly["error"] = "no machine_name / date / region_id in the metadata"
        else:
            found = find_run(machine, date)
            if found is None:
                fly["error"] = f"no run of {machine} on {date[:10]} in {RESULTS_DIR}"
            else:
                fly["path"], fly["run"] = found
        flies.append(fly)
    flies.sort(key=lambda f: f["identity"])
    return flies


def add_remote(flies):
    """Flies without a local run get `remote`: their run on REMOTE_HOST (or None). Returns the
    error of the remote listing, if any."""
    want = sorted({(f["machine_name"], f["date"][:10]) for f in flies
                   if not f["path"] and f["machine_name"] and f["date"] and f["region_id"] is not None})
    if not want:
        return None
    try:
        remote = find_remote_runs(want)
    except Exception as error:
        logger.warning("ethoscope runs on %s: %s", REMOTE_HOST, error)
        return str(error)
    for f in flies:
        if not f["path"] and f["machine_name"] and f["date"]:
            f["remote"] = remote.get((f["machine_name"], f["date"][:10]))
            if f["remote"]:
                f["error"] = f"run {f['remote']['run']} only on {REMOTE_HOST}"
    return None


_downloading = set()
_download_lock = Lock()


def _rsync(job, index, relpath):
    """Copy the run folder of `relpath` (db and its .txt log) from REMOTE_HOST into RESULTS_DIR.
    Partial files stay in .rsync-partial/ so a broken copy never looks like a run."""
    folder = os.path.dirname(relpath)
    cmd = ["rsync", "-a", "--relative", "--partial-dir=.rsync-partial", "--info=progress2",
           "--no-inc-recursive", "-e", " ".join(SSH),
           f"{REMOTE_HOST}:{REMOTE_RESULTS}/./{folder}/", f"{RESULTS_DIR}/"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b""
    while True:
        data = process.stdout.read(4096)
        if not data:
            break
        buffer += data
        parts = re.split(rb"[\r\n]", buffer)
        buffer = parts.pop()
        for part in parts:
            match = re.search(rb"\s(\d+)%\s", part)
            if match:
                job.update(index, progress=min(int(match.group(1)) / 100, 0.999))
    stderr = process.stderr.read().decode(errors="replace")
    if process.wait() != 0:
        raise RuntimeError(f"rsync exited with {process.returncode}: {stderr.strip()[-600:]}")


def start_download(flies):
    """Job copying the remote runs of `flies` (one item per run). Raises ValueError (nothing to
    copy / no room) or RuntimeError (already copying)."""
    runs = {}
    for f in flies:
        r = f.get("remote")
        if r and not f["path"]:
            runs[r["relpath"]] = r
    if not runs:
        raise ValueError(f"No run of these flies is waiting on {REMOTE_HOST}")
    need = sum(r["size"] for r in runs.values())
    os.makedirs(RESULTS_DIR, exist_ok=True)
    free = shutil.disk_usage(RESULTS_DIR).free
    if free - need < DOWNLOAD_MARGIN:
        raise ValueError(f"{RESULTS_DIR} has {free / 1e9:.1f} GB free, {need / 1e9:.1f} GB needed "
                         f"plus {DOWNLOAD_MARGIN / 1e9:.0f} GB margin")
    with _download_lock:
        busy = _downloading & set(runs)
        if busy:
            raise RuntimeError(f"Already copying {', '.join(sorted(busy))}")
        _downloading.update(runs)
    relpaths = sorted(runs)

    def work(job):
        failed = []
        try:
            for i, relpath in enumerate(relpaths):
                job.update(i, status="running")
                try:
                    _rsync(job, i, relpath)
                    if not os.path.isfile(os.path.join(RESULTS_DIR, relpath)):
                        raise RuntimeError(f"{relpath} missing after the copy")
                    job.update(i, status="done", progress=1.0)
                except Exception as error:
                    logger.exception("copying %s from %s", relpath, REMOTE_HOST)
                    job.update(i, status="failed", error=str(error))
                    failed.append(relpath)
        finally:
            with _download_lock:
                _downloading.difference_update(relpaths)
            with _remote_lock:
                _remote_cache.clear()
        job.result = {"copied": [p for p in relpaths if p not in failed], "failed": failed}
        if failed:
            raise RuntimeError(f"{len(failed)} of {len(relpaths)} run(s) not copied")

    return start_job(f"Copy {len(relpaths)} ethoscope run(s) from {REMOTE_HOST} ({need / 1e6:.0f} MB)",
                     [os.path.basename(p) for p in relpaths], work)


def _run_info(path):
    con = _connect(path)
    try:
        meta = dict(con.execute("SELECT field, value FROM METADATA").fetchall())
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()
    return float(meta["date_time"]), tables


def _ms_after_ref(start_unix, reference_hour):
    """scopr's get_ms_after_ref: ms between ZT0 and the run start (UTC hour of day, mod 24)."""
    sod = int(start_unix) % 86400      # scopr formats the start with whole seconds
    return ((sod / 3600.0 - reference_hour) % 24) * 3600 * 1000


def score_sleep(t, xy, interacted, params):
    """sleepr's max_velocity_detector + sleep_annotation on one ROI (t in s, ZT aligned).
    Returns (t of the first window, moving and asleep per window, interpolated per window)."""
    tw = params["time_window_length"]
    # curate_sparse_roi_data: drop 60 s windows with <= 20 rows
    tw60 = np.floor(t / SPARSE_WINDOW)
    _, inverse, counts = np.unique(tw60, return_inverse=True, return_counts=True)
    keep = counts[inverse] > SPARSE_MIN
    t, xy, interacted = t[keep], xy[keep], interacted[keep]
    if len(t) < 2:
        return None
    dt = np.empty_like(t)
    dt[0] = np.nan
    dt[1:] = np.diff(t)
    dist = np.power(10.0, xy / 1000.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = dist / dt
    # masking after interactions (interaction_id = cumsum(has_interacted))
    if params["masking_duration"] > 0 and interacted.any():
        iid = np.cumsum(interacted)
        first = np.r_[0, np.flatnonzero(np.diff(iid)) + 1]
        block_t0 = np.repeat(t[first], np.diff(np.r_[first, len(t)]))
        masked = (t < block_t0 + params["masking_duration"]) & (iid != 0)
        velocity[masked] = 0
    with np.errstate(invalid="ignore"):
        corrected = velocity * dt / params["velocity_correction_coef"]
    # max over the rows of each window but its first; as in R, a window with one row or a
    # row without velocity (dt = 0) has none (NA): it is left out, like a window without data
    win = np.floor(t / tw).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(win)) + 1]
    corrected[starts] = -np.inf
    bad = ~(np.isfinite(corrected) | (corrected == -np.inf))
    bad[starts] = False
    corrected[bad] = -np.inf
    max_v = np.maximum.reduceat(corrected, starts)
    sizes = np.diff(np.r_[starts, len(t)])
    valid = (sizes > 1) & ~np.logical_or.reduceat(bad, starts)
    windows = win[starts][valid]
    moving_w = max_v[valid] > 1
    # regular grid of windows: missing ones are not moving (interpolated)
    w0 = int(windows[0])
    n = int(windows[-1]) - w0 + 1
    moving = np.zeros(n, dtype=bool)
    interpolated = np.ones(n, dtype=bool)
    moving[windows - w0] = moving_w
    interpolated[windows - w0] = False
    # sleep_contiguous: runs of not moving lasting >= min_time_immobile
    min_len = params["min_time_immobile"] / tw
    asleep = np.zeros(n, dtype=bool)
    still = ~moving
    edges = np.flatnonzero(np.diff(np.r_[0, still.astype(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= min_len:
            asleep[a:b] = True
    return w0 * tw, moving, asleep, interpolated


def _fly_trace(fly, params):
    path, region = fly["path"], fly["region_id"]
    ref = fly["reference_hour"]
    moved_at = fly.get("moved_at")
    key = (path, os.path.getmtime(path), region, ref, moved_at, tuple(sorted(params.items())))
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    start_unix, tables = _run_info(path)
    table = f"ROI_{region}"
    if table not in tables:
        raise ValueError(f"{table} not in {os.path.basename(path)}")
    con = _connect(path)
    try:
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        sel = ["t", "xy_dist_log10x1000",
               "has_interacted" if "has_interacted" in cols else "0",
               "is_inferred" if "is_inferred" in cols else "0"]
        # rows after the fly went into the FlyHostel are an empty tube
        limit = "" if moved_at is None else f" WHERE t < {(moved_at - start_unix) * 1000:.0f}"
        data = np.array(con.execute(f"SELECT {', '.join(sel)} FROM {table}{limit} ORDER BY t").fetchall(),
                        dtype=np.float64).reshape(-1, 4)
        last_t = con.execute(f"SELECT MAX(t) FROM {table}").fetchone()[0]
    finally:
        con.close()
    keep = (data[:, 3] == 0) | (data[:, 2] != 0)
    data = data[keep]
    offset = _ms_after_ref(start_unix, ref) if ref is not None else 0.0
    t = (data[:, 0] + offset) / 1000.0
    scored = score_sleep(t, data[:, 1], data[:, 2], params) if len(t) >= 100 else None
    if scored is None:
        result = {"t0": None, "asleep": "", "moving": "", "interpolated": 0, "rows": int(len(t))}
    else:
        t0, moving, asleep, interpolated = scored
        # trailing immobility: the fly was taken out (curate_dead_animals keeps up to the last movement)
        last = int(np.flatnonzero(moving)[-1]) + 1 if moving.any() else len(moving)
        trimmed_tail = (len(moving) - last) * params["time_window_length"]
        moving, asleep, interpolated = moving[:last], asleep[:last], interpolated[:last]
        # one character per window: 1 asleep, 0 awake (moving or immobile < min_time_immobile), - no data
        chars = np.where(asleep, "1", "0").astype("<U1")
        chars[interpolated & ~asleep] = "-"
        result = {"t0": float(t0), "asleep": "".join(chars.tolist()),
                  "moving": "".join(np.where(moving, "1", "0").tolist()),
                  "interpolated": int(interpolated.sum()), "rows": int(len(t)),
                  "trimmed_tail": float(trimmed_tail)}
    run_end = start_unix + (last_t or 0) / 1000
    result.update(start=start_unix, zt_aligned=ref is not None,
                  cut_after_move=float(max(0.0, run_end - moved_at)) if moved_at is not None else 0.0)
    with _cache_lock:
        if len(_cache) >= CACHE_SIZE:
            _cache.pop(next(iter(_cache)))
        _cache[key] = result
    return result


def _params():
    params = {}
    for k, default in DEFAULTS.items():
        try:
            v = float(request.args.get(k, default))
        except ValueError:
            v = default
        lo, hi = LIMITS[k]
        params[k] = min(max(v, lo), hi)
    return params


def register_ethoscope(app, get_rows):
    """get_rows() -> (experiment, list of metadata row dicts) of the loaded experiment, or None."""

    def _flies():
        state = get_rows()
        if state is None:
            return None, []
        experiment, rows = state
        return experiment, fly_entries(rows or [], experiment)

    @app.route("/api/ethoscope/flies", methods=["GET"])
    def ethoscope_flies():
        experiment, flies = _flies()
        remote_error = add_remote(flies)
        return jsonify({"experiment": experiment, "results_dir": RESULTS_DIR,
                        "available": any(f["path"] for f in flies),
                        "remote_host": REMOTE_HOST, "remote_error": remote_error,
                        "downloadable": any(f.get("remote") for f in flies),
                        "flies": flies, "defaults": DEFAULTS})

    @app.route("/api/ethoscope/download", methods=["POST"])
    def ethoscope_download():
        experiment, flies = _flies()
        body = request.get_json(silent=True) or {}
        if body.get("experiment") and body["experiment"] != experiment:
            return jsonify({"error": f"{body['experiment']} is no longer the loaded experiment"}), 409
        error = add_remote(flies)
        if error:
            return jsonify({"error": error}), 502
        try:
            job = start_download(flies)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 409
        return jsonify(job.to_dict()), 202

    @app.route("/api/ethoscope/sleep", methods=["GET"])
    def ethoscope_sleep():
        experiment, flies = _flies()
        params = _params()
        todo = [f for f in flies if f["path"]]

        def one(fly):
            try:
                return {**fly, "trace": _fly_trace(fly, params)}
            except Exception as error:
                logger.exception("ethoscope sleep of %s fly %s", experiment, fly["identity"])
                return {**fly, "trace": None, "error": str(error)}

        with ThreadPoolExecutor(max_workers=8) as pool:
            done = {f["identity"]: r for f, r in zip(todo, pool.map(one, todo))}
        out = [done.get(f["identity"], {**f, "trace": None}) for f in flies]
        return jsonify({"experiment": experiment, "params": params, "flies": out})
