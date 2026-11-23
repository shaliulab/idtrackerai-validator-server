"""Ethoscope tab: sleep of the loaded experiment's flies in the ethoscope they were in.

    GET /api/ethoscope/flies   the flies (ethoscope_metadata rows) and the ethoscope run
                               of each (`available` = at least one fly has one)
    GET /api/ethoscope/sleep   per fly the sleep trace scored from its ROI
    GET /api/ethoscope/snapshots, /api/ethoscope/snapshot?fly=<key>&id=
                               the run's IMG_SNAPSHOTS (whole frame every ~5 min) on the
                               trace's clock, and one of them cropped to the fly's ROI

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
from flask import Response, jsonify, request

from idtrackerai_validator_server.jobs import start_job
from idtrackerai_validator_server import lab_paths

logger = logging.getLogger(__name__)

RESULTS_DIR = lab_paths.ETHOSCOPE_RESULTS
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
REMOTE_HOST = lab_paths.ETHOSCOPE_REMOTE_HOST
REMOTE_RESULTS = lab_paths.ETHOSCOPE_REMOTE_RESULTS
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
        fly = {"identity": identity, "key": str(identity), "label": f"Fly {identity}", "in_experiment": True,
               "machine_name": machine, "date": date,
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
    if not len(windows):          # too few rows (an empty tube): no window with a velocity
        return None
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


# --- snapshots: IMG_SNAPSHOTS (id, t ms since the run start, img JPEG of the whole frame, every ~5 min) ---
SNAPSHOT_PAD = 14            # px around the ROI in a crop
_snap_cache = {}             # path -> ((mtime, size), [(rowid, t ms)], {roi_idx: (x, y, w, h)}, (width, height))
_snap_lock = Lock()


def _snapshot_index(path):
    """Snapshots of a run (rowid, t), its ROIs and frame size; cached per file mtime / size (listing t
    reads every blob page: ~4 s cold for 2000 snapshots)."""
    stat = os.stat(path)
    key = (stat.st_mtime, stat.st_size)
    with _snap_lock:
        cached = _snap_cache.get(path)
    if cached and cached[0] == key:
        return cached[1:]
    con = _connect(path)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        snaps = con.execute("SELECT rowid, t FROM IMG_SNAPSHOTS ORDER BY t").fetchall() if "IMG_SNAPSHOTS" in tables else []
        rois = {int(r[0]): tuple(int(v) for v in r[1:]) for r in con.execute("SELECT roi_idx, x, y, w, h FROM ROI_MAP")} \
            if "ROI_MAP" in tables else {}
        meta = dict(con.execute("SELECT field, value FROM METADATA").fetchall())
    finally:
        con.close()
    size = (_number(meta.get("frame_width")), _number(meta.get("frame_height")))
    with _snap_lock:
        if len(_snap_cache) >= 64:
            _snap_cache.pop(next(iter(_snap_cache)))
        _snap_cache[path] = (key, snaps, rois, size)
    return snaps, rois, size


def fly_snapshots(fly):
    """The snapshots of a fly's run on the trace's clock: `snapshots` [[id, t]] (t = s since ZT0 of the
    day the run started, as the trace, or since the start without reference_hour), its ROI, the frame
    size and `moved_t` (when it went into the FlyHostel, same clock: later snapshots show an empty tube)."""
    snaps, rois, size = _snapshot_index(fly["path"])
    start_unix, _ = _run_info(fly["path"])
    ref = fly["reference_hour"]
    offset = _ms_after_ref(start_unix, ref) if ref is not None else 0.0
    moved = fly.get("moved_at")
    return {"snapshots": [[rowid, round((t + offset) / 1000.0, 1)] for rowid, t in snaps],
            "roi": rois.get(fly["region_id"]), "frame": size,
            "moved_t": None if moved is None else round(((moved - start_unix) * 1000 + offset) / 1000.0, 1)}


def snapshot_jpeg(path, rowid, roi=None, crop=True, width=None):
    """JPEG of one snapshot: cropped to the ROI (± SNAPSHOT_PAD), or the whole frame with the ROI outlined;
    scaled down to `width` px."""
    import cv2
    con = _connect(path)
    try:
        row = con.execute("SELECT img FROM IMG_SNAPSHOTS WHERE rowid = ?", (rowid,)).fetchone()
    finally:
        con.close()
    if row is None:
        raise KeyError(rowid)
    if roi is None and width is None:
        return bytes(row[0])
    img = cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_COLOR)
    if roi is not None:
        x, y, w, h = roi
        if crop:
            p = SNAPSHOT_PAD
            img = img[max(0, y - p):y + h + p, max(0, x - p):x + w + p]
        else:
            cv2.rectangle(img, (x - 3, y - 3), (x + w + 3, y + h + 3), (40, 120, 230), 3)
    if width and img.shape[1] > width:
        img = cv2.resize(img, (int(width), max(1, round(img.shape[0] * width / img.shape[1]))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


# --- other flies of the runs: the whole ethoscope run(s), or every run started the same date (a batch) ---
SCOPES = ("experiment", "run", "batch")
_sheet_cache = {"key": None, "index": {}}
_progress = {}                # token of a /sleep request -> {done, total, t}
_progress_lock = Lock()
_sheet_lock = Lock()


def sheet_index():
    """{(machine_name, date YYYY-MM-DD, region_id): sheet row} from the latest export of every metadata
    sheet (overview.sheet_files; the sheets also list flies that stayed in the ethoscopes); cached per
    file mtime. Empty when the sheets cannot be read."""
    import csv
    try:
        from idtrackerai_validator_server.overview import sheet_files
        files = sorted(sheet_files()[0].values())
        key = tuple((p, os.path.getmtime(p)) for p in files)
    except Exception as error:
        logger.warning("ethoscope: cannot list the metadata sheets: %s", error)
        return {}
    with _sheet_lock:
        if _sheet_cache["key"] == key:
            return _sheet_cache["index"]
    index = {}
    for path in files:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                machine, date, region = _clean(row.get("machine_name")), _clean(row.get("date")), _number(row.get("region_id"))
                if machine and date and region is not None:
                    index.setdefault((machine, date[:10], int(region)), row)
    with _sheet_lock:
        _sheet_cache.update(key=key, index=index)
    return index


def runs_on(date):
    """{path: (machine_name, run)}: the last run of every machine started on `date` (as find_run)"""
    machines = {}
    for folder in glob.glob(os.path.join(RESULTS_DIR, "*", "*", f"{glob.escape(date[:10])}_*")):
        machine = os.path.basename(os.path.dirname(folder))
        if RUN_RE.match(os.path.basename(folder)):
            machines.setdefault(machine, []).append(folder)
    out = {}
    for machine in machines:
        found = find_run(machine, date)
        if found:
            out[found[0]] = (machine, found[1])
    return out


def _run_rois(path):
    con = _connect(path)
    try:
        return sorted(int(r[0]) for r in con.execute("SELECT roi_idx FROM ROI_MAP"))
    finally:
        con.close()


def extra_fly(machine, run, path, region, row, default_ref):
    """A fly of a run that is not in the loaded experiment: its sheet row (if any) gives genotype, sex,
    reference_hour and when it went into the FlyHostel."""
    row = row or {}
    ref = _number(row.get("reference_hour"))
    moved = _unix(f"{_clean(row.get('flyhostel_date')) or ''}_{_clean(row.get('flyhostel_time')) or ''}")
    basedir = _clean(row.get("basedir"))
    m = re.search(r"(FlyHostel\d+/\d+X/\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", basedir or "")
    fly = {"identity": None, "key": f"{machine}/{run}/{region}", "in_experiment": False,
           "label": f"{machine.replace('ETHOSCOPE_', 'E')} ROI {region}",
           "machine_name": machine, "date": run[:10], "region_id": region,
           "reference_hour": ref if ref is not None else default_ref, "path": path, "run": run, "error": None,
           "moved_at": moved, "experiment": m.group(1) if m else None, "in_sheet": bool(row)}
    for c in FLY_COLUMNS:
        fly[c] = _clean(row.get(c))
    return fly


def scope_flies(flies, scope):
    """The experiment's flies plus, for scope `run`, the other ROIs of their runs, for `batch` every ROI of
    every run started on the same dates."""
    if scope not in ("run", "batch"):
        return flies
    runs = {f["path"]: (f["machine_name"], f["run"]) for f in flies if f["path"]}
    if scope == "batch":
        for date in sorted({run[:10] for _, run in runs.values()}):
            for path, value in runs_on(date).items():
                runs.setdefault(path, value)
    refs = [f["reference_hour"] for f in flies if f["reference_hour"] is not None]
    default_ref = max(set(refs), key=refs.count) if refs else None
    taken = {(f["path"], f["region_id"]) for f in flies}
    sheet = sheet_index()
    extra = []
    for path, (machine, run) in sorted(runs.items(), key=lambda kv: kv[1]):
        try:
            rois = _run_rois(path)
        except sqlite3.Error as error:
            logger.warning("ethoscope: ROI_MAP of %s: %s", path, error)
            continue
        for region in rois:
            if (path, region) not in taken:
                extra.append(extra_fly(machine, run, path, region, sheet.get((machine, run[:10], region)), default_ref))
    return flies + extra


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

    def _flies(scope="experiment"):
        state = get_rows()
        if state is None:
            return None, []
        experiment, rows = state
        return experiment, scope_flies(fly_entries(rows or [], experiment), scope)

    def _scope():
        scope = request.args.get("scope", "experiment")
        return scope if scope in SCOPES else "experiment"

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

    @app.route("/api/ethoscope/snapshots", methods=["GET"])
    def ethoscope_snapshots():
        """Per fly with a run (key -> fly_snapshots, ?scope= as /sleep): its snapshots on the trace's clock;
        one listing per run."""
        experiment, flies = _flies(_scope())
        todo = [f for f in flies if f["path"]]

        def one(fly):
            try:
                return fly["key"], fly_snapshots(fly)
            except Exception as error:
                logger.exception("ethoscope snapshots of %s fly %s", experiment, fly["key"])
                return fly["key"], {"snapshots": [], "error": str(error)}

        # flies of one run share the listing: list each path once first
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda p: _snapshot_index(p) if os.path.exists(p) else None, sorted({f["path"] for f in todo})))
            out = dict(pool.map(one, todo))
        return jsonify({"experiment": experiment, "flies": out})

    @app.route("/api/ethoscope/snapshot", methods=["GET"])
    def ethoscope_snapshot():
        """JPEG of snapshot `id` of fly `fly` (its key: an identity of the experiment, or
        <machine_name>/<run>/<roi>): ?crop=roi (default: its ROI) or full (ROI outlined), &w= px"""
        key = request.args.get("fly", "")
        try:
            rowid = int(request.args["id"])
        except (KeyError, ValueError):
            return jsonify({"error": "id required"}), 400
        if "/" in key:
            parts = key.split("/")
            if len(parts) != 3 or not all(SAFE_RE.match(p) for p in parts) or not parts[2].isdigit():
                return jsonify({"error": f"bad fly {key}"}), 400
            dbs = sorted(glob.glob(os.path.join(RESULTS_DIR, "*", parts[0], parts[1], "*.db")))
            fly = {"path": dbs[-1], "region_id": int(parts[2])} if dbs else None
        else:
            _, flies = _flies()
            fly = next((f for f in flies if f["key"] == key and f["path"]), None)
        if fly is None:
            return jsonify({"error": f"fly {key} has no ethoscope run"}), 404
        try:
            width = int(request.args["w"]) if request.args.get("w") else None
        except ValueError:
            return jsonify({"error": "w must be an integer"}), 400
        _, rois, _ = _snapshot_index(fly["path"])
        try:
            data = snapshot_jpeg(fly["path"], rowid, rois.get(fly["region_id"]),
                                 crop=request.args.get("crop", "roi") != "full", width=width)
        except KeyError:
            return jsonify({"error": f"no snapshot {rowid}"}), 404
        response = Response(data, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "private, max-age=86400"
        return response

    @app.route("/api/ethoscope/sleep", methods=["GET"])
    def ethoscope_sleep():
        token = request.args.get("progress")      # the client polls /api/ethoscope/sleep/progress with it
        experiment, flies = _flies(_scope())
        params = _params()
        todo = [f for f in flies if f["path"]]
        if token:
            with _progress_lock:
                _progress[token] = {"done": 0, "total": len(todo), "t": time.time()}

        def one(fly):
            try:
                return {**fly, "trace": _fly_trace(fly, params)}
            except Exception as error:
                logger.exception("ethoscope sleep of %s fly %s", experiment, fly["key"])
                return {**fly, "trace": None, "error": str(error)}
            finally:
                if token:
                    with _progress_lock:
                        _progress[token]["done"] += 1

        # one fly after the other: reading a ROI hands the GIL back and forth on every row (sqlite3), so threads
        # only fought over it (40 flies: > 2 min with 8 threads, ~25 s in turn) and starved the other requests
        try:
            done = {f["key"]: one(f) for f in todo}
        finally:
            if token:
                with _progress_lock:
                    _progress.pop(token, None)
                    # forget tokens of requests that never ended
                    for k in [k for k, v in _progress.items() if time.time() - v["t"] > 3600]:
                        _progress.pop(k)
        out = [done.get(f["key"], {**f, "trace": None}) for f in flies]
        return jsonify({"experiment": experiment, "params": params, "flies": out})

    @app.route("/api/ethoscope/sleep/progress", methods=["GET"])
    def ethoscope_sleep_progress():
        """?token= of a running /sleep request: flies scored so far (`done`, `total`); null once it ended"""
        with _progress_lock:
            entry = _progress.get(request.args.get("token", ""))
            return jsonify(dict(entry) if entry else None)

    @app.route("/api/ethoscope/scopes", methods=["GET"])
    def ethoscope_scopes():
        """How many flies (ROIs with a local run) and runs each scope would plot, from ROI_MAP only"""
        _, flies = _flies()
        out = {}
        for scope in SCOPES:
            try:
                chosen = scope_flies(flies, scope)
            except Exception as error:
                logger.warning("ethoscope scope %s: %s", scope, error)
                out[scope] = None
                continue
            out[scope] = {"flies": sum(1 for f in chosen if f["path"]),
                          "runs": len({f["path"] for f in chosen if f["path"]}),
                          "in_sheet": sum(1 for f in chosen if f["path"] and (f.get("in_experiment") or f.get("in_sheet")))}
        return jsonify(out)
