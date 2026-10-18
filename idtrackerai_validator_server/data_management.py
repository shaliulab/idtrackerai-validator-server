"""
data_management.py  —  disk usage and backup status of every experiment folder, for
the frontend's Data management tab, and deletion of the experiments that are backed up.

Every folder $FLYHOSTEL_VIDEOS/FlyHostel<N>/<X>X/<date>_<time>/ is compared with its
copy in $FLYHOSTEL_BACKUP (default /flyhostel_data/backup, same relative path).
The backup is not a mirror: flyhostel/scripts/main.py (FlyHostelGroup.backup and
FlyHostelLoader.backup in flyhostel) copies only the files that cannot be remade
cheaply. `is_critical` lists the same files:

    <experiment>.db, index.db, metadata.yaml, landmarks.toml, *.pfs, <date>.conf, <date>.toml
    raw chunks 0-399 (NNNNNN.mp4 / .npz / .png)
    flyhostel/single_animal/<local id>/<chunk>.mp4[.predictions.slp|.h5], chunks 0-399
        tracked in the database (ROI_0); many experiments end before chunk 399
    flyhostel/validation/validation.csv, validation_lags.csv, *_annotations.zip
    flyhostel/proboscis_extensions/*_traces.feather, pe_bouts/
    interactions/<experiment>_database.feather
    motionmapper/<NN>/<experiment>__<NN>*

A critical file is backed up when the copy has the same size and the local file is
not newer (rsync -a keeps the mtime). Status of an experiment:
    ok          every critical file is backed up
    outdated    some differ from their copy (e.g. the database changed since)
    incomplete  some have no copy
    none        no backup folder
Chunks must be consecutive: `tracking_gaps` (chunks without rows in ROI_0 between
its first and last chunk) and `raw_gaps` (missing NNNNNN.mp4 between the first and
last one) are listed as `problems`, shown apart from the backup status.
`local_only_bytes` is what is not in the backup at all (critical or not): chunks
>= 400, idtrackerai sessions, ... and is lost on delete.

Dropbox (dropbox.py) is checked apart, by a job (POST /api/data_management/dropbox),
kept in dropbox.json next to $DATA_MANAGEMENT_CACHE: every top-level file of the
experiment folder by size, and the raw chunk files (NNNNNN.mp4/.npz/.png/.avi) outside
the chunks tracked in the database, which no analysis uses: with "verify" their
contents too. Those verified are `deletable_bytes`.

Walking ~800 folders takes minutes on the HDDs, so it runs as a job (one thread per
FlyHostel) and the result is kept in $DATA_MANAGEMENT_CACHE until rescanned.

POST /api/data_management/delete rescans the experiment first and only deletes it
when it is "ok", not loaded, and its folder does not resolve into the backup.
"""
import os
import re
import json
import time
import shutil
import sqlite3
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from flask import jsonify, request

from idtrackerai_validator_server import dropbox
from idtrackerai_validator_server.jobs import start_job
from idtrackerai_validator_server.experiments import experiment_folder

logger = logging.getLogger(__name__)

BACKUP_ROOT = os.environ.get("FLYHOSTEL_BACKUP", "/flyhostel_data/backup")
CACHE_FILE = os.path.expanduser(os.environ.get(
    "DATA_MANAGEMENT_CACHE", "~/.cache/idtrackerai-validator/data_management.json"))
DROPBOX_CACHE = os.path.join(os.path.dirname(CACHE_FILE), "dropbox.json")
# rclone calls to Dropbox at once
DROPBOX_WORKERS = 8
# chunks=list(range(0, 400)) in flyhostel/scripts/main.py
BACKUP_CHUNKS = 400
# mtimes on different file systems may be rounded
MTIME_TOLERANCE = 2
MAX_LISTED = 50

HOSTEL_RE = re.compile(r"FlyHostel\d+$")
GROUP_RE = re.compile(r"\d+X$")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
CHUNK_RE = re.compile(r"(\d{6})\.(mp4|npz|png)$")
# every file of a raw chunk (the .avi has the chunk's first frame)
RAW_RE = re.compile(r"(\d{6})\.(mp4|npz|png|avi)$")
POSE_RE = re.compile(r"flyhostel/single_animal/[^/]+/(\d{6})\.mp4(\.predictions\.(slp|h5))?$")

_results = {}               # experiment (slash form) -> record
_results_lock = threading.Lock()
_scan_lock = threading.Lock()
_deleting = set()
_dropbox = {}               # experiment -> last Dropbox check
_dropbox_lock = threading.Lock()


def flat_name(experiment):
    return experiment.replace("/", "_")


def compact_ranges(numbers):
    """[1, 2, 3, 7] -> '1-3, 7'"""
    runs = []
    for n in sorted(numbers):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def is_critical(relpath, experiment, tracked=None):
    """Whether flyhostel's backup copies `relpath` (relative to the experiment folder).
    `tracked`: (first, last) chunk tracked in the database, None if it has no tracking"""
    flat = flat_name(experiment)
    date_time = experiment.split("/")[-1]
    if "/" not in relpath:
        if relpath in (f"{flat}.db", "index.db", "metadata.yaml", "landmarks.toml",
                       f"{date_time}.conf", f"{date_time}.toml") or relpath.endswith(".pfs"):
            return True
        match = CHUNK_RE.fullmatch(relpath)
        return bool(match) and int(match.group(1)) < BACKUP_CHUNKS
    match = POSE_RE.fullmatch(relpath)
    if match:
        chunk = int(match.group(1))
        return tracked is not None and tracked[0] <= chunk <= tracked[1] and chunk < BACKUP_CHUNKS
    if relpath in ("flyhostel/validation/validation.csv", "flyhostel/validation/validation_lags.csv",
                   f"interactions/{flat}_database.feather"):
        return True
    if re.fullmatch(r"flyhostel/validation/[^/]+_annotations\.zip", relpath):
        return True
    if re.fullmatch(r"flyhostel/proboscis_extensions/[^/]+_traces\.feather", relpath):
        return True
    if relpath.startswith("flyhostel/proboscis_extensions/pe_bouts/"):
        return True
    match = re.match(r"motionmapper/(\d{2})/([^/]+)", relpath)
    return bool(match) and match.group(2).startswith(f"{flat}__{match.group(1)}")


def list_experiments(root):
    """Slash-form names of every experiment folder FlyHostel<N>/<X>X/<date>_<time>."""
    experiments = []
    for hostel in sorted(os.listdir(root)):
        if not HOSTEL_RE.match(hostel) or not os.path.isdir(os.path.join(root, hostel)):
            continue
        for group in sorted(os.listdir(os.path.join(root, hostel))):
            folder = os.path.join(root, hostel, group)
            if not GROUP_RE.match(group) or not os.path.isdir(folder):
                continue
            for date in sorted(os.listdir(folder)):
                if DATE_RE.match(date) and os.path.isdir(os.path.join(folder, date)):
                    experiments.append(f"{hostel}/{group}/{date}")
    return experiments


def _walk(top):
    """relpath -> (size, mtime, bytes on disk) of every file under `top`. Symbolic
    links inside are not followed (deleting removes the link, not its target)."""
    files = {}
    stack = [""]
    while stack:
        rel = stack.pop()
        try:
            entries = list(os.scandir(os.path.join(top, rel)))
        except OSError as error:
            logger.warning("Cannot list %s: %s", os.path.join(top, rel), error)
            continue
        for entry in entries:
            path = f"{rel}/{entry.name}" if rel else entry.name
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(path)
                    continue
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            files[path] = (st.st_size, st.st_mtime, st.st_blocks * 512)
    return files


def tracked_chunks(dbfile):
    """(first, last) chunk with rows in ROI_0 and the chunks without rows in between,
    or (None, []) when there is no tracking. min and max are separate queries so that
    both use the frame_number index (together they scan the table: ~1 min)"""
    with sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True) as conn:
        chunksize = int(float(conn.execute("SELECT value FROM METADATA WHERE field = 'chunksize'").fetchone()[0]))
        first = conn.execute("SELECT MIN(frame_number) FROM ROI_0").fetchone()[0]
        last = conn.execute("SELECT MAX(frame_number) FROM ROI_0").fetchone()[0]
        if first is None:
            return None, []
        first, last = first // chunksize, last // chunksize
        gaps = [chunk for chunk in range(first + 1, last)
                if conn.execute("SELECT 1 FROM ROI_0 WHERE frame_number >= ? AND frame_number < ? LIMIT 1",
                                (chunk * chunksize, (chunk + 1) * chunksize)).fetchone() is None]
    return (first, last), gaps


def scan_experiment(root, experiment):
    folder = os.path.join(root, experiment)
    backup = os.path.join(BACKUP_ROOT, experiment)
    local = _walk(folder)
    copies = _walk(backup) if os.path.isdir(backup) else None

    tracked, tracking_gaps, problems = None, [], []
    if f"{flat_name(experiment)}.db" in local:
        try:
            tracked, tracking_gaps = tracked_chunks(os.path.join(folder, f"{flat_name(experiment)}.db"))
        except sqlite3.Error as error:
            problems.append(f"cannot read the tracking of the database: {error}")

    missing, changed = [], []
    critical_bytes = 0
    local_only_bytes = 0
    local_only = defaultdict(int)      # top-level entry -> bytes not in the backup
    chunks = set()
    for relpath, (size, mtime, used) in local.items():
        match = CHUNK_RE.fullmatch(relpath)
        if match:
            chunks.add(int(match.group(1)))
        copy = copies.get(relpath) if copies is not None else None
        same = copy is not None and copy[0] == size and mtime <= copy[1] + MTIME_TOLERANCE
        if not same:
            local_only_bytes += used
            local_only[relpath.split("/")[0] + "/" if "/" in relpath else
                       ("raw chunks" if match else "other files")] += used
        if is_critical(relpath, experiment, tracked):
            critical_bytes += size
            if copy is None:
                missing.append(relpath)
            elif not same:
                changed.append(relpath)

    n_critical = sum(1 for relpath in local if is_critical(relpath, experiment, tracked))
    raw_gaps = sorted(set(range(min(chunks), max(chunks))) - chunks) if chunks else []
    if tracking_gaps:
        problems.append(f"{len(tracking_gaps)} chunk(s) without tracking between chunks {tracked[0]} and {tracked[1]}: "
                        + compact_ranges(tracking_gaps))
    if raw_gaps:
        problems.append(f"{len(raw_gaps)} raw chunk(s) missing between chunks {min(chunks)} and {max(chunks)}: "
                        + compact_ranges(raw_gaps))
    if copies is None:
        status = "none"
    elif missing:
        status = "incomplete"
    elif changed:
        status = "outdated"
    else:
        status = "ok"

    hostel, group, date = experiment.split("/")
    return {
        "experiment": experiment,
        "flyhostel": hostel,
        "group_size": int(group[:-1]),
        "date": date,
        "path": os.path.realpath(folder),
        "backup_path": backup if copies is not None else None,
        "has_database": f"{flat_name(experiment)}.db" in local,
        "n_files": len(local),
        "local_bytes": sum(used for _, _, used in local.values()),
        "backup_bytes": sum(used for _, _, used in copies.values()) if copies is not None else 0,
        "local_only_bytes": local_only_bytes,
        "local_only": dict(sorted(local_only.items(), key=lambda kv: -kv[1])),
        "n_chunks": len(chunks),
        "chunks_after_backup": sum(1 for c in chunks if c >= BACKUP_CHUNKS),
        "tracked_chunks": list(tracked) if tracked else None,
        "tracking_gaps": tracking_gaps,
        "raw_gaps": raw_gaps,
        "problems": problems,
        "n_critical": n_critical,
        "critical_bytes": critical_bytes,
        "n_missing": len(missing),
        "n_changed": len(changed),
        "missing": sorted(missing)[:MAX_LISTED],
        "changed": sorted(changed)[:MAX_LISTED],
        "status": status,
        "scanned_at": time.time(),
    }


def dropbox_check(root, experiment, verify=False):
    """Top-level files of the experiment folder against their copy on Dropbox, and
    the raw chunk files outside the tracked chunks (contents compared when `verify`)."""
    folder = os.path.join(root, experiment)
    remote = dropbox.list_remote(experiment)
    tracked = None
    dbfile = os.path.join(folder, f"{flat_name(experiment)}.db")
    if os.path.exists(dbfile):
        tracked, _ = tracked_chunks(dbfile)

    statuses = {}
    outside = []                       # (name, chunk, bytes on disk, status)
    for entry in os.scandir(folder):
        if not entry.is_file(follow_symlinks=False):
            continue
        match = RAW_RE.fullmatch(entry.name)
        chunk = int(match.group(1)) if match else None
        is_outside = chunk is not None and tracked is not None and not tracked[0] <= chunk <= tracked[1]
        status = dropbox.compare(entry.path, (remote or {}).get(entry.name), verify=verify and is_outside)
        statuses[entry.name] = status
        if is_outside:
            outside.append((entry.name, chunk, entry.stat(follow_symlinks=False).st_blocks * 512, status))

    def counts(values):
        result = defaultdict(int)
        for value in values:
            result[value] += 1
        return dict(result)

    not_backed_up = sorted(name for name, status in statuses.items()
                           if status not in (dropbox.SAME_SIZE, dropbox.VERIFIED))
    return {
        "checked_at": time.time(),
        "remote": f"{dropbox.REMOTE}/{experiment}",
        "on_dropbox": remote is not None,
        "verified": verify,
        "n_files": len(statuses),
        "counts": counts(statuses.values()),
        "n_not_backed_up": len(not_backed_up),
        "not_backed_up": not_backed_up[:MAX_LISTED],
        "tracked_chunks": list(tracked) if tracked else None,
        "outside": {
            "chunks": compact_ranges({chunk for _, chunk, _, _ in outside}),
            "n_files": len(outside),
            "bytes": sum(used for _, _, used, _ in outside),
            "counts": counts(status for _, _, _, status in outside),
            # contents identical on Dropbox: can go without touching any analysis
            "deletable_bytes": sum(used for _, _, used, status in outside if status == dropbox.VERIFIED),
            "not_on_dropbox": sorted(name for name, _, _, status in outside
                                     if status not in (dropbox.SAME_SIZE, dropbox.VERIFIED))[:MAX_LISTED],
        },
    }


def _load_json(path, into):
    try:
        with open(path) as handle:
            into.update(json.load(handle))
    except (OSError, ValueError):
        pass


def _save_json(path, data, lock):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with lock:
        data = dict(data)
    with open(tmp, "w") as handle:
        json.dump(data, handle)
    os.replace(tmp, path)


def _load_cache():
    _load_json(CACHE_FILE, _results)
    _load_json(DROPBOX_CACHE, _dropbox)


def _save_cache():
    _save_json(CACHE_FILE, _results, _results_lock)


def disks(root):
    """Free / total space of every drive holding experiments or backups."""
    seen = {}
    for base in (root, BACKUP_ROOT):
        try:
            names = os.listdir(base)
        except OSError:
            continue
        for hostel in sorted(names):
            path = os.path.realpath(os.path.join(base, hostel))
            if not HOSTEL_RE.match(hostel) or not os.path.isdir(path):
                continue
            device = os.stat(path).st_dev
            usage = shutil.disk_usage(path)
            entry = seen.setdefault(device, {
                "path": path, "total": usage.total, "used": usage.used, "free": usage.free,
                "hostels": [], "role": "backup" if base == BACKUP_ROOT else "videos",
            })
            entry["hostels"].append(hostel if base == root else f"{hostel} (backup)")
    return list(seen.values())


def _remove_from_index(root, experiment):
    """Drop the experiment's database from index.txt; returns how many lines went."""
    path = os.path.join(root, "index.txt")
    with open(path) as handle:
        lines = handle.read().splitlines()
    def of_experiment(line):
        # <experiment>/*.db, and the databases of its arenas (<experiment>/arenas/<n>/*.db)
        parts = os.path.normpath(line).split(os.sep)
        return parts[-4:-1] == experiment.split("/") or (
            len(parts) >= 6 and parts[-3] == "arenas" and parts[-6:-3] == experiment.split("/"))
    kept = [line for line in lines if not of_experiment(line)]
    if len(kept) != len(lines):
        tmp = path + ".tmp"
        with open(tmp, "w") as handle:
            handle.write("\n".join(kept) + "\n")
        os.replace(tmp, path)
    return len(lines) - len(kept)


def _delete_tree(top, on_progress):
    """Remove `top` file by file (for progress), then its folders."""
    files = []
    for dirpath, dirnames, filenames in os.walk(top):
        files += [os.path.join(dirpath, name) for name in filenames]
        files += [os.path.join(dirpath, name) for name in dirnames
                  if os.path.islink(os.path.join(dirpath, name))]
    for i, path in enumerate(files):
        os.unlink(path)
        if i % 200 == 0:
            on_progress(0.95 * i / max(1, len(files)))
    shutil.rmtree(top)


def register_data_management(app, get_selected_experiment):
    root = os.environ["FLYHOSTEL_VIDEOS"]
    _load_cache()

    @app.route("/api/data_management", methods=["GET"])
    def get_data_management():
        try:
            experiments = list_experiments(root)
        except OSError as error:
            return jsonify({"error": str(error)}), 500
        with _results_lock:
            known = set(experiments)
            for gone in [e for e in _results if e not in known]:
                del _results[gone]
            records = [dict(_results.get(e) or {"experiment": e, "flyhostel": e.split("/")[0],
                                                "group_size": int(e.split("/")[1][:-1]),
                                                "date": e.split("/")[2], "status": None},
                            dropbox=_dropbox.get(e))
                       for e in experiments]
        return jsonify({
            "experiments": records,
            "disks": disks(root),
            "backup_root": BACKUP_ROOT,
            "backup_chunks": BACKUP_CHUNKS,
            # its experiment folder when an arena is loaded
            "loaded": experiment_folder(get_selected_experiment()) if get_selected_experiment() else None,
            "scanning": _scan_lock.locked(),
            "dropbox_remote": dropbox.REMOTE,
            "dropbox_checking": _dropbox_lock.locked(),
        })

    @app.route("/api/data_management/scan", methods=["POST"])
    def scan():
        """Job scanning the given experiments (default: all), one thread per FlyHostel."""
        wanted = (request.get_json(silent=True) or {}).get("experiments")
        experiments = list_experiments(root)
        if wanted:
            experiments = [e for e in experiments if e in set(wanted)]
        by_hostel = defaultdict(list)
        for experiment in experiments:
            by_hostel[experiment.split("/")[0]].append(experiment)
        hostels = sorted(by_hostel)
        if not _scan_lock.acquire(blocking=False):
            return jsonify({"error": "A scan is already running"}), 409

        def work(job):
            errors = []

            def run(index):
                hostel = hostels[index]
                job.update(index, status="running")
                todo = by_hostel[hostel]
                for i, experiment in enumerate(todo):
                    try:
                        record = scan_experiment(root, experiment)
                        with _results_lock:
                            _results[experiment] = record
                    except Exception as error:
                        logger.exception("Could not scan %s", experiment)
                        errors.append(f"{experiment}: {error}")
                    job.update(index, progress=(i + 1) / len(todo))
                job.update(index, status="done", progress=1.0)

            try:
                with ThreadPoolExecutor(max_workers=max(1, len(hostels))) as pool:
                    list(pool.map(run, range(len(hostels))))
                _save_cache()
                job.result = {"scanned": len(experiments), "errors": errors}
            finally:
                _scan_lock.release()

        title = f"Scan {experiments[0]}" if len(experiments) == 1 else f"Scan {len(experiments)} experiments"
        try:
            job = start_job(title, [f"{h} ({len(by_hostel[h])})" for h in hostels], work)
        except Exception:
            _scan_lock.release()
            raise
        return jsonify(job.to_dict())

    @app.route("/api/data_management/dropbox", methods=["POST"])
    def check_dropbox():
        """{"experiments": [...] (default: all), "verify": bool}: job comparing them with Dropbox."""
        body = request.get_json(silent=True) or {}
        experiments = list_experiments(root)
        if body.get("experiments"):
            experiments = [e for e in experiments if e in set(body["experiments"])]
        verify = bool(body.get("verify"))
        by_hostel = defaultdict(list)
        for experiment in experiments:
            by_hostel[experiment.split("/")[0]].append(experiment)
        hostels = sorted(by_hostel)
        if not _dropbox_lock.acquire(blocking=False):
            return jsonify({"error": "A Dropbox check is already running"}), 409

        def work(job):
            errors = []
            done = defaultdict(int)
            progress_lock = threading.Lock()

            def run(experiment):
                hostel = experiment.split("/")[0]
                index = hostels.index(hostel)
                job.update(index, status="running")
                try:
                    result = dropbox_check(root, experiment, verify=verify)
                    with _results_lock:
                        _dropbox[experiment] = result
                except Exception as error:
                    logger.exception("Could not check %s on Dropbox", experiment)
                    errors.append(f"{experiment}: {error}")
                with progress_lock:
                    done[hostel] += 1
                    finished = done[hostel] == len(by_hostel[hostel])
                    job.update(index, progress=done[hostel] / len(by_hostel[hostel]),
                               **({"status": "done"} if finished else {}))

            try:
                with ThreadPoolExecutor(max_workers=DROPBOX_WORKERS) as pool:
                    list(pool.map(run, experiments))
                _save_json(DROPBOX_CACHE, _dropbox, _results_lock)
                job.result = {"checked": len(experiments), "errors": errors, "verify": verify}
            finally:
                _dropbox_lock.release()

        title = (f"Check {experiments[0]} on Dropbox" if len(experiments) == 1
                 else f"Check {len(experiments)} experiments on Dropbox") + (" (verify contents)" if verify else "")
        try:
            job = start_job(title, [f"{h} ({len(by_hostel[h])})" for h in hostels], work)
        except Exception:
            _dropbox_lock.release()
            raise
        return jsonify(job.to_dict())

    @app.route("/api/data_management/delete", methods=["POST"])
    def delete():
        """{"experiment", "confirm"}: job deleting the experiment's folder, if backed up."""
        body = request.get_json(silent=True) or {}
        experiment = body.get("experiment")
        if experiment not in list_experiments(root):
            return jsonify({"error": f"No experiment folder {experiment}"}), 404
        if body.get("confirm") != experiment:
            return jsonify({"error": "Type the experiment name to confirm"}), 400
        loaded = get_selected_experiment()
        if loaded and experiment_folder(loaded) == experiment:
            return jsonify({"error": f"{experiment} is loaded in the viewer: load another experiment first"}), 409
        if experiment in _deleting:
            return jsonify({"error": f"{experiment} is already being deleted"}), 409
        _deleting.add(experiment)

        def work(job):
            try:
                job.update(0, status="running", progress=0.0)
                folder = os.path.join(root, experiment)
                target = os.path.realpath(folder)
                backup = os.path.realpath(os.path.join(BACKUP_ROOT, experiment))
                backup_tops = {os.path.realpath(os.path.join(BACKUP_ROOT, h)) for h in os.listdir(BACKUP_ROOT)}
                # never delete (into) a backup
                if target == backup or any(target == t or target.startswith(t + os.sep) for t in backup_tops):
                    raise RuntimeError(f"{target} is inside the backup, refusing to delete it")
                if not DATE_RE.match(os.path.basename(target)):
                    raise RuntimeError(f"{target} does not look like an experiment folder")

                record = scan_experiment(root, experiment)
                with _results_lock:
                    _results[experiment] = record
                if record["status"] != "ok":
                    raise RuntimeError(
                        f"backup is {record['status']} ({record['n_missing']} missing, "
                        f"{record['n_changed']} changed critical files): not deleting")
                job.update(0, progress=0.05)

                _delete_tree(target, lambda p: job.update(0, progress=0.05 + p))
                if os.path.islink(folder):
                    os.unlink(folder)
                removed = _remove_from_index(root, experiment)
                with _results_lock:
                    _results.pop(experiment, None)
                _save_cache()
                logger.warning("Deleted %s (%s, %d bytes); %d line(s) removed from index.txt",
                               experiment, target, record["local_bytes"], removed)
                job.result = {"experiment": experiment, "path": target, "freed": record["local_bytes"],
                              "index_lines_removed": removed}
                job.update(0, status="done", progress=1.0)
            finally:
                _deleting.discard(experiment)

        try:
            job = start_job(f"Delete {experiment}", [experiment], work)
        except Exception:
            _deleting.discard(experiment)
            raise
        return jsonify(job.to_dict())
