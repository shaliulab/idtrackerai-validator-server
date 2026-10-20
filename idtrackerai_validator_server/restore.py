"""
restore.py  —  bring back experiment folders that are only on Dropbox, for the
"Restore from Dropbox" card of the Data management tab.

Dropbox ($DROPBOX_REMOTE, see dropbox.py) mirrors $FLYHOSTEL_VIDEOS, also for the
experiments deleted here to free space. A candidate is a folder
FlyHostel<N>/<X>X/<date>_<time>/ on Dropbox with nothing at the same path in
$FLYHOSTEL_VIDEOS (it may still be in $FLYHOSTEL_BACKUP, which is shown, since only
its critical files are there).

The candidates are annotated with the deepethogram (DEG) projects that use them: the
videos of a project are folders <project>/DATA/<experiment>_<chunk>_<local id>/ and a
video is labelled when it holds a *labels*.csv. Projects: $DEG_PROJECTS/DEG*/*/
(default /flyhostel_data/fiftyone/FlyBehaviors).

Listing (POST /api/data_management/restore/list) is a job: rclone lsf of the folders
(~15 s), the DEG projects (< 1 s warm), then `rclone lsjson -R` of the candidates to
inspect (size, files, database; ~10 s each): those given, by default the ones with
labelled DEG videos. Kept in restore.json next to $DATA_MANAGEMENT_CACHE.

Restoring (POST /api/data_management/restore) copies each experiment with
`rclone copy` (sizes and Dropbox hashes checked by rclone, mtimes kept) into
FlyHostel<N>/.restoring/<X>X/<date>_<time>/ (same drive, ignored by the scans and,
5 levels deep, by the index command), one experiment at a time per drive. It refuses when the drive would keep less than RESTORE_MARGIN free. When every
file listed on Dropbox is there with its size, the folder is renamed into place, its
databases are added to index.txt and the experiment is scanned. An interrupted
restore leaves its folder in .restoring: restoring again resumes it. Every restore is
logged to restored.jsonl next to $DATA_MANAGEMENT_CACHE.
"""
import os
import re
import glob
import json
import time
import shutil
import logging
import threading
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from flask import jsonify, request

from idtrackerai_validator_server import dropbox
from idtrackerai_validator_server import data_management as dm
from idtrackerai_validator_server.jobs import start_job

logger = logging.getLogger(__name__)

DEG_PROJECTS = os.environ.get("DEG_PROJECTS", "/flyhostel_data/fiftyone/FlyBehaviors")
RESTORE_CACHE = os.path.join(os.path.dirname(dm.CACHE_FILE), "restore.json")
RESTORE_LOG = os.path.join(os.path.dirname(dm.CACHE_FILE), "restored.jsonl")
# free space a drive keeps after a restore
RESTORE_MARGIN = 50e9
# rclone calls listing candidates at once
INSPECT_WORKERS = 8

EXPERIMENT_RE = re.compile(r"(FlyHostel\d+)/(\d+X)/(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})/?$")
# DATA/<FlyHostel<N>_<X>X_<date>_<time>>[_arena<n>]_<chunk>_<local id>
DEG_VIDEO_RE = re.compile(r"(FlyHostel\d+)_(\d+X)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_")

_state = {"listed_at": None, "remote": [], "deg": {}, "inspected": {}}
_state_lock = threading.Lock()
_list_lock = threading.Lock()
_restore_lock = threading.Lock()
_restoring = set()          # experiments of the running restore


def deg_annotations(base=DEG_PROJECTS):
    """experiment (slash form) -> {"videos", "labelled", "projects"} over every DEG project."""
    result = {}
    for data in sorted(glob.glob(os.path.join(base, "DEG*", "*", "DATA"))):
        project = os.path.relpath(os.path.dirname(data), base)
        for entry in os.scandir(data):
            match = DEG_VIDEO_RE.match(entry.name)
            if not match or not entry.is_dir():
                continue
            record = result.setdefault("/".join(match.groups()), {"videos": 0, "labelled": 0, "projects": {}})
            labelled = any("labels" in name and name.endswith(".csv") for name in os.listdir(entry.path))
            record["videos"] += 1
            record["labelled"] += labelled
            counts = record["projects"].setdefault(project, [0, 0])
            counts[0] += 1
            counts[1] += labelled
    return result


def list_remote_experiments(timeout=600):
    """Slash-form names of the experiment folders on Dropbox."""
    proc = subprocess.run([dropbox.RCLONE, "lsf", "-R", "--dirs-only", "--max-depth", "3", dropbox.REMOTE],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        raise RuntimeError(f"rclone lsf {dropbox.REMOTE} failed: {lines[-1] if lines else proc.returncode}")
    return sorted("/".join(m.groups()) for m in map(EXPERIMENT_RE.fullmatch, proc.stdout.splitlines()) if m)


def inspect(experiment, files=None):
    """What restoring would bring: size, files, raw chunks and databases on Dropbox."""
    if files is None:
        files = dropbox.list_remote(experiment, recursive=True) or {}
    flat = dm.flat_name(experiment)
    chunks = {int(m.group(1)) for m in map(dm.CHUNK_RE.fullmatch, files) if m}
    return {
        "inspected_at": time.time(),
        "bytes": sum(size for size, _ in files.values()),
        "n_files": len(files),
        "n_chunks": len(chunks),
        "chunks": dm.compact_ranges(chunks),
        "has_database": f"{flat}.db" in files,
        "databases": sorted(p for p in files if p.endswith("db") and os.path.basename(p) != "index.db"
                            and ("/" not in p or re.fullmatch(r"arenas/\d+/[^/]+", p))),
        "has_pose": any(dm.POSE_RE.fullmatch(p) for p in files),
    }


def destination(root, experiment):
    """(final folder, temporary folder) of a restore; the FlyHostel<N> folder must exist."""
    hostel, group, date = experiment.split("/")
    if not os.path.isdir(os.path.join(root, hostel)):
        raise RuntimeError(f"{os.path.join(root, hostel)} does not exist (the drive is not mounted?)")
    return os.path.join(root, hostel, group, date), os.path.join(root, hostel, ".restoring", group, date)


def drive_of(root, experiment):
    path = os.path.realpath(os.path.join(root, experiment.split("/")[0]))
    return path, os.stat(path).st_dev


def _add_to_index(root, experiment, databases):
    """Append the experiment's databases to index.txt; returns how many lines were added."""
    path = os.path.join(root, "index.txt")
    with open(path) as handle:
        lines = handle.read().splitlines()
    known = {os.path.normpath(line) for line in lines}
    new = [f"./{experiment}/{db}" for db in databases if os.path.normpath(f"./{experiment}/{db}") not in known]
    if new:
        tmp = path + ".tmp"
        with open(tmp, "w") as handle:
            handle.write("\n".join(lines + new) + "\n")
        os.replace(tmp, path)
    return len(new)


def _local_sizes(top):
    return {relpath: size for relpath, (size, _, _) in dm._walk(top).items()}


def _rclone_copy(source, target, on_bytes):
    """rclone copy, calling on_bytes(bytes transferred) from its stats. Raises on failure."""
    cmd = [dropbox.RCLONE, "copy", source, target, "--transfers", "4", "--checkers", "8",
           "--use-json-log", "--stats", "5s", "--stats-log-level", "NOTICE"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    errors = []
    for line in proc.stderr:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if "stats" in entry:
            on_bytes(entry["stats"].get("bytes", 0))
        elif entry.get("level") == "error":
            errors.append(f"{entry.get('object', '')}: {entry.get('msg', '')}".strip(": "))
    if proc.wait() != 0:
        raise RuntimeError(f"rclone copy failed ({len(errors)} error(s)){': ' + errors[-1] if errors else ''}")


def restore_experiment(root, experiment, on_progress):
    """Copy the experiment from Dropbox into $FLYHOSTEL_VIDEOS; returns a summary."""
    final, tmp = destination(root, experiment)
    if os.path.lexists(final):
        raise RuntimeError(f"{final} already exists")
    files = dropbox.list_remote(experiment, recursive=True)
    if not files:
        raise RuntimeError(f"{experiment} is not on Dropbox")
    info = inspect(experiment, files)
    with _state_lock:
        _state["inspected"][experiment] = info

    have = _local_sizes(tmp) if os.path.isdir(tmp) else {}
    already = sum(size for p, (size, _) in files.items() if have.get(p) == size)
    needed = info["bytes"] - already
    free = shutil.disk_usage(drive_of(root, experiment)[0]).free
    if free - needed < RESTORE_MARGIN:
        raise RuntimeError(f"needs {needed / 1e9:.1f} GB but {free / 1e9:.1f} GB are free "
                           f"(at least {RESTORE_MARGIN / 1e9:.0f} GB must stay free)")
    os.makedirs(tmp, exist_ok=True)
    started = time.time()
    on_progress(already / max(1, info["bytes"]))
    _rclone_copy(f"{dropbox.REMOTE}/{experiment}", tmp,
                 lambda done: on_progress(0.97 * min(1.0, (already + done) / max(1, info["bytes"]))))

    have = _local_sizes(tmp)
    wrong = sorted(p for p, (size, _) in files.items() if have.get(p) != size)
    if wrong:
        raise RuntimeError(f"{len(wrong)} file(s) missing or of another size after copying, "
                           f"restore again to resume: {', '.join(wrong[:5])}")
    if os.path.lexists(final):
        raise RuntimeError(f"{final} appeared while restoring; the copy is in {tmp}")
    os.makedirs(os.path.dirname(final), exist_ok=True)
    os.rename(tmp, final)
    for empty in (os.path.dirname(tmp), os.path.dirname(os.path.dirname(tmp))):
        try:
            os.rmdir(empty)
        except OSError:
            break
    added = _add_to_index(root, experiment, info["databases"])

    try:
        record = dm.scan_experiment(root, experiment)
        with dm._results_lock:
            dm._results[experiment] = record
        dm._save_cache()
    except Exception:
        logger.exception("Could not scan %s after restoring it", experiment)

    summary = {"experiment": experiment, "path": os.path.realpath(final), "bytes": info["bytes"],
               "copied_bytes": needed, "n_files": info["n_files"], "index_lines_added": added,
               "seconds": round(time.time() - started), "remote": f"{dropbox.REMOTE}/{experiment}",
               "restored_at": time.time()}
    os.makedirs(os.path.dirname(RESTORE_LOG), exist_ok=True)
    with open(RESTORE_LOG, "a") as handle:
        handle.write(json.dumps(summary) + "\n")
    logger.warning("Restored %s from Dropbox (%d bytes, %d files) into %s",
                   experiment, info["bytes"], info["n_files"], summary["path"])
    return summary


def register_restore(app):
    root = os.environ["FLYHOSTEL_VIDEOS"]
    dm._load_json(RESTORE_CACHE, _state)

    def candidates():
        with _state_lock:
            remote, deg, inspected = list(_state["remote"]), dict(_state["deg"]), dict(_state["inspected"])
        drives = {}
        records = []
        for experiment in remote:
            if os.path.lexists(os.path.join(root, experiment)):
                continue
            hostel, group, date = experiment.split("/")
            try:
                final, tmp = destination(root, experiment)
                partial = os.path.isdir(tmp)
                path, device = drive_of(root, experiment)
                drives.setdefault(device, {"path": path, "free": shutil.disk_usage(path).free, "hostels": set()})
                drives[device]["hostels"].add(hostel)
            except (RuntimeError, OSError):
                partial, device = False, None
            records.append({
                "experiment": experiment, "flyhostel": hostel, "group_size": int(group[:-1]), "date": date,
                "deg": deg.get(experiment), "dropbox": inspected.get(experiment),
                "in_backup": os.path.isdir(os.path.join(dm.BACKUP_ROOT, experiment)),
                "partial": partial, "drive": device, "restoring": experiment in _restoring,
            })
        # DEG videos of experiments that are neither here nor on Dropbox
        listed = set(remote)
        lost = sorted(e for e in deg if e not in listed and not os.path.lexists(os.path.join(root, e)))
        return records, [dict(d, device=k, hostels=sorted(d["hostels"])) for k, d in drives.items()], lost

    @app.route("/api/data_management/restore", methods=["GET"])
    def get_restore():
        try:
            records, drives, lost = candidates()
        except OSError as error:
            return jsonify({"error": str(error)}), 500
        with _state_lock:
            listed_at = _state["listed_at"]
        return jsonify({
            "candidates": records, "drives": drives, "listed_at": listed_at,
            "deg_root": DEG_PROJECTS, "remote": dropbox.REMOTE, "margin": RESTORE_MARGIN,
            "deg_not_found": [{"experiment": e, "deg": _state["deg"][e],
                               "in_backup": os.path.isdir(os.path.join(dm.BACKUP_ROOT, e))} for e in lost],
            "listing": _list_lock.locked(), "restoring": _restore_lock.locked(),
        })

    @app.route("/api/data_management/restore/list", methods=["POST"])
    def list_restore():
        """{"inspect": [...]} (default: the candidates with labelled DEG videos not inspected yet):
        job listing the experiments on Dropbox and the DEG projects, then inspecting those."""
        wanted = (request.get_json(silent=True) or {}).get("inspect")
        if not _list_lock.acquire(blocking=False):
            return jsonify({"error": "Dropbox is already being listed"}), 409

        def work(job):
            try:
                job.update(0, status="running")
                remote = list_remote_experiments()
                job.update(0, status="done", progress=1.0)
                job.update(1, status="running")
                deg = deg_annotations()
                with _state_lock:
                    _state.update(remote=remote, deg=deg, listed_at=time.time())
                job.update(1, status="done", progress=1.0)

                todo = [r["experiment"] for r in candidates()[0]
                        if (r["experiment"] in set(wanted) if wanted is not None
                            else (r["deg"] or {}).get("labelled") and not r["dropbox"])]
                job.update(2, status="running", name=f"Inspect {len(todo)} candidate(s)")
                errors, done = [], [0]
                progress_lock = threading.Lock()

                def run(experiment):
                    try:
                        info = inspect(experiment)
                        with _state_lock:
                            _state["inspected"][experiment] = info
                    except Exception as error:
                        logger.exception("Could not inspect %s on Dropbox", experiment)
                        errors.append(f"{experiment}: {error}")
                    with progress_lock:
                        done[0] += 1
                        job.update(2, progress=done[0] / len(todo))

                with ThreadPoolExecutor(max_workers=INSPECT_WORKERS) as pool:
                    list(pool.map(run, todo))
                job.update(2, status="done", progress=1.0)
                dm._save_json(RESTORE_CACHE, _state, _state_lock)
                job.result = {"remote": len(remote), "inspected": len(todo), "errors": errors}
            finally:
                _list_lock.release()

        try:
            job = start_job("List experiments on Dropbox",
                            ["List Dropbox folders", "Read DEG projects", "Inspect candidates"], work)
        except Exception:
            _list_lock.release()
            raise
        return jsonify(job.to_dict())

    @app.route("/api/data_management/restore", methods=["POST"])
    def restore():
        """{"experiments": [...]}: job copying them from Dropbox, one at a time per drive."""
        experiments = list(dict.fromkeys((request.get_json(silent=True) or {}).get("experiments") or []))
        if not experiments:
            return jsonify({"error": "No experiment given"}), 400
        bad = [e for e in experiments if not EXPERIMENT_RE.fullmatch(e)]
        if bad:
            return jsonify({"error": f"Not an experiment folder: {', '.join(bad)}"}), 400
        here = [e for e in experiments if os.path.lexists(os.path.join(root, e))]
        if here:
            return jsonify({"error": f"Already in {root}: {', '.join(here)}"}), 409
        try:
            by_drive = defaultdict(list)
            for experiment in experiments:
                destination(root, experiment)
                by_drive[drive_of(root, experiment)[1]].append(experiment)
        except (RuntimeError, OSError) as error:
            return jsonify({"error": str(error)}), 400
        if not _restore_lock.acquire(blocking=False):
            return jsonify({"error": "A restore is already running"}), 409

        def work(job):
            restored, errors = [], []

            def run(todo):
                for experiment in todo:
                    index = experiments.index(experiment)
                    job.update(index, status="running")
                    try:
                        restored.append(restore_experiment(
                            root, experiment, lambda p, i=index: job.update(i, progress=p)))
                        job.update(index, status="done", progress=1.0)
                    except Exception as error:
                        logger.exception("Could not restore %s", experiment)
                        errors.append(f"{experiment}: {error}")
                        job.update(index, status="failed", error=str(error))
                    finally:
                        _restoring.discard(experiment)

            try:
                with ThreadPoolExecutor(max_workers=len(by_drive)) as pool:
                    list(pool.map(run, by_drive.values()))
                dm._save_json(RESTORE_CACHE, _state, _state_lock)
                job.result = {"restored": restored, "errors": errors}
                if errors and not restored:
                    raise RuntimeError("; ".join(errors))
            finally:
                _restore_lock.release()

        title = (f"Restore {experiments[0]} from Dropbox" if len(experiments) == 1
                 else f"Restore {len(experiments)} experiments from Dropbox")
        _restoring.update(experiments)
        try:
            job = start_job(title, experiments, work)
        except Exception:
            _restoring.difference_update(experiments)
            _restore_lock.release()
            raise
        return jsonify(job.to_dict())
