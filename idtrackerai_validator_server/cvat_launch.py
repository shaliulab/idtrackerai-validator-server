"""Send an experiment to CVAT for validation (manual/validation/main.sh), after checking it is worth it.

    POST /api/cvat_launch/check   {experiment}                    job (jobs.py) running the checks below
    POST /api/cvat_launch/start   {experiment, job_id, force}     run main.sh in the background
    GET  /api/cvat_launch/status?experiment=...                   state of the last run of the experiment
    GET  /api/cvat_launch/runs                                    state of every run started by this server

`experiment` is the folder relative to $FLYHOSTEL_VIDEOS (or the flat name), as in the Overview tab.

Checks (level ok / warning / error). Starting with an error needs force=True:
  - one tracking database in the folder, and no main.sh already running for the experiment
  - the idtrackerai .conf is not newer than the database (else the tracking may not use it)
  - frames with fewer flies than expected (ROI_0): a big fraction means a tracking problem
    (e.g. an ROI on the dark wall) and validation would export images of most of the recording
  - chunks whose QC row has no fragments: idtrackerai crashed (np.int in the crossing detector)
    and the chunk was exported with fragment 0 / local_identity 0 for every blob
  - rows of IDENTITY with local identity 0 (validation only needs local identities; the identities across
    chunks are reported but not required, since integrating the annotations links the chunks again)
  - CVAT tasks or project already named after the experiment (the upload would duplicate them)
  - free space on the experiment's disk, and what an earlier validation run left there

main.sh runs in its own session (it survives a restart of this server) with the validation conda
env first in PATH, from $FLYHOSTEL_VALIDATION_SCRIPTS, logging to <folder>/flyhostel/validation/main.log
(an earlier log is renamed main.<time>.log). cvat_launch.json in that folder records the run.
While this server runs, a watcher stops the run (SIGTERM to its process group) when the free space
of the disk falls below MIN_FREE_GB.
"""
import glob
import io
import json
import logging
import os
import shutil
import signal
import sqlite3
import subprocess
import threading
import time

import pandas as pd
import requests
from flask import jsonify, request

from idtrackerai_validator_server import cvat
from idtrackerai_validator_server.experiments import to_flat, to_relpath
from idtrackerai_validator_server.jobs import _jobs, start_job
from idtrackerai_validator_server.overview import ARENA_RE, PATH_RE

logger = logging.getLogger(__name__)

SCRIPTS = os.path.expanduser(os.environ.get("FLYHOSTEL_VALIDATION_SCRIPTS", "~/opt/vsc-scripts/manual/validation"))
# conda env with make-identogram, auto-annotate-qc, export-images, cvat-cli and cvat_sdk
VALIDATION_ENV = os.path.expanduser(os.environ.get("FLYHOSTEL_VALIDATION_ENV", "~/mambaforge/envs/rapids-23.04"))

# Lines main.sh prints after each step, in order.
STAGES = [
    ("Done generating scenes", "scenes"),
    ("Done annotating scenes", "QC of the scenes"),
    ("Done exporting images", "images exported"),
    ("Done uploading to CVAT", "uploaded to CVAT"),
    ("Done moving the tasks to the CVAT project", "moved to the CVAT project"),
    ("Done pre-populating CVAT", "pre-populated with yolov7"),
]

SHORT_FRAMES = {"warning": 0.01, "error": 0.05}      # fraction of frames with fewer flies than expected
NO_LOCAL_ID = {"warning": 0.10, "error": 0.30}       # fraction of IDENTITY rows with local_identity 0
                                                     # (crossings: ~0-2% with 2-4 flies, ~5% with 12)
FREE_GB = {"warning": 150, "error": 50}              # free space on the experiment's disk before starting
MIN_FREE_GB = 30                                     # the watcher stops a running main.sh below this
CHECK_MAX_AGE = 15 * 60                              # seconds a check job stays valid to start from
WATCH_EVERY = 60

_runs = {}           # flat name -> run info (dict, as in cvat_launch.json)
_runs_lock = threading.Lock()


# ---------------------------------------------------------------- paths

def _paths(experiment):
    relpath = to_relpath(experiment)
    folder = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], relpath)
    validation = os.path.join(folder, "flyhostel", "validation")
    return relpath, to_flat(relpath), folder, validation


def _databases(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "*.db")) if os.path.basename(p) != "index.db")


def _record_path(validation):
    return os.path.join(validation, "cvat_launch.json")


def _read_record(validation):
    try:
        with open(_record_path(validation)) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _write_record(validation, record):
    tmp = _record_path(validation) + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(record, handle, indent=2)
    os.replace(tmp, _record_path(validation))


def _alive(pid, identifier):
    """Whether pid is still a main.sh of this experiment (pids get reused)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            cmdline = handle.read().split(b"\0")
    except OSError:
        return False
    return any(part.endswith(b"main.sh") for part in cmdline) and identifier.encode() in cmdline


def _running_elsewhere(identifier):
    """pids of main.sh processes of the experiment, started by anyone (shell, nohup, this server)."""
    pids = []
    for entry in os.listdir("/proc"):
        if entry.isdigit() and _alive(int(entry), identifier):
            pids.append(int(entry))
    return pids


def _free_gb(path):
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path).free / 1e9


def _du_gb(path):
    try:
        out = subprocess.run(["du", "-sb", path], capture_output=True, text=True, timeout=120).stdout
        return int(out.split()[0]) / 1e9
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


# ---------------------------------------------------------------- checks

def _expected_animals(relpath, conn):
    """NX of the path; for an arena of a multi-arena recording, its rows in ethoscope_metadata."""
    match = PATH_RE.search(relpath)
    if not ARENA_RE.search(relpath):
        return int(match.group(2)) if match else None, "from the folder name"
    row = conn.execute("SELECT value FROM METADATA WHERE field = 'ethoscope_metadata'").fetchone()
    if row and row[0]:
        return len(pd.read_csv(io.StringIO(row[0]), index_col=0)), "metadata rows of the arena"
    return None, "arena without metadata"


def _cvat_existing(identifier):
    """(projects, tasks) in CVAT named after the experiment."""
    auth = (os.environ["CVAT_USERNAME"], os.environ["CVAT_PASSWORD"])
    base = cvat._api_base()
    with requests.Session() as s:
        s.auth = auth
        projects = [p for p in cvat._get_all(s, f"{base}/projects", {"search": identifier})
                    if cvat._matches(p["name"], identifier)]
        tasks = [t for t in cvat._get_all(s, f"{base}/tasks", {"search": identifier})
                 if cvat._matches(t["name"], identifier)]
    return projects, tasks


def _level(value, limits):
    return "error" if value >= limits["error"] else "warning" if value >= limits["warning"] else "ok"


def run_checks(experiment, progress=lambda fraction: None):
    """[{level, name, text}] and facts about the experiment (number of animals, paths, ...)."""
    relpath, identifier, folder, validation = _paths(experiment)
    checks = []
    facts = {"experiment": relpath, "identifier": identifier, "folder": folder, "validation": validation}

    def add(level, name, text):
        checks.append({"level": level, "name": name, "text": text})

    # the pipeline itself
    missing = [p for p in (os.path.join(SCRIPTS, "main.sh"), os.path.join(VALIDATION_ENV, "bin", "make-identogram"))
               if not os.path.exists(p)]
    if missing:
        add("error", "setup", f"Not found: {', '.join(missing)} (FLYHOSTEL_VALIDATION_SCRIPTS / FLYHOSTEL_VALIDATION_ENV)")
    if not cvat.configured():
        add("error", "setup", "CVAT_HOST / CVAT_USERNAME / CVAT_PASSWORD are not set in the server's environment")

    running = _running_elsewhere(identifier)
    if running:
        add("error", "running", f"main.sh is already running for this experiment (pid {', '.join(map(str, running))})")

    databases = _databases(folder)
    if len(databases) != 1:
        add("error", "database", f"{len(databases)} tracking databases in {folder}" + (f": {', '.join(map(os.path.basename, databases))}" if databases else ""))
        return checks, facts
    db = databases[0]
    facts["database"] = db

    # .conf newer than the database: the tracking may come from older settings
    confs = [p for p in glob.glob(os.path.join(folder, "*.conf"))]
    newer = [os.path.basename(p) for p in confs if os.path.getmtime(p) > os.path.getmtime(db)]
    if newer:
        add("warning", "conf", f"{', '.join(newer)} is newer than the database: was it tracked again with these settings?")
    elif confs:
        add("ok", "conf", "The idtrackerai .conf is older than the database")

    progress(0.1)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        n_animals, source = _expected_animals(relpath, conn)
        facts["number_of_animals"] = n_animals
        if not n_animals:
            add("error", "animals", f"Could not tell the number of animals ({source})")
            return checks, facts
        add("ok" if n_animals > 1 else "warning", "animals",
            f"{n_animals} animal{'s' if n_animals > 1 else ''} ({source})"
            + ("" if n_animals > 1 else ": a single fly has no identities to validate"))

        # flies per frame
        counts = dict(conn.execute(
            "SELECT n, COUNT(*) FROM (SELECT frame_number, COUNT(*) n FROM ROI_0 GROUP BY frame_number) GROUP BY n"
        ).fetchall())
        first, last = conn.execute("SELECT MIN(frame_number), MAX(frame_number) FROM ROI_0").fetchone()
        progress(0.45)
        frames = (last - first + 1) if first is not None else 0
        empty = frames - sum(counts.values())
        short = empty + sum(c for n, c in counts.items() if n < n_animals)
        extra = sum(c for n, c in counts.items() if n > n_animals)
        fraction = short / frames if frames else 1
        facts["frames"] = frames
        add(_level(fraction, SHORT_FRAMES), "flies per frame",
            f"{short:,} of {frames:,} frames ({100 * fraction:.2f}%) have fewer than {n_animals} flies"
            + (f" ({empty:,} with none)" if empty else "")
            + (" — check the ROI and the intensity / area thresholds before validating" if fraction >= SHORT_FRAMES["warning"] else ""))
        if extra:
            add("warning" if extra / max(frames, 1) >= SHORT_FRAMES["warning"] else "ok", "flies per frame",
                f"{extra:,} frames have more than {n_animals} blobs")

        # chunks where idtrackerai crashed and blobs were exported without fragments
        if "QC" in tables:
            broken = [row[0] for row in conn.execute(
                "SELECT chunk FROM QC WHERE fragments IS NULL OR global_fragments IS NULL ORDER BY chunk")]
            if broken:
                add("error", "chunks", f"{len(broken)} chunk{'s' if len(broken) > 1 else ''} without fragments "
                    f"(idtrackerai crashed; fragment 0 for every blob): {', '.join(map(str, broken[:30]))}"
                    f"{' …' if len(broken) > 30 else ''}. Retrack them first")
            else:
                add("ok", "chunks", "Every chunk has fragments in QC")
        else:
            add("warning", "chunks", "No QC table: cannot tell whether a chunk was exported without fragments")
        progress(0.6)

        # identities: validation only uses local identities (the scenes, their QC, the movies, the yolov7
        # pre-population and the integration of the annotations, which links the chunks again). The identities
        # of the whole recording (concatenation) are only reported.
        if "IDENTITY" in tables:
            total, no_local, no_global = conn.execute(
                "SELECT COUNT(*), SUM(local_identity = 0), SUM(identity = 0) FROM IDENTITY").fetchone()
            fraction = (no_local or 0) / total if total else 1
            add(_level(fraction, NO_LOCAL_ID) if n_animals > 1 else "ok", "local identities",
                f"{100 * fraction:.2f}% of the rows of IDENTITY have local identity 0 (crossings)"
                + (" — idtrackerai did not identify many blobs" if fraction >= NO_LOCAL_ID["warning"] else ""))
            fraction = (no_global or 0) / total if total else 1
            add("ok", "identities", f"{100 * fraction:.2f}% of the rows have no identity across chunks "
                "(not needed: the validation links the chunks again)")
        else:
            add("error", "identities", "No IDENTITY table")
        progress(0.85)

    # CVAT
    if cvat.configured():
        try:
            projects, tasks = _cvat_existing(identifier)
            if projects or tasks:
                add("warning", "cvat", f"CVAT already has {len(tasks)} task{'s' if len(tasks) != 1 else ''} and "
                    f"{len(projects)} project{'s' if len(projects) != 1 else ''} named after this experiment "
                    f"({', '.join(p['name'] for p in projects) or 'no project'}): the upload adds new tasks next to them")
            else:
                add("ok", "cvat", "No CVAT task or project of this experiment yet")
        except Exception as error:
            add("error", "cvat", f"Could not reach CVAT: {error}")

    # disk
    free = _free_gb(folder)
    facts["free_gb"] = round(free, 1)
    add(_level(-free, {k: -v for k, v in FREE_GB.items()}), "disk",
        f"{free:,.0f} GB free on the experiment's disk"
        + (f" (the run is stopped below {MIN_FREE_GB} GB)" if free >= FREE_GB["error"] else ""))

    # an earlier run
    if os.path.isdir(validation):
        size = _du_gb(validation)
        log = os.path.join(validation, "main.log")
        done = os.path.exists(log) and STAGES[-1][0] in open(log, errors="replace").read()
        add("warning" if done else "ok", "earlier run",
            f"An earlier validation folder exists ({size:.1f} GB): " +
            ("its main.log says it reached CVAT already" if done else "it is reused (make-identogram --cache)"))
    progress(1.0)
    return checks, facts


# ---------------------------------------------------------------- runs

def _status(identifier, validation):
    record = _read_record(validation)
    with _runs_lock:
        record = _runs.get(identifier, record)
    if record is None:
        return {"state": "none"}
    log = record.get("log") or os.path.join(validation, "main.log")
    try:
        with open(log, errors="replace") as handle:
            text = handle.read()
    except OSError:
        text = ""
    stages = [label for marker, label in STAGES if marker in text]
    if _alive(record["pid"], identifier):
        state = "running"
    elif record.get("stopped"):
        state = "stopped"
    elif STAGES[-1][0] in text:
        state = "done"
    else:
        state = "failed"
    return {
        **record,
        "state": state,
        "stages": stages,
        "n_stages": len(STAGES),
        "tail": text.splitlines()[-40:],
        "free_gb": round(_free_gb(validation), 1),
    }


def _watch(identifier, validation, process):
    """Reap main.sh and stop it if the disk fills up."""
    while process.poll() is None:
        try:
            free = _free_gb(validation)
        except OSError:
            free = None
        if free is not None and free < MIN_FREE_GB:
            logger.error("Stopping main.sh of %s: %.0f GB free", identifier, free)
            with _runs_lock:
                _runs[identifier]["stopped"] = f"stopped: only {free:.0f} GB free on the disk (limit {MIN_FREE_GB} GB)"
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=WATCH_EVERY)
        except subprocess.TimeoutExpired:
            pass
    with _runs_lock:
        record = _runs[identifier]
        record["returncode"] = process.returncode
        record["finished"] = time.time()
        record = dict(record)
    try:
        _write_record(validation, record)
    except OSError as error:
        logger.error("Could not save %s: %s", _record_path(validation), error)


def start(experiment, checks, facts, force):
    relpath, identifier, folder, validation = _paths(experiment)
    if _running_elsewhere(identifier):
        raise RuntimeError("main.sh is already running for this experiment")
    os.makedirs(validation, exist_ok=True)
    log = os.path.join(validation, "main.log")
    if os.path.exists(log):
        os.rename(log, os.path.join(validation, time.strftime("main.%Y%m%d-%H%M%S.log", time.localtime(os.path.getmtime(log)))))

    command = ["bash", os.path.join(SCRIPTS, "main.sh"), identifier, folder, str(facts["number_of_animals"])]
    env = dict(os.environ)
    env["PATH"] = os.path.join(VALIDATION_ENV, "bin") + os.pathsep + env.get("PATH", "")
    env["CONDA_PREFIX"] = VALIDATION_ENV
    with open(log, "ab") as handle:
        handle.write(f"# started from the validator server {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(command)}\n".encode())
        handle.flush()
        process = subprocess.Popen(command, cwd=SCRIPTS, env=env, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    record = {
        "experiment": relpath, "identifier": identifier, "pid": process.pid, "command": command,
        "log": log, "started": time.time(), "forced": bool(force),
        "failed_checks": [c["text"] for c in checks if c["level"] == "error"],
        "warnings": [c["text"] for c in checks if c["level"] == "warning"],
    }
    with _runs_lock:
        _runs[identifier] = record
    _write_record(validation, record)
    threading.Thread(target=_watch, args=(identifier, validation, process), name=f"main.sh {identifier}", daemon=True).start()
    logger.info("Started main.sh for %s (pid %s)%s", identifier, process.pid, " despite failed checks" if force else "")
    return record


# ---------------------------------------------------------------- routes

def register_cvat_launch(app):

    @app.route("/api/cvat_launch/check", methods=["POST"])
    def cvat_launch_check():
        experiment = (request.get_json(silent=True) or {}).get("experiment")
        if not experiment:
            return jsonify({"error": "no experiment given"}), 400

        def work(job):
            job.update(0, status="running")
            checks, facts = run_checks(experiment, progress=lambda f: job.update(0, progress=f))
            job.result = {"checks": checks, "facts": facts, "checked_at": time.time(),
                          "status": _status(facts["identifier"], facts["validation"])}
            job.update(0, status="done", progress=1.0)

        job = start_job(f"Checks before validation of {experiment}", [to_flat(experiment)], work)
        return jsonify(job.to_dict())

    @app.route("/api/cvat_launch/start", methods=["POST"])
    def cvat_launch_start():
        body = request.get_json(silent=True) or {}
        experiment, force = body.get("experiment"), bool(body.get("force"))
        job = _jobs.get(str(body.get("job_id")))
        if not experiment or job is None or job.status != "done" or not job.result:
            return jsonify({"error": "run the checks first"}), 400
        result = job.result
        if to_relpath(result["facts"]["experiment"]) != to_relpath(experiment):
            return jsonify({"error": "these checks are of another experiment"}), 400
        if time.time() - result["checked_at"] > CHECK_MAX_AGE:
            return jsonify({"error": "the checks are too old: check again"}), 409
        errors = [c for c in result["checks"] if c["level"] == "error"]
        if errors and not force:
            return jsonify({"error": f"{len(errors)} check(s) failed", "checks": errors}), 409
        if not result["facts"].get("number_of_animals"):
            return jsonify({"error": "unknown number of animals"}), 409
        try:
            record = start(experiment, result["checks"], result["facts"], force)
        except Exception as error:
            logger.exception("Could not start main.sh for %s", experiment)
            return jsonify({"error": str(error)}), 500
        return jsonify(record)

    @app.route("/api/cvat_launch/status", methods=["GET"])
    def cvat_launch_status():
        experiment = request.args.get("experiment")
        if not experiment:
            return jsonify({"error": "no experiment given"}), 400
        _, identifier, _, validation = _paths(experiment)
        return jsonify(_status(identifier, validation))

    @app.route("/api/cvat_launch/runs", methods=["GET"])
    def cvat_launch_runs():
        with _runs_lock:
            runs = [dict(r) for r in _runs.values()]
        out = {}
        for run in runs:
            _, identifier, _, validation = _paths(run["experiment"])
            status = _status(identifier, validation)
            out[run["experiment"]] = {k: status.get(k) for k in ("state", "stages", "n_stages", "started", "stopped")}
        return jsonify(out)
