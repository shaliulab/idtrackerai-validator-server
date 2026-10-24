"""
monitor.py  —  what the nextflow pipelines of this computer are doing, for the Monitor tab.

Each pipeline is a folder of $NEXTFLOW_PIPELINES (default
/home/vibflysleep/opt/vsc-scripts/nextflow/pipelines) with a CSV of experiments; a
row is selected when one of its columns (`mark`) holds the value the run filters on:

    idtrackerai          idtrackerai_v3/idtrackerai.csv   basedir,1,...           (no queue)
    pose_estimation      pose_estimation/videos.csv       experiment,basedir,1,...
    behavior_prediction  behavior_prediction/animals.csv  experiment,basedir,identity,1,...

With a queue (run_queue.sh, started with `nohup ./run_queue.sh > run_queue.log 2>&1 &`)
the rows marked 1 are waiting; at launch they are stamped with the batch (the date,
or date_HH-MM) which is passed to nextflow as --batch, so the rows of the running
batch are those holding its stamp. Without a queue (idtrackerai) the rows marked 1
are the ones a run started now would take, and those of the running run if any.

A run is a `nextflow.cli.Launcher run` java process whose cwd is the pipeline folder.
Its tasks come from the folder's .nextflow.log (Submitted / Cached process, Task
completed lines, parsed again when the file changes) and each task is attributed to
an experiment of the run by the longest experiment name or basedir found in its
.command.sh (cached per work dir: they do not change).
"""
import os
import re
import csv
import glob
import time
import logging
import datetime
import threading

from flask import jsonify

from idtrackerai_validator_server.experiments import to_flat

logger = logging.getLogger(__name__)

PIPELINES_DIR = os.environ.get("NEXTFLOW_PIPELINES", "/home/vibflysleep/opt/vsc-scripts/nextflow/pipelines")
FLYHOSTEL_VIDEOS = os.environ.get("FLYHOSTEL_VIDEOS", "/flyhostel_data/videos")

PIPELINES = [
    {"id": "idtrackerai", "label": "idtrackerai", "folder": "idtrackerai_v3", "csv": "idtrackerai.csv",
     "experiment": None, "basedir": 0, "identity": None, "mark": 1, "status": 2},
    {"id": "pose_estimation", "label": "Pose estimation", "folder": "pose_estimation", "csv": "videos.csv",
     "experiment": 0, "basedir": 1, "identity": None, "mark": 2, "status": 3},
    {"id": "behavior_prediction", "label": "Behavior prediction", "folder": "behavior_prediction", "csv": "animals.csv",
     "experiment": 0, "basedir": 1, "identity": 2, "mark": 3, "status": 4},
]
# batches shown under "Recent batches"
RECENT_BATCHES = 5
# lines of run_queue.log returned
QUEUE_LOG_LINES = 20
# running tasks listed per run
MAX_RUNNING_TASKS = 50

BATCH_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(_\d{2}-\d{2})?$")
LOG_TIME_RE = re.compile(r"^(\w{3}-\d{2} \d{2}:\d{2}:\d{2})\.(\d{3})")
SUBMITTED_RE = re.compile(r"\[([0-9a-f]{2}/[0-9a-f]{6})\] (Submitted|Cached) process > (.+?)\s*$")
COMPLETED_RE = re.compile(r"Task completed > TaskHandler\[id: \d+; name: (.+?); status: (\w+); exit: ([^;]+); "
                          r"error: ([^;]*); workDir: (\S+?)\]")
LAUNCHER_RE = re.compile(r"nextflow\.cli\.Launcher - \$> (.+)$")

_log_cache = {}         # path -> (mtime, size, parsed)
_task_experiment = {}   # work dir -> experiment key or None
_cache_lock = threading.Lock()


def _log_time(line, year):
    match = LOG_TIME_RE.match(line)
    if not match:
        return None
    t = datetime.datetime.strptime(f"{year}-{match.group(1)}", "%Y-%b-%d %H:%M:%S")
    if t > datetime.datetime.now() + datetime.timedelta(days=1):    # a log of last year
        t = t.replace(year=year - 1)
    return t.timestamp() + int(match.group(2)) / 1000


def parse_log(path):
    """Tasks of the run that wrote this .nextflow.log: name -> state, plus the run's command and times."""
    stat = os.stat(path)
    with _cache_lock:
        cached = _log_cache.get(path)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]

    year = datetime.datetime.fromtimestamp(stat.st_mtime).year
    run = {"command": None, "run_name": None, "work_dir": None, "started": None, "ended": None, "last_line": None}
    tasks = {}      # hash -> task
    by_name = {}    # name -> hash (the last submission of a name: retries reuse it)
    last = ""
    with open(path, errors="replace") as fh:
        for line in fh:
            if "Submitted process >" in line or "Cached process >" in line:
                match = SUBMITTED_RE.search(line)
                if match:
                    h, kind, name = match.groups()
                    tasks[h] = {"hash": h, "name": name, "process": name.split(" (")[0],
                                "status": "cached" if kind == "Cached" else "running",
                                "submitted": _log_time(line, year), "ended": None, "exit": None}
                    by_name[name] = h
            elif "Task completed >" in line:
                match = COMPLETED_RE.search(line)
                if match:
                    name, status, exit_code, error, work_dir = match.groups()
                    h = by_name.get(name)
                    task = tasks.get(h) if h else None
                    if task is None:
                        continue
                    ok = status == "COMPLETED" and exit_code.strip() == "0"
                    task.update(status="completed" if ok else "failed", ended=_log_time(line, year),
                                exit=exit_code.strip(), work_dir=work_dir,
                                error=None if error.strip() in ("-", "") else error.strip())
            elif "nextflow.cli.Launcher" in line and run["command"] is None:
                match = LAUNCHER_RE.search(line)
                if match:
                    run["command"] = match.group(1).strip()
                    run["started"] = _log_time(line, year)
            elif "Run name:" in line and run["run_name"] is None:
                run["run_name"] = line.rsplit("Run name:", 1)[1].strip()
            elif "Work-dir:" in line and run["work_dir"] is None:
                run["work_dir"] = line.rsplit("Work-dir:", 1)[1].split(" [")[0].strip()
            elif "Execution complete -- Goodbye" in line:
                run["ended"] = _log_time(line, year)
            if line.strip():
                last = line
    run["last_line"] = _log_time(last, year) if tasks or run["command"] else None
    parsed = {"run": run, "tasks": list(tasks.values())}
    with _cache_lock:
        _log_cache[path] = (stat.st_mtime, stat.st_size, parsed)
    return parsed


def nextflow_runs(folder):
    """nextflow processes running in this folder: pid, started, args."""
    runs = []
    real = os.path.realpath(folder)
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = fh.read().decode(errors="replace").split("\0")
            if "nextflow.cli.Launcher" not in argv or os.path.realpath(f"/proc/{pid}/cwd") != real:
                continue
            started = os.stat(f"/proc/{pid}").st_mtime
        except (OSError, ValueError):
            continue
        args = argv[argv.index("nextflow.cli.Launcher") + 1:]
        args = [a for a in args if a]
        batch = args[args.index("--batch") + 1] if "--batch" in args[:-1] else None
        runs.append({"pid": int(pid), "args": " ".join(args), "batch": batch,
                     "script": args[1] if len(args) > 1 and args[0] == "run" else None,
                     "started": _process_start(pid) or started})
    return runs


def _process_start(pid):
    try:
        with open(f"/proc/{pid}/stat") as fh:
            ticks = int(fh.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/stat") as fh:
            boot = next(int(l.split()[1]) for l in fh if l.startswith("btime"))
        return boot + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, StopIteration):
        return None


# outputs of a fly in behavior_prediction, in the order the pipeline makes them
# (relative to the basedir; {e} experiment, {i} identity, {f} fly = {e}__{i})
FLY_OUTPUTS = [
    ("pose", "motionmapper/{i}/pose_raw/{f}/{f}.h5", "compiled pose (Pose tab)"),
    ("movie", "motionmapper/{i}/movie/movie.m3u8", "movie of the fly (Ethograms tab)"),
    ("pe", "flyhostel/proboscis_extensions/{f}_traces.feather", "proboscis extension traces (PE tab)"),
    ("behavior", "motionmapper/{i}/{f}.feather", "behavior predictions (Ethograms tab)"),
    ("sleep", "sleep/{i}/{f}_sleep5min.feather", "sleep"),
    ("ethogram", "motionmapper/{i}/{f}.json", "ethogram figure, published last"),
]


def fly_outputs(basedir, experiment, identity, since):
    """Outputs of a fly: stage -> {mtime, new} (new: written after `since`), None if missing,
    and its state: done (ethogram figure and sleep new), explorable (behavior predictions new:
    the Ethograms tab works), pending."""
    fly = f"{experiment}__{identity}"
    outputs = {}
    for stage, pattern, _ in FLY_OUTPUTS:
        try:
            mtime = os.path.getmtime(os.path.join(basedir, pattern.format(e=experiment, i=identity, f=fly)))
        except OSError:
            outputs[stage] = None
            continue
        outputs[stage] = {"mtime": mtime, "new": since is not None and mtime >= since}
    new = {stage for stage, o in outputs.items() if o and o["new"]}
    state = "done" if {"ethogram", "sleep"} <= new else "explorable" if "behavior" in new else "pending"
    return {"fly": fly, "outputs": outputs, "state": state}


def _batch_time(batch):
    """Launch time of a batch from its stamp (date, or date_HH-MM)."""
    for fmt in ("%Y-%m-%d_%H-%M", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(batch, fmt).timestamp()
        except (TypeError, ValueError):
            pass
    return None


def queue_runner(folder):
    """run_queue.sh processes running in this folder, and the end of run_queue.log."""
    script = os.path.join(folder, "run_queue.sh")
    if not os.path.exists(script):
        return None
    real = os.path.realpath(folder)
    pids = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = fh.read().decode(errors="replace").split("\0")
            if not any(a.endswith("run_queue.sh") for a in argv[:2]) or os.path.realpath(f"/proc/{pid}/cwd") != real:
                continue
        except OSError:
            continue
        pids.append(int(pid))
    log_path = os.path.join(folder, "run_queue.log")
    lines, log_mtime = [], None
    if os.path.exists(log_path):
        with open(log_path, errors="replace") as fh:
            lines = [l.rstrip("\n") for l in fh.readlines()[-QUEUE_LOG_LINES:]]
        log_mtime = os.path.getmtime(log_path)
    return {"pids": pids, "log": lines, "log_mtime": log_mtime}


def _experiment_name(spec, row):
    if spec["experiment"] is not None:
        return row[spec["experiment"]].strip()
    basedir = row[spec["basedir"]].strip().rstrip("/")
    try:
        return to_flat(os.path.relpath(basedir, FLYHOSTEL_VIDEOS))
    except Exception:
        return basedir


def read_rows(spec, folder):
    """Rows of the pipeline's CSV with experiment, basedir, identity, mark and status."""
    path = os.path.join(folder, spec["csv"])
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="", errors="replace") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if len(row) <= max(spec["basedir"], spec["mark"]):
                continue
            rows.append({
                "line": i + 1,
                "experiment": _experiment_name(spec, row),
                "basedir": row[spec["basedir"]].strip(),
                "identity": row[spec["identity"]].strip() if spec["identity"] is not None and len(row) > spec["identity"] else None,
                "mark": row[spec["mark"]].strip(),
                "note": ",".join(c.strip() for c in row[spec["status"]:] if c.strip()) if len(row) > spec["status"] else "",
            })
    return rows


def _attribute(task, work_root, candidates):
    """Experiment key of a task, from its .command.sh (longest candidate string found)."""
    work_dir = task.get("work_dir")
    if work_dir is None and work_root:
        matches = glob.glob(os.path.join(work_root, task["hash"] + "*"))
        work_dir = matches[0] if matches else None
    if work_dir is None:
        return None
    with _cache_lock:
        if work_dir in _task_experiment:
            return _task_experiment[work_dir]
    try:
        with open(os.path.join(work_dir, ".command.sh"), errors="replace") as fh:
            script = fh.read()
    except OSError:
        return None     # not written yet: try again next time
    best = None
    for needle, key in candidates:
        if needle and needle in script and (best is None or len(needle) > len(best[0])):
            best = (needle, key)
    key = best[1] if best else None
    with _cache_lock:
        _task_experiment[work_dir] = key
    return key


def _summary(tasks):
    counts = {"running": 0, "completed": 0, "cached": 0, "failed": 0}
    for t in tasks:
        counts[t["status"]] += 1
    return counts


def pipeline_status(spec):
    folder = os.path.join(PIPELINES_DIR, spec["folder"])
    if not os.path.isdir(folder):
        return {**_public(spec), "error": f"{folder} not found"}
    rows = read_rows(spec, folder)
    runs = nextflow_runs(folder)
    runner = queue_runner(folder)
    has_queue = runner is not None

    log_path = os.path.join(folder, ".nextflow.log")
    log = parse_log(log_path) if os.path.exists(log_path) else None

    # the run the log belongs to: the running one launched last
    run = None
    if runs:
        run = max(runs, key=lambda r: r["started"])
        batch = run["batch"] or "1"
        # launched by hand without --batch, then stamped with its launch time (date_HH-MM)
        if run["batch"] is None and has_queue:
            stamp = datetime.datetime.fromtimestamp(run["started"]).strftime("%Y-%m-%d_%H-%M")
            if any(r["mark"] == stamp for r in rows):
                batch = stamp
        run["batch"] = batch
    elif log and log["run"]["command"]:
        args = log["run"]["command"].split()
        batch = args[args.index("--batch") + 1] if "--batch" in args[:-1] else None
        run = {"pid": None, "args": log["run"]["command"].replace("nextflow ", "", 1), "batch": batch,
               "started": log["run"]["started"]}

    in_run = [r for r in rows if run and run["batch"] and r["mark"] == run["batch"]] if run and run["pid"] else []
    queued = [r for r in rows if r["mark"] == "1"] if has_queue or not (run and run["pid"]) else []

    if run is not None and log is not None:
        run.update(run_name=log["run"]["run_name"], log_started=log["run"]["started"], ended=log["run"]["ended"],
                   last_activity=log["run"]["last_line"], work_dir=log["run"]["work_dir"])
        # the log is of an older run if the running process started after its last line
        if run["pid"] and log["run"]["started"] and log["run"]["started"] < run["started"] - 120:
            run["log_stale"] = True
        tasks = [] if run.get("log_stale") else log["tasks"]
        work_root = log["run"]["work_dir"]
        if work_root and not os.path.isabs(work_root):
            work_root = os.path.join(folder, work_root)
        # experiments the tasks are looked up in: the run's rows, else every row (a run whose rows
        # lost their stamp, or one without batch)
        members = in_run or ([r for r in rows if r["mark"] == run["batch"]] if run["batch"] else []) or rows
        candidates = []
        for r in members:
            candidates += [(r["experiment"], r["experiment"]), (r["basedir"].rstrip("/"), r["experiment"])]
        per_experiment = {}
        running = []
        for t in tasks:
            key = _attribute(t, work_root, candidates) if candidates else None
            per_experiment.setdefault(key, []).append(t)
            if t["status"] == "running":
                running.append({**{k: t[k] for k in ("hash", "name", "process", "submitted")}, "experiment": key})
        processes = {}
        for t in tasks:
            processes.setdefault(t["process"], []).append(t)
        run["summary"] = _summary(tasks)
        run["processes"] = [{"process": p, **_summary(ts)} for p, ts in processes.items()]
        run["running_tasks"] = sorted(running, key=lambda t: t["submitted"] or 0)[:MAX_RUNNING_TASKS]
        run["n_running_tasks"] = len(running)
        run["experiments"] = {
            (k or ""): {"summary": _summary(ts),
                        "processes": [{"process": p, **_summary([t for t in ts if t["process"] == p])}
                                      for p in dict.fromkeys(t["process"] for t in ts)],
                        "failed": [t["name"] for t in ts if t["status"] == "failed"][:20]}
            for k, ts in per_experiment.items()
        }
        run["failed_tasks"] = [{"name": t["name"], "exit": t["exit"], "work_dir": t.get("work_dir")}
                               for t in tasks if t["status"] == "failed"][:50]

        # no row holds the running batch's stamp (the CSV was saved over by an editor holding
        # the old version, ...): rows marked 1 with tasks in the run are being processed, and
        # run_queue.sh will take them again in its next round
        if run["pid"] and not in_run and has_queue:
            active = {k for k in per_experiment if k}
            in_run = [{**r, "unstamped": True} for r in queued if r["experiment"] in active]
            queued = [r for r in queued if r["experiment"] not in active]

    # completed/<batch>.txt (pose_estimation): experiments that went through the whole pipeline
    def completed(batch):
        path = os.path.join(folder, "completed", f"{batch}.txt")
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return [l.strip() for l in fh if l.strip()]

    batches = {}
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    for r in rows:
        if BATCH_RE.match(r["mark"]) and r["mark"] < tomorrow:     # typos such as 2206-06-05 left out
            batches.setdefault(r["mark"], []).append(r)
    recent = [{"batch": b, "rows": _rows_public(batches[b]), "completed": completed(b)}
              for b in sorted(batches, reverse=True)[:RECENT_BATCHES]]

    # per fly: what is on disk, compared with the start of its batch
    if spec["identity"] is not None:
        def annotate(public_rows, since):
            for r in public_rows:
                if r["identity"]:
                    r.update(fly_outputs(r["basedir"], r["experiment"], r["identity"], since))
        in_run_public = _rows_public(in_run)
        annotate(in_run_public, run["started"] if run and run["pid"] else None)
        for b in recent:
            since = run["started"] if run and run["pid"] and b["batch"] == run["batch"] else _batch_time(b["batch"])
            annotate(b["rows"], since)
        queued_public = _rows_public(queued)
        annotate(queued_public, None)
    else:
        in_run_public, queued_public = _rows_public(in_run), _rows_public(queued)
    if run is not None and run.get("batch"):
        run["completed"] = completed(run["batch"])

    return {
        **_public(spec),
        "folder": folder,
        "csv_path": os.path.join(folder, spec["csv"]),
        "has_queue": has_queue,
        "queue_runner": runner,
        "running": bool(runs),
        "runs": [{k: r[k] for k in ("pid", "args", "batch", "started")} for r in runs],
        "run": run,
        "in_run": in_run_public,
        "queued": queued_public,
        "fly_outputs": [{"stage": stage, "path": pattern, "description": d} for stage, pattern, d in FLY_OUTPUTS]
        if spec["identity"] is not None else None,
        "recent_batches": recent,
    }


def _public(spec):
    return {"id": spec["id"], "label": spec["label"], "per_fly": spec["identity"] is not None}


def _rows_public(rows):
    return [{**{k: r[k] for k in ("line", "experiment", "basedir", "identity", "mark", "note")},
             "unstamped": r.get("unstamped", False)} for r in rows]


def register_monitor(app):

    @app.route("/api/monitor", methods=["GET"])
    def monitor():
        result = []
        for spec in PIPELINES:
            try:
                result.append(pipeline_status(spec))
            except Exception as error:
                logger.exception("monitor: %s", spec["id"])
                result.append({**_public(spec), "error": str(error)})
        return jsonify({"pipelines": result, "pipelines_dir": PIPELINES_DIR, "now": time.time()})
