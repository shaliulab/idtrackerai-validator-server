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

Relaunching failed tasks (pipelines with a queue): the profiles use errorStrategy
"ignore", so a run ends OK even when tasks failed, and their fly / experiment stops
there. `relaunch` gathers, over the rotated logs (.nextflow.log, .nextflow.log.1, ...),
the latest run of every batch with failed tasks whose rows still hold that batch (a row
re-queued gets a new stamp), with the end of each task's .command.err, and two ways of
running them again, both resuming the nextflow session so completed tasks are cached:
re-queue the rows (mark them 1 again: run_queue.sh takes them in its next round, as a
new batch; POST /api/monitor/<pipeline>/requeue does it, `requeue` is the same as a
shell command) or run the batch again by hand with -resume <session> (not started from
here: a run started by the server would die with it).
"""
import shlex
import os
import re
import csv
import glob
import time
import logging
import datetime
import threading

import fcntl
import signal

from flask import jsonify, request

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
    run = {"command": None, "run_name": None, "session": None, "work_dir": None, "started": None, "ended": None,
           "last_line": None}
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
            elif "Session UUID:" in line and run["session"] is None:
                run["session"] = line.rsplit("Session UUID:", 1)[1].strip()
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
    sleeping = _queue_sleep(pids)
    lines, log_mtime = [], None
    if os.path.exists(log_path):
        with open(log_path, errors="replace") as fh:
            lines = [l.rstrip("\n") for l in fh.readlines()[-QUEUE_LOG_LINES:]]
        log_mtime = os.path.getmtime(log_path)
    return {"pids": pids, "log": lines, "log_mtime": log_mtime,
            # polling: when run_queue.sh looks at the CSV again (null while it waits for a run)
            "next_check": sleeping["until"] if sleeping else None}


def _queue_sleep(runner_pids):
    """The `sleep N` run_queue.sh polls with (a child of one of runner_pids): pid and when it ends."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as fh:
                ppid = int(fh.read().rsplit(")", 1)[1].split()[1])
            if ppid not in runner_pids:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = [a for a in fh.read().decode(errors="replace").split("\0") if a]
        except (OSError, ValueError, IndexError):
            continue
        if len(argv) == 2 and os.path.basename(argv[0]) == "sleep" and argv[1].isdigit():
            started = _process_start(pid)
            return {"pid": int(pid), "until": started + int(argv[1]) if started else None}
    return None


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


# lines of a failed task's .command.err kept
ERROR_LINES = 15
# a run's failed tasks listed in `relaunch`
MAX_FAILED = 100
IDENTITY_RE = re.compile(r"--identity[ =]+(\S+)")
EXCEPTION_RE = re.compile(r"^(\w+\.)*\w*(Error|Exception|Interrupt)\b")
FRAME_RE = re.compile(r'^\s*File "([^"]+)", line \d+')
SAFE_RE = re.compile(r"^[\w.\-]+$")
_task_details = {}      # work dir -> {"identity", "error"}


def _task_details_of(work_dir):
    """Identity in a finished task's .command.sh (--identity) and the end of its .command.err
    (else .command.log), cached: a finished task's files do not change."""
    with _cache_lock:
        if work_dir in _task_details:
            return _task_details[work_dir]
    details = {"identity": None, "error": [], "exception": None, "where": None}
    try:
        with open(os.path.join(work_dir, ".command.sh"), errors="replace") as fh:
            match = IDENTITY_RE.search(fh.read())
            details["identity"] = match.group(1).strip("'\"") if match else None
    except OSError:
        pass
    for name in (".command.err", ".command.log"):
        try:
            with open(os.path.join(work_dir, name), errors="replace") as fh:
                lines = [l.rstrip() for l in fh.readlines()[-200:] if l.strip()]
        except OSError:
            continue
        if lines:
            details["error"] = lines[-ERROR_LINES:]
            # the exception raised, and the deepest frame outside site-packages (where to look)
            # with the lines after it: multi-line messages (ethoscopy names the flies at fault there)
            k = next((k for k in range(len(lines) - 1, -1, -1) if EXCEPTION_RE.match(lines[k].strip())), None)
            details["exception"] = "\n".join(l.rstrip() for l in lines[k:k + 6]) if k is not None else None
            for i in range(len(lines) - 1, -1, -1):
                match = FRAME_RE.match(lines[i])
                if match and "site-packages" not in match.group(1):
                    details["where"] = [lines[i].strip()] + ([lines[i + 1].strip()] if i + 1 < len(lines) else [])
                    break
            break
    with _cache_lock:
        _task_details[work_dir] = details
    return details


def _resume_args(command, session):
    """The run's nextflow arguments with -resume <session> (in place of a bare -resume)."""
    args = shlex.split(command)
    if args and args[0] == "nextflow":
        args = args[1:]
    if "-resume" in args:
        i = args.index("-resume")
        if i + 1 < len(args) and not args[i + 1].startswith("-"):
            args[i + 1] = session
        else:
            args.insert(i + 1, session)
    else:
        args += ["-resume", session]
    return args


def relaunch_info(spec, folder, rows, live):
    """Failed tasks still to be finished, per batch, and the commands to run them again."""
    logs = sorted(glob.glob(os.path.join(folder, ".nextflow.log*")),
                  key=lambda p: int(p.rsplit(".", 1)[1]) if p.rsplit(".", 1)[1].isdigit() else 0)
    latest = {}     # batch -> parsed log of its latest run
    for path in logs:
        try:
            parsed = parse_log(path)
        except OSError:
            continue
        command = parsed["run"]["command"]
        if not command or "-preview" in command.split():
            continue
        args = command.split()
        batch = args[args.index("--batch") + 1] if "--batch" in args[:-1] else None
        if batch is None or batch == "1":
            continue    # its rows cannot be told apart any more
        if batch not in latest or (parsed["run"]["started"] or 0) > (latest[batch]["run"]["started"] or 0):
            latest[batch] = parsed

    per_fly = spec["identity"] is not None
    batches = []
    for batch, parsed in sorted(latest.items(), reverse=True):
        failed = [t for t in parsed["tasks"] if t["status"] == "failed"]
        if not failed:
            continue
        # a run still going on: its failures are shown by the run itself
        if live and live.get("batch") == batch:
            continue
        members = [r for r in rows if r["mark"] == batch]
        if not members:
            continue    # every row of the batch was re-queued since
        candidates = []
        for r in members:
            candidates += [(r["experiment"], r["experiment"]), (r["basedir"].rstrip("/"), r["experiment"])]
        work_root = parsed["run"]["work_dir"]
        if work_root and not os.path.isabs(work_root):
            work_root = os.path.join(folder, work_root)
        tasks, flies = [], {}
        for t in failed[:MAX_FAILED]:
            experiment = _attribute(t, work_root, candidates)
            details = _task_details_of(t["work_dir"]) if t.get("work_dir") else \
                {"identity": None, "error": [], "exception": None, "where": None}
            identity = details["identity"] if per_fly else None
            # the rows the task belongs to
            targets = [r for r in members if r["experiment"] == experiment
                       and (identity is None or r["identity"] == identity)] if experiment else []
            for r in targets:
                key = (r["experiment"], r["identity"])
                flies.setdefault(key, {"experiment": r["experiment"], "identity": r["identity"], "line": r["line"],
                                       "tasks": []})["tasks"].append(t["name"])
            tasks.append({"name": t["name"], "process": t["process"], "exit": t["exit"], "work_dir": t.get("work_dir"),
                          "experiment": experiment, "identity": identity, "error": details["error"],
                          "exception": details["exception"], "where": details["where"]})
        if not flies:
            continue
        errors = {t["exception"] or t["error"][-1] for t in tasks if t["error"]}
        batches.append({
            "batch": batch,
            "run_name": parsed["run"]["run_name"],
            "session": parsed["run"]["session"],
            "started": parsed["run"]["started"],
            "ended": parsed["run"]["ended"],
            "n_failed": len(failed),
            "tasks": tasks,
            "flies": sorted(flies.values(), key=lambda f: (f["experiment"], f["identity"] or "")),
            # every failure ends with the same line: probably one cause, a rerun alone will not help
            "same_error": next(iter(errors)) if len(errors) == 1 and len(tasks) > 1 else None,
            "command": (f"cd {shlex.quote(folder)} && nohup nextflow "
                        f"{shlex.join(_resume_args(parsed['run']['command'], parsed['run']['session']))}"
                        f" > nextflow_run_{batch}_retry.out 2>&1 &") if parsed["run"]["session"] else None,
        })

    flies = [f for b in batches for f in b["flies"]]
    requeue = None
    if flies and all(SAFE_RE.match(f["experiment"]) and (f["identity"] is None or SAFE_RE.match(f["identity"]))
                     for f in flies):
        e, m = spec["experiment"] + 1 if spec["experiment"] is not None else None, spec["mark"] + 1
        conditions = []
        for b in batches:
            for f in b["flies"]:
                c = [f'x=="{b["batch"]}"']
                c.append(f'${e}=="{f["experiment"]}"' if e else f'NR=={f["line"]}')
                if per_fly and f["identity"] is not None:
                    c.append(f'${spec["identity"] + 1}=="{f["identity"]}"')
                conditions.append("(" + " && ".join(c) + ")")
        program = f'{{x=${m}; gsub(/ /, "", x)}} {" || ".join(conditions)} {{${m}=1}} 1'
        csv_name = spec["csv"]
        # as run_queue.sh: under flock, written in place
        requeue = (f"cd {shlex.quote(folder)} && ( flock 9; awk -F, -v OFS=, {shlex.quote(program)} {csv_name}"
                   f" > {csv_name}.tmp && cat {csv_name}.tmp > {csv_name} && rm {csv_name}.tmp ) 9>>{csv_name}")
    return {"batches": batches, "requeue": requeue, "live": bool(live)} if batches else None


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
                        "failed": [{"name": t["name"], "process": t["process"], "exit": t["exit"],
                                    "work_dir": t.get("work_dir")}
                                   for t in ts if t["status"] == "failed"][:20]}
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
    relaunch = relaunch_info(spec, folder, rows, run if run and run["pid"] else None) if has_queue or os.path.exists(
        os.path.join(folder, "run_queue.sh")) else None

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
        "relaunch": relaunch,
    }


def _public(spec):
    return {"id": spec["id"], "label": spec["label"], "per_fly": spec["identity"] is not None}


def _rows_public(rows):
    return [{**{k: r[k] for k in ("line", "experiment", "basedir", "identity", "mark", "note")},
             "unstamped": r.get("unstamped", False)} for r in rows]


def requeue_rows(spec, wanted):
    """Mark the rows (batch, experiment, identity) of the pipeline's CSV 1 again, as the
    `requeue` command does: under flock on the CSV (run_queue.sh and pose_estimation's
    run_queue.sh take the same lock), written in place. Returns the CSV line numbers changed."""
    path = os.path.join(PIPELINES_DIR, spec["folder"], spec["csv"])
    m, e = spec["mark"], spec["experiment"]
    i = spec["identity"]
    changed = []
    with open(path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with open(path, newline="", errors="replace") as fh:
                lines = fh.read().split("\n")
            for n, line in enumerate(lines):
                fields = line.split(",")
                if len(fields) <= m:
                    continue
                row = {"experiment": fields[e].strip() if e is not None and len(fields) > e else None,
                       "basedir": fields[spec["basedir"]].strip() if len(fields) > spec["basedir"] else "",
                       "identity": fields[i].strip() if i is not None and len(fields) > i else None}
                if row["experiment"] is None:
                    row["experiment"] = _experiment_name(spec, fields)
                key = (fields[m].replace(" ", ""), row["experiment"], row["identity"])
                if key in wanted:
                    fields[m] = "1"
                    lines[n] = ",".join(fields)
                    changed.append(n + 1)
            if changed:
                with open(path, "w", newline="") as fh:    # in place, as run_queue.sh (cat tmp > csv)
                    fh.write("\n".join(lines))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return changed


def register_monitor(app):

    @app.route("/api/monitor/<pipeline>/check_now", methods=["POST"])
    def monitor_check_now(pipeline):
        """End the `sleep` run_queue.sh polls with, so it looks at the CSV now: its loop just goes
        on to the next check (nothing to do while it waits for a run to end)."""
        spec = next((s for s in PIPELINES if s["id"] == pipeline), None)
        if spec is None:
            return jsonify({"error": f"unknown pipeline {pipeline}"}), 404
        runner = queue_runner(os.path.join(PIPELINES_DIR, spec["folder"]))
        if not runner or not runner["pids"]:
            return jsonify({"error": "run_queue.sh is not running"}), 409
        sleeping = _queue_sleep(runner["pids"])
        if sleeping is None:
            return jsonify({"error": "run_queue.sh is not polling (it waits for a nextflow run to end)"}), 409
        try:
            os.kill(sleeping["pid"], signal.SIGTERM)
        except OSError as error:
            return jsonify({"error": f"could not wake run_queue.sh: {error}"}), 500
        logger.info("monitor: woke run_queue.sh of %s (sleep %s)", pipeline, sleeping["pid"])
        return jsonify({"ok": True})

    @app.route("/api/monitor/<pipeline>/requeue", methods=["POST"])
    def monitor_requeue(pipeline):
        """{"batches": [batch, ...]} (default: all): mark 1 again the rows of the failed flies
        listed by relaunch_info for those batches, so run_queue.sh takes them in its next round."""
        spec = next((s for s in PIPELINES if s["id"] == pipeline), None)
        if spec is None:
            return jsonify({"error": f"unknown pipeline {pipeline}"}), 404
        status = pipeline_status(spec)
        relaunch = status.get("relaunch")
        if not relaunch:
            return jsonify({"error": "no failed tasks to re-queue"}), 400
        body = request.get_json(silent=True) or {}
        batches = body.get("batches")
        wanted = {(b["batch"], f["experiment"], f["identity"] if spec["identity"] is not None else None)
                  for b in relaunch["batches"] if batches is None or b["batch"] in batches for f in b["flies"]}
        if not wanted:
            return jsonify({"error": "nothing to re-queue for those batches"}), 400
        try:
            changed = requeue_rows(spec, wanted)
        except OSError as error:
            return jsonify({"error": f"could not edit {spec['csv']}: {error}"}), 500
        logger.info("monitor: re-queued lines %s of %s", changed, spec["csv"])
        runner = status.get("queue_runner") or {}
        return jsonify({"lines": changed, "rows": len(changed), "queue_running": bool(runner.get("pids")),
                        "next_check": runner.get("next_check"), "csv": spec["csv"]})

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
