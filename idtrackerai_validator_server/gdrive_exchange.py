"""
gdrive_exchange.py  —  experiments and their annotations through a shared Google Drive folder,
so people running their own instance of the server annotate (Annotate tab) by themselves.

Transport: rclone (as dropbox.py), with a Google Drive remote each person configures once
(`rclone config`; the token stays in rclone's config, the server never sees credentials).
The remote is $GDRIVE_REMOTE (default `gdrive:FlyHostelExchange`), or what the Drive
exchange tab saved in $EXCHANGE_CONFIG (default ~/.config/idtrackerai-validator/exchange.json,
also holding the annotator's name). One folder shared with every annotator (Editor):
annotators point their remote at it (rclone's `root_folder_id`, or a shortcut in My Drive).

Layout, one folder per experiment (flat name):

    <flat>/manifest.json                  written last by the publisher: the publish is complete
    <flat>/base/validator_annotations.db   the publisher's annotations at publish time (snapshot)
    <flat>/full/...                        kind "full": the experiment folder (the live annotations
                                           database and exchange.json left out)
    <flat>/minimal/...                     kind "minimal": the bundle of exchange_bundle.py (scene clips,
                                           cut-down database, contours, scenes.json), laid out as an
                                           experiment folder; built in $EXCHANGE_CACHE/bundles/<flat>/ and
                                           synced (--checksum: unchanged clips are not sent again)
    <flat>/annotations/<name>/validator_annotations.db, meta.json   each annotator's copy

Roles (`<basedir>/flyhostel/validation/exchange.json`): the publisher (publishes, imports
the annotators' copies) and annotators (download, upload their copy, pull the publisher's
newer snapshots). Merging is annotation_merge.py, by log rowid: the manifest gives the
snapshot's rowid (`snapshot_rowid`), an annotator records where their downloaded copy ended
(`base_rowid`, their own work comes after it) and the last snapshot merged (`pulled_rowid`,
in the publisher's numbering); meta.json carries both to the publisher.

An annotator's copy keeps the kind it was downloaded as (exchange.json `kind`): a pull refreshes
full copies' PULL_FILES from full/, and minimal copies from the whole minimal/ bundle (new scenes
and their clips come with it).

Routes (/api/exchange/*): status + config, experiments on Drive, local state of an
experiment, and jobs: publish, download, upload (annotator), import (publisher: plan of an
annotator's copy) / pull (annotator: plan of the publisher's snapshot, plus its scenes and
proposals), then apply a plan with a conflict rule.
"""
import os
import re
import json
import time
import shutil
import socket
import logging
import tempfile
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor

from flask import jsonify, request, session

from idtrackerai_validator_server import annotation_merge as merge
from idtrackerai_validator_server import exchange_bundle
from idtrackerai_validator_server import box_annotations as boxes_mod
from idtrackerai_validator_server.experiments import to_flat, to_relpath
from idtrackerai_validator_server.jobs import start_job, _jobs
from idtrackerai_validator_server.restore import _rclone_copy, _add_to_index

logger = logging.getLogger(__name__)

RCLONE = os.environ.get("RCLONE", "rclone")
DEFAULT_REMOTE = os.environ.get("GDRIVE_REMOTE", "gdrive:FlyHostelExchange")
CONFIG_FILE = os.path.expanduser(os.environ.get("EXCHANGE_CONFIG", "~/.config/idtrackerai-validator/exchange.json"))
CACHE_DIR = os.path.expanduser(os.environ.get("EXCHANGE_CACHE", "~/.cache/idtrackerai-validator/exchange"))
# free space a download leaves on the drive
DOWNLOAD_MARGIN = float(os.environ.get("EXCHANGE_DOWNLOAD_MARGIN_GB", "5")) * 1e9
FORMAT = 1
KINDS = ("full", "minimal")
PUBLISHER = "publisher"            # the publisher's label in the annotators' databases
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
LOCAL_STATE = "exchange.json"
# never sent to Drive with the folder: the live annotations database (sent as a snapshot) and the local state
FULL_EXCLUDES = ["flyhostel/validation/validator_annotations.db", "flyhostel/validation/validator_annotations.db-*",
                 f"flyhostel/validation/{LOCAL_STATE}", "*.db-journal", ".downloading/**"]
# what a pull refreshes from full/: the scenes and the integration's problem frames
PULL_FILES = ["flyhostel/validation/scene_qc.csv", "flyhostel/validation/validator_problem_frames.csv"]
# what an annotations-only publish of a minimal bundle refreshes (its scenes are fixed in the bundle)
PULL_FILES_MINIMAL = ["flyhostel/validation/validator_problem_frames.csv"]

_busy = {}                  # flat name -> job id of the transfer running on it
_busy_lock = threading.Lock()
_plans = {}                 # job id -> (flat, plan, kind "import" | "pull", extra)
_listing = {"at": None, "experiments": [], "error": None}


# ---------------------------------------------------------------- configuration

def read_config():
    try:
        with open(CONFIG_FILE) as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        config = {}
    return {"remote": config.get("remote") or DEFAULT_REMOTE,
            "annotator": config.get("annotator") or os.environ.get("ANNOTATOR_NAME") or ""}


def write_config(values):
    config = read_config()
    for key in ("remote", "annotator"):
        if key in values:
            config[key] = (values[key] or "").strip()
    if config["annotator"] and not NAME_RE.match(config["annotator"]):
        raise ValueError("annotator name: letters, digits, _ . - (up to 64)")
    if config["remote"] and ":" not in config["remote"]:
        raise ValueError("remote: an rclone remote, e.g. gdrive:FlyHostelExchange")
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    tmp = CONFIG_FILE + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(config, handle, indent=1)
    os.replace(tmp, CONFIG_FILE)
    return config


def annotator_name():
    name = read_config()["annotator"] or session.get("user") or os.environ.get("USER") or ""
    return name if NAME_RE.match(name) else ""


# ---------------------------------------------------------------- rclone

def _rclone(*args, input=None, timeout=120):
    proc = subprocess.run([RCLONE, *args], input=input, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        lines = [line for line in proc.stderr.strip().splitlines() if line.strip()]
        raise RuntimeError(f"rclone {args[0]} failed: {lines[-1] if lines else proc.returncode}")
    return proc.stdout


def _remote(*parts):
    base = read_config()["remote"].rstrip("/")
    return "/".join([base, *[p.strip("/") for p in parts if p]])


def _cat_json(path, timeout=60):
    try:
        return json.loads(_rclone("cat", path, timeout=timeout))
    except (RuntimeError, ValueError):
        return None


def _put_json(path, value):
    _rclone("rcat", path, input=json.dumps(value, indent=1), timeout=120)


def _lsjson(path, *args, timeout=300):
    try:
        return json.loads(_rclone("lsjson", *args, path, timeout=timeout))
    except RuntimeError as error:
        if "directory not found" in str(error).lower() or "not found" in str(error).lower():
            return None
        raise


def remote_status():
    """rclone installed, the remote configured and reachable"""
    config = read_config()
    out = {"remote": config["remote"], "annotator": config["annotator"], "rclone": shutil.which(RCLONE) is not None,
           "configured": False, "reachable": False, "error": None, "config_file": CONFIG_FILE}
    if not out["rclone"]:
        out["error"] = f"{RCLONE} not found: install rclone (https://rclone.org/install/)"
        return out
    try:
        remotes = _rclone("listremotes", timeout=20).split()
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        out["error"] = str(error)
        return out
    name = config["remote"].split(":")[0] + ":"
    out["configured"] = name in remotes
    if not out["configured"]:
        out["error"] = f"no rclone remote {name} (run `rclone config` and add a Google Drive remote named {name[:-1]})"
        return out
    try:
        _rclone("lsf", "--max-depth", "1", "--dirs-only", config["remote"], timeout=60)
        out["reachable"] = True
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        msg = str(error)
        if "directory not found" in msg.lower():
            out["reachable"], out["error"] = True, None     # made on first publish
            out["empty"] = True
        else:
            out["error"] = msg
    return out


# ---------------------------------------------------------------- local state

def _root():
    return os.environ["FLYHOSTEL_VIDEOS"]


def basedir_of(experiment):
    return os.path.join(_root(), to_relpath(experiment))


def _validation(basedir):
    return os.path.join(basedir, "flyhostel", "validation")


def read_local(basedir):
    try:
        with open(os.path.join(_validation(basedir), LOCAL_STATE)) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def write_local(basedir, state):
    folder = _validation(basedir)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, LOCAL_STATE)
    with open(path + ".tmp", "w") as handle:
        json.dump(state, handle, indent=1)
    os.replace(path + ".tmp", path)


def _excludes(basedir):
    """rclone --exclude patterns of a full publish: FULL_EXCLUDES, and the arenas of a whole recording (each arena
    is an experiment of its own, published by itself)"""
    return FULL_EXCLUDES + (["/arenas/**"] if os.path.isdir(os.path.join(basedir, "arenas")) else [])


def _folder_files(basedir):
    """{relpath: size} of what a full publish sends"""
    import fnmatch
    patterns = [p.lstrip("/") for p in _excludes(basedir)]
    out = {}
    for top, dirs, files in os.walk(basedir):
        if top == basedir:
            dirs[:] = [d for d in dirs if d != ".downloading" and not (d == "arenas" and "arenas/**" in patterns)]
        for name in files:
            path = os.path.join(top, name)
            rel = os.path.relpath(path, basedir)
            if any(fnmatch.fnmatch(rel, pattern) for pattern in patterns):
                continue
            try:
                out[rel] = os.stat(path).st_size
            except OSError:
                pass
    return out


def _tree_files(folder):
    """{relpath: size} of every file under folder"""
    out = {}
    for top, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(top, name)
            out[os.path.relpath(path, folder)] = os.path.getsize(path)
    return out


def bundle_dir(flat):
    return os.path.join(_cache(), "bundles", flat)


def generate_dbfile(flat):
    return os.path.join(basedir_of(flat), to_flat(flat) + ".db")


def _databases(files):
    return sorted(p for p in files if "/" not in p and p.endswith("db") and p != "index.db" and files[p] > 0)


def local_info(experiment):
    basedir = basedir_of(experiment)
    out = {"experiment": to_flat(experiment), "relpath": to_relpath(experiment), "exists": os.path.isdir(basedir),
           "exchange": None, "annotations": None, "scene_qc": False}
    if not out["exists"]:
        return out
    out["exchange"] = read_local(basedir)
    sidecar = boxes_mod.sidecar_path(basedir)
    if os.path.exists(sidecar):
        out["annotations"] = merge.summary(sidecar)
    out["scene_qc"] = os.path.exists(os.path.join(_validation(basedir), "scene_qc.csv"))
    bundle = exchange_bundle.bundle_info(basedir)
    out["bundle"] = None if bundle is None else {k: bundle.get(k) for k in ("n_scenes", "frames", "created_at", "has_contours")}
    return out


# ---------------------------------------------------------------- listing Drive

def _experiment_entry(flat):
    manifest = _cat_json(_remote(flat, "manifest.json"))
    annotators = []
    for entry in _lsjson(_remote(flat, "annotations"), "--dirs-only", timeout=120) or []:
        meta = _cat_json(_remote(flat, "annotations", entry["Name"], "meta.json")) or {}
        annotators.append({"name": entry["Name"], **meta})
    local = local_info(flat)
    return {"experiment": flat, "manifest": manifest, "annotators": annotators,
            "local": {"exists": local["exists"], "exchange": local["exchange"]}}


def list_drive():
    entries = _lsjson(read_config()["remote"], "--dirs-only", timeout=300)
    names = sorted(e["Name"] for e in entries or [] if e["Name"].startswith("FlyHostel"))
    with ThreadPoolExecutor(8) as pool:
        experiments = list(pool.map(_experiment_entry, names))
    _listing.update(at=time.time(), experiments=experiments, error=None)
    return experiments


# ---------------------------------------------------------------- jobs

def _claim(flat):
    """Reserve the experiment for a transfer: None, or the job id (or "starting") of the one running"""
    with _busy_lock:
        running = _busy.get(flat)
        if running == "starting" or (running in _jobs and _jobs[running].status == "running"):
            return running
        _busy[flat] = "starting"
        return None


def _release(flat, job_id):
    with _busy_lock:
        if _busy.get(flat) == job_id:
            del _busy[flat]


def _snapshot_sidecar(basedir, dst):
    """Snapshot of the experiment's annotations database (an empty one if it has none) -> max log rowid"""
    sidecar = boxes_mod.sidecar_path(basedir)
    if not os.path.exists(sidecar):
        merge.connect_path(dst).close()
        return 0
    return merge.snapshot(sidecar, dst, lock=boxes_mod._lock)


def publish(job, experiment, kind, annotations_only, publisher, commit):
    flat, basedir = to_flat(experiment), basedir_of(experiment)
    state = read_local(basedir) or {}
    previous = _cat_json(_remote(flat, "manifest.json"))
    if previous and previous.get("kind") != kind and annotations_only:
        raise RuntimeError(f"published on Drive as {previous.get('kind')}: publish it again whole to change the kind")

    # 1. snapshot of the annotations
    job.update(0, status="running")
    workdir = tempfile.mkdtemp(prefix="exchange_", dir=_cache())
    try:
        base = os.path.join(workdir, boxes_mod.DB_NAME)
        # with its import bookkeeping: an annotator pulling it recognizes their own work coming back
        snapshot_rowid = _snapshot_sidecar(basedir, base)
        job.update(0, status="done", progress=1.0)

        # 2. the folder or the bundle (or, annotations only, the validation files a pull refreshes)
        job.update(1, status="running")
        bundle = None
        if annotations_only:
            if not previous:
                raise RuntimeError("not on Drive yet: publish it first")
            files = (previous or {}).get("files") or {}
            total = (previous or {}).get("bytes", 0)
            bundle = (previous or {}).get("bundle")
            for rel in (PULL_FILES if kind == "full" else PULL_FILES_MINIMAL):
                if os.path.exists(os.path.join(basedir, rel)):
                    _rclone("copyto", os.path.join(basedir, rel), _remote(flat, kind, rel), timeout=600)
        elif kind == "minimal":
            # build (70 %), then sync: clips of chunks without scenes any more go
            folder = bundle_dir(flat)
            info = exchange_bundle.export(basedir, folder, lambda f, text=None: job.update(1, progress=0.7 * f))
            bundle = {k: info.get(k) for k in ("n_scenes", "frames", "first_frame", "has_contours", "scene_source")}
            files = _tree_files(folder)
            total = sum(files.values())
            _rclone_copy(folder, _remote(flat, "minimal"),
                         lambda done: job.update(1, progress=min(0.99, 0.7 + 0.3 * done / max(1, total))),
                         extra=["--checksum"], command="sync")
        else:
            files = _folder_files(basedir)
            total = sum(files.values())
            extra = [arg for pattern in _excludes(basedir) for arg in ("--exclude", pattern)]
            _rclone_copy(basedir, _remote(flat, "full"),
                         lambda done: job.update(1, progress=min(0.99, done / max(1, total))), extra=extra)
        job.update(1, status="done", progress=1.0)

        # 3. base snapshot, then the manifest (last: a manifest means a complete publish)
        job.update(2, status="running")
        _rclone("copyto", base, _remote(flat, "base", boxes_mod.DB_NAME), timeout=1800)
        now = time.time()
        manifest = {
            "format": FORMAT, "experiment": flat, "relpath": to_relpath(experiment), "kind": kind,
            "published_at": now, "first_published_at": (previous or {}).get("first_published_at") or now,
            "publisher": publisher, "host": socket.gethostname(), "app_commit": commit,
            "snapshot_rowid": snapshot_rowid, "annotated_frames": merge.summary(base)["frames"],
            "bytes": total, "n_files": len(files), "databases": _databases(files),
            "files": files, "bundle": bundle,
        }
        _put_json(_remote(flat, "manifest.json"), manifest)
        job.update(2, status="done", progress=1.0)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    state.update(role="publisher", experiment=flat, remote=read_config()["remote"], kind=kind,
                 published_at=now, snapshot_rowid=snapshot_rowid)
    write_local(basedir, state)
    job.result = {"experiment": flat, "kind": kind, "bytes": total, "n_files": len(files),
                  "snapshot_rowid": snapshot_rowid, "annotations_only": annotations_only}


def download(job, experiment):
    flat = to_flat(experiment)
    manifest = _cat_json(_remote(flat, "manifest.json"))
    if manifest is None:
        raise RuntimeError(f"{flat} has no manifest on Drive (not published, or still being published)")
    kind = manifest.get("kind")
    if kind not in KINDS:
        raise RuntimeError(f"{flat} is published as {kind}, which this version of the app does not know: update it")
    relpath = manifest.get("relpath") or to_relpath(flat)
    root = _root()
    final = os.path.join(root, relpath)
    if os.path.lexists(final):
        raise RuntimeError(f"{final} exists already (pull updates instead)")
    hostel, rest = relpath.split("/", 1)
    tmp = os.path.join(root, hostel, ".downloading", rest)
    os.makedirs(tmp, exist_ok=True)

    files = manifest.get("files") or {}
    have = sum(size for rel, size in files.items()
               if os.path.exists(os.path.join(tmp, rel)) and os.path.getsize(os.path.join(tmp, rel)) == size)
    needed = manifest.get("bytes", 0) - have
    free = shutil.disk_usage(tmp).free
    if free - needed < DOWNLOAD_MARGIN:
        raise RuntimeError(f"needs {needed / 1e9:.1f} GB, {free / 1e9:.1f} GB free "
                           f"({DOWNLOAD_MARGIN / 1e9:.0f} GB must stay free)")

    job.update(0, status="running")
    total = max(1, manifest.get("bytes", 0))
    _rclone_copy(_remote(flat, kind), tmp, lambda done: job.update(0, progress=min(0.99, (have + done) / total)))
    wrong = sorted(rel for rel, size in files.items()
                   if not os.path.exists(os.path.join(tmp, rel)) or os.path.getsize(os.path.join(tmp, rel)) != size)
    if wrong:
        raise RuntimeError(f"{len(wrong)} file(s) missing or of another size, download again to resume: {', '.join(wrong[:5])}")
    job.update(0, status="done", progress=1.0)

    job.update(1, status="running")
    sidecar = boxes_mod.sidecar_path(tmp)
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
    _rclone("copyto", _remote(flat, "base", boxes_mod.DB_NAME), sidecar, timeout=1800)
    merge.reset_exchange(sidecar)
    base_rowid = merge.summary(sidecar)["last_rowid"]
    write_local(tmp, {"role": "annotator", "experiment": flat, "remote": read_config()["remote"], "kind": kind,
                      "downloaded_at": time.time(), "published_at": manifest.get("published_at"),
                      "publisher": manifest.get("publisher"),
                      "base_rowid": base_rowid, "pulled_rowid": manifest.get("snapshot_rowid", 0)})
    if os.path.lexists(final):
        raise RuntimeError(f"{final} appeared while downloading; the copy is in {tmp}")
    os.makedirs(os.path.dirname(final), exist_ok=True)
    os.rename(tmp, final)
    for empty in _parents(os.path.dirname(tmp), os.path.join(root, hostel)):
        try:
            os.rmdir(empty)
        except OSError:
            break
    index = os.path.join(root, "index.txt")
    if not os.path.exists(index):
        open(index, "a").close()
    added = _add_to_index(root, relpath, manifest.get("databases") or [])
    job.update(1, status="done", progress=1.0)
    job.result = {"experiment": flat, "kind": kind, "path": final, "bytes": manifest.get("bytes"),
                  "index_lines_added": added}


def _parents(path, stop):
    """path and its parents up to (not including) stop"""
    out = []
    while os.path.normpath(path) != os.path.normpath(stop) and len(path) > len(stop):
        out.append(path)
        path = os.path.dirname(path)
    return out


def upload(job, experiment, name, commit):
    flat, basedir = to_flat(experiment), basedir_of(experiment)
    state = read_local(basedir)
    if not state or state.get("role") != "annotator":
        raise RuntimeError("this copy was not downloaded from Drive: only annotators upload their annotations")
    job.update(0, status="running")
    workdir = tempfile.mkdtemp(prefix="exchange_", dir=_cache())
    try:
        copy = os.path.join(workdir, boxes_mod.DB_NAME)
        last = _snapshot_sidecar(basedir, copy)
        info = merge.summary(copy)
        _rclone("copyto", copy, _remote(flat, "annotations", name, boxes_mod.DB_NAME), timeout=1800)
        meta = {"annotator": name, "uploaded_at": time.time(), "host": socket.gethostname(), "app_commit": commit,
                "base_rowid": state.get("base_rowid", 0), "pulled_rowid": state.get("pulled_rowid", 0),
                "last_rowid": last, "frames": info["frames"],
                "own_frames": info["authors"].get(name, 0), "authors": info["authors"]}
        _put_json(_remote(flat, "annotations", name, "meta.json"), meta)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    state.update(uploaded_at=meta["uploaded_at"], annotator=name)
    write_local(basedir, state)
    job.update(0, status="done", progress=1.0)
    job.result = {"experiment": flat, "annotator": name, "frames": info["frames"]}


def _cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def _summarize(the_plan, limit=200):
    def brief(state):
        if state is None:
            return None
        return {"author": state["author"], "updated": state["updated"], "n_boxes": len(state["boxes"]),
                "labels": sorted(b["label"] for b in state["boxes"])}
    return {"source": the_plan["source"], "take": len(the_plan["take"]), "same": the_plan["same"],
            "n_conflicts": len(the_plan["conflicts"]),
            "take_frames": [e["frame_number"] for e in the_plan["take"]][:limit],
            "conflicts": [{"frame_number": e["frame_number"], "local": brief(e["local"]), "remote": brief(e["remote"]),
                           "default": merge._winner(e, "newest")} for e in the_plan["conflicts"][:limit]]}


def import_plan(job, experiment, name):
    """Publisher: plan of annotator `name`'s copy"""
    flat, basedir = to_flat(experiment), basedir_of(experiment)
    job.update(0, status="running")
    meta = _cat_json(_remote(flat, "annotations", name, "meta.json"))
    if meta is None:
        raise RuntimeError(f"{name} has not uploaded annotations of {flat}")
    copy = os.path.join(_cache(), f"{flat}__{name}.db")
    _rclone("copyto", _remote(flat, "annotations", name, boxes_mod.DB_NAME), copy, timeout=1800)
    sidecar = boxes_mod.sidecar_path(basedir)
    the_plan = merge.plan(sidecar, copy, source=name, me=PUBLISHER,
                          remote_since=meta.get("base_rowid", 0), local_since=meta.get("pulled_rowid", 0))
    _plans[job.id] = (flat, the_plan, "import", {"copy": copy, "meta": meta})
    job.update(0, status="done", progress=1.0)
    job.result = {"experiment": flat, "kind": "import", "annotator": name, "meta": meta, **_summarize(the_plan)}


def pull_plan(job, experiment):
    """Annotator: plan of the publisher's latest snapshot (scenes and problem frames are refreshed now)"""
    flat, basedir = to_flat(experiment), basedir_of(experiment)
    state = read_local(basedir)
    if not state or state.get("role") != "annotator":
        raise RuntimeError("this copy was not downloaded from Drive")
    job.update(0, status="running")
    manifest = _cat_json(_remote(flat, "manifest.json"))
    if manifest is None:
        raise RuntimeError(f"{flat} is no longer on Drive")
    copy = os.path.join(_cache(), f"{flat}__base.db")
    _rclone("copyto", _remote(flat, "base", boxes_mod.DB_NAME), copy, timeout=1800)
    refreshed, bundle_updated = [], False
    if state.get("kind") == "minimal":
        # the whole bundle (new scenes, their clips and rows); unchanged files are skipped by checksum
        before = (exchange_bundle.bundle_info(basedir) or {}).get("created_at")
        try:
            _rclone_copy(_remote(flat, "minimal"), basedir, lambda done: None, extra=["--checksum"])
            refreshed.append("bundle")
        except RuntimeError as error:
            logger.info("Bundle not refreshed: %s", error)
        bundle_updated = (exchange_bundle.bundle_info(basedir) or {}).get("created_at") != before
    else:
        for rel in PULL_FILES:
            try:
                _rclone("copyto", _remote(flat, "full", rel), os.path.join(basedir, rel), timeout=600)
                refreshed.append(rel)
            except RuntimeError as error:
                logger.info("Not refreshed %s: %s", rel, error)
    me = state.get("annotator") or annotator_name() or "annotator"
    sidecar = boxes_mod.sidecar_path(basedir)
    the_plan = merge.plan(sidecar, copy, source=PUBLISHER, me=me,
                          remote_since=state.get("pulled_rowid", 0), local_since=state.get("base_rowid", 0))
    _plans[job.id] = (flat, the_plan, "pull", {"copy": copy, "manifest": manifest})
    job.update(0, status="done", progress=1.0)
    job.result = {"experiment": flat, "kind": "pull", "refreshed": refreshed, "bundle_updated": bundle_updated,
                  "published_at": manifest.get("published_at"), **_summarize(the_plan)}


def apply_plan(job_id, prefer, overrides):
    if job_id not in _plans:
        raise KeyError(job_id)
    flat, the_plan, kind, extra = _plans.pop(job_id)
    basedir = basedir_of(flat)
    sidecar = boxes_mod.sidecar_path(basedir)
    out = merge.apply(sidecar, the_plan, prefer=prefer, overrides=overrides, lock=boxes_mod._lock)
    boxes_mod._chunk_labels.clear()
    state = read_local(basedir) or {}
    if kind == "pull":
        out["proposals_copied"] = merge.copy_proposals(sidecar, extra["copy"], lock=boxes_mod._lock)
        state["pulled_rowid"] = extra["manifest"].get("snapshot_rowid", state.get("pulled_rowid", 0))
        state["pulled_at"] = time.time()
        state["published_at"] = extra["manifest"].get("published_at")
    else:
        state.setdefault("imports", {})[the_plan["source"]] = {"at": time.time(), **out}
    write_local(basedir, state)
    try:
        os.remove(extra["copy"])
    except OSError:
        pass
    return {"experiment": flat, "kind": kind, **out}


# ---------------------------------------------------------------- routes

def register_gdrive_exchange(app, get_loaded=lambda: None, get_commit=lambda: None):
    """get_loaded() -> relpath of the loaded experiment (or None); get_commit() -> {"commit", ...}"""

    def run(flat, title, items, work):
        """Job running work(job) on one experiment, one transfer per experiment at a time (409)"""
        running = _claim(flat)
        if running:
            return jsonify({"error": f"{flat}: a transfer is running (job {running})", "job": running}), 409

        def wrapped(job):
            with _busy_lock:
                _busy[flat] = job.id
            try:
                work(job)
            finally:
                _release(flat, job.id)
        try:
            job = start_job(title, items, wrapped)
        except Exception:
            _release(flat, "starting")
            raise
        return jsonify({"job": job.to_dict()}), 202

    def body_experiment():
        body = request.get_json(silent=True) or {}
        experiment = body.get("experiment") or get_loaded()
        if not experiment:
            return body, None, (jsonify({"error": "no experiment given"}), 400)
        return body, to_flat(experiment), None

    def commit():
        return (get_commit() or {}).get("commit")

    @app.route("/api/exchange/status", methods=["GET"])
    def exchange_status():
        status = remote_status()
        status["loaded"] = to_flat(get_loaded()) if get_loaded() else None
        status["annotator_effective"] = annotator_name()
        return jsonify(status)

    @app.route("/api/exchange/config", methods=["POST"])
    def exchange_config():
        try:
            config = write_config(request.get_json(silent=True) or {})
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        return jsonify(config)

    @app.route("/api/exchange/experiments", methods=["GET"])
    def exchange_experiments():
        if request.args.get("refresh") or _listing["at"] is None:
            try:
                list_drive()
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                _listing.update(error=str(error))
                return jsonify({"error": str(error), "experiments": _listing["experiments"], "at": _listing["at"]}), 502
        # local state may have changed since the listing
        experiments = [dict(e, local={"exists": os.path.isdir(basedir_of(e["experiment"])),
                                      "exchange": read_local(basedir_of(e["experiment"]))})
                       for e in _listing["experiments"]]
        return jsonify({"experiments": experiments, "at": _listing["at"], "remote": read_config()["remote"]})

    @app.route("/api/exchange/local", methods=["GET"])
    def exchange_local():
        experiment = request.args.get("experiment") or get_loaded()
        if not experiment:
            return jsonify({"error": "no experiment given"}), 400
        info = local_info(experiment)
        if info["exists"] and request.args.get("size"):
            files = _folder_files(basedir_of(experiment))
            info["bytes"], info["n_files"] = sum(files.values()), len(files)
        return jsonify(info)

    @app.route("/api/exchange/publish", methods=["POST"])
    def exchange_publish():
        body, flat, error = body_experiment()
        if error:
            return error
        kind = body.get("kind", "full")
        if kind not in KINDS:
            return jsonify({"error": f"kind must be one of {KINDS}"}), 400
        basedir = basedir_of(flat)
        if not os.path.isdir(basedir):
            return jsonify({"error": f"{basedir} does not exist"}), 404
        state = read_local(basedir) or {}
        if state.get("role") == "annotator" and not body.get("force"):
            return jsonify({"error": "this copy was downloaded from Drive: annotators upload their annotations, "
                                     "they do not publish"}), 409
        publisher = annotator_name() or "publisher"
        annotations_only = bool(body.get("annotations_only"))
        title = f"Drive: publish {flat}" + (" (annotations)" if annotations_only else "")
        if exchange_bundle.bundle_info(basedir) is not None:
            return jsonify({"error": "this is a minimal bundle: only a whole experiment can be published"}), 409
        if kind == "minimal" and not annotations_only and not os.path.exists(generate_dbfile(flat)):
            return jsonify({"error": "a minimal bundle needs the experiment's tracking database"}), 400
        step = ("Scenes" if annotations_only else
                "Scene clips and database (built, then sent)" if kind == "minimal" else "Experiment folder")
        return run(flat, title, ["Snapshot of the annotations", step, "Annotations and manifest"],
                   lambda job: publish(job, flat, kind, annotations_only, publisher, commit()))

    @app.route("/api/exchange/download", methods=["POST"])
    def exchange_download():
        _, flat, error = body_experiment()
        if error:
            return error
        if os.path.lexists(basedir_of(flat)):
            return jsonify({"error": f"{flat} is here already: pull its updates instead"}), 409
        return run(flat, f"Drive: download {flat}", ["Experiment folder", "Annotations"], lambda job: download(job, flat))

    @app.route("/api/exchange/upload", methods=["POST"])
    def exchange_upload():
        _, flat, error = body_experiment()
        if error:
            return error
        name = annotator_name()
        if not name:
            return jsonify({"error": "set your annotator name first"}), 400
        state = read_local(basedir_of(flat))
        if not state or state.get("role") != "annotator":
            return jsonify({"error": "only a copy downloaded from Drive uploads its annotations"}), 409
        return run(flat, f"Drive: upload annotations of {flat}", ["Annotations"],
                   lambda job: upload(job, flat, name, commit()))

    @app.route("/api/exchange/import", methods=["POST"])
    def exchange_import():
        body, flat, error = body_experiment()
        if error:
            return error
        name = body.get("annotator") or ""
        if not NAME_RE.match(name):
            return jsonify({"error": "annotator name missing or invalid"}), 400
        state = read_local(basedir_of(flat))
        if state and state.get("role") == "annotator":
            return jsonify({"error": "this copy was downloaded from Drive: pull the publisher's updates instead"}), 409
        return run(flat, f"Drive: check {name}'s annotations of {flat}", ["Download and compare"],
                   lambda job: import_plan(job, flat, name))

    @app.route("/api/exchange/pull", methods=["POST"])
    def exchange_pull():
        _, flat, error = body_experiment()
        if error:
            return error
        state = read_local(basedir_of(flat))
        if not state or state.get("role") != "annotator":
            return jsonify({"error": "only a copy downloaded from Drive pulls updates"}), 409
        return run(flat, f"Drive: check updates of {flat}", ["Download and compare"], lambda job: pull_plan(job, flat))

    @app.route("/api/exchange/apply", methods=["POST"])
    def exchange_apply():
        body = request.get_json(silent=True) or {}
        try:
            out = apply_plan(str(body.get("job_id")), body.get("prefer", "newest"), body.get("overrides"))
        except KeyError:
            return jsonify({"error": "no plan for that job (applied already, or the server restarted): check again"}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        return jsonify(out)
