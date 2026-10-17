"""Links from the Group viewer to the CVAT project of the loaded experiment.

    GET /api/cvat[?refresh=1]   CVAT projects of the loaded experiment and the
                                frames selected for validation in their tasks

A project belongs to the experiment when its name is the experiment identifier
(`FlyHostel1_3X_2026-09-22_14-00-00`), optionally followed by a suffix
(`..._long`, `... - has a male`). Task images are named
`<experiment>_<chunk>_<frame_number0>_<block>.png` (one frame per image, so the
frame is frame_number0 + block), or `<frame_number>_....png` in the older format.

CVAT is reached with CVAT_HOST / CVAT_USERNAME / CVAT_PASSWORD (as in flyhostel).
Links handed to the browser use CVAT_PUBLIC_URL if set, else CVAT_HOST, or the
host name the browser used to reach this server when CVAT_HOST is local.
"""
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from urllib.parse import urlsplit

import requests
from flask import jsonify, request

from idtrackerai_validator_server.experiments import to_flat

logger = logging.getLogger(__name__)

CVAT_PORT = 8080
CACHE_SECONDS = 300
TIMEOUT = 15
LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}

_V1_NAME = re.compile(r"_(\d+)_(\d+)_(\d+)\.\w+$")
_V2_NAME = re.compile(r"^(\d+)")

_cache = {}     # experiment identifier -> (time, result)
_cache_lock = Lock()


def configured():
    return all(os.environ.get(v) for v in ("CVAT_HOST", "CVAT_USERNAME", "CVAT_PASSWORD"))


def _api_base():
    return f"http://{os.environ['CVAT_HOST']}:{CVAT_PORT}/api"


def public_base(browser_host=None):
    """CVAT URL for a browser: CVAT_PUBLIC_URL, else CVAT_HOST
    (replaced by browser_host, if given, when CVAT_HOST is local)."""
    if os.environ.get("CVAT_PUBLIC_URL"):
        return os.environ["CVAT_PUBLIC_URL"].rstrip("/")
    host = os.environ.get("CVAT_HOST", "localhost")
    if host in LOCAL_HOSTS and browser_host:
        host = browser_host
    return f"http://{host}:{CVAT_PORT}"


def _public_base():
    return public_base(urlsplit(request.host_url).hostname)


def frame_url(base_url, task, job, task_frame):
    """Link opening a frame of a task in the CVAT annotation view."""
    return f"{base_url}/tasks/{task}/jobs/{job}?frame={task_frame}"


def experiment_identifier(experiment):
    """FlyHostel1/3X/2026-09-22_14-00-00 -> FlyHostel1_3X_2026-09-22_14-00-00
    (FlyHostel1/4X/<date>/arenas/1 -> FlyHostel1_4X_<date>_arena1)"""
    return to_flat(experiment)


def frame_number_of(image_name):
    name = os.path.basename(image_name)
    match = _V1_NAME.search(name)
    if match:
        return int(match.group(2)) + int(match.group(3))
    match = _V2_NAME.match(name)
    return int(match.group(1)) if match else None


def _get_all(session, url, params):
    """Every result of a paginated CVAT listing."""
    results = []
    params = {**params, "page_size": 1000}
    while url:
        r = session.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        results.extend(data["results"])
        url, params = data.get("next"), None
    return results


def _matches(name, identifier):
    if not name.startswith(identifier):
        return False
    rest = name[len(identifier):]
    return rest == "" or not rest[0].isalnum()


def _task_frames(auth, task_id):
    """[(frame_number, job, task frame)] of one task."""
    base = _api_base()
    with requests.Session() as s:
        s.auth = auth
        jobs = [
            (j["start_frame"], j["stop_frame"], j["id"])
            for j in _get_all(s, f"{base}/jobs", {"task_id": task_id})
            if j.get("type", "annotation") == "annotation"
        ]
        r = s.get(f"{base}/tasks/{task_id}/data/meta", timeout=TIMEOUT)
        r.raise_for_status()
        meta = r.json()
    deleted = set(meta.get("deleted_frames") or [])
    start = meta.get("start_frame", 0)
    out = []
    for i, image in enumerate(meta.get("frames", [])):
        task_frame = start + i
        if task_frame in deleted:
            continue
        frame_number = frame_number_of(image["name"])
        job = next((jid for a, b, jid in jobs if a <= task_frame <= b), None)
        if frame_number is not None and job is not None:
            out.append((frame_number, job, task_frame))
    return out


def build_index(identifier):
    """Projects of the experiment and frame_number -> [task, job, task frame].

    A frame present in several tasks points to the task of the exact-name project
    first, then to the newest task."""
    auth = (os.environ["CVAT_USERNAME"], os.environ["CVAT_PASSWORD"])
    base = _api_base()
    with requests.Session() as s:
        s.auth = auth
        projects = [
            {"id": p["id"], "name": p["name"]}
            for p in _get_all(s, f"{base}/projects", {"search": identifier})
            if _matches(p["name"], identifier)
        ]
        # exact name first, then the others by name
        projects.sort(key=lambda p: (p["name"] != identifier, p["name"]))
        # least preferred first: later tasks overwrite earlier ones
        task_ids = []
        for project in reversed(projects):
            tasks = sorted(t["id"] for t in _get_all(s, f"{base}/tasks", {"project_id": project["id"]}))
            project["tasks"] = len(tasks)
            task_ids.extend(tasks)

    frames = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for task_id, rows in zip(task_ids, pool.map(lambda t: _task_frames(auth, t), task_ids)):
            for frame_number, job, task_frame in rows:
                frames[frame_number] = [task_id, job, task_frame]

    return {"projects": projects, "frames": frames, "n_tasks": len(task_ids)}


def register_cvat(app, get_selected_experiment):

    @app.route("/api/cvat", methods=["GET"])
    def cvat_index():
        experiment = get_selected_experiment()
        if not experiment:
            return jsonify({"error": "no experiment loaded"}), 503
        if not configured():
            return jsonify({"configured": False, "projects": [], "frames": {}})

        identifier = experiment_identifier(experiment)
        refresh = request.args.get("refresh") == "1"
        with _cache_lock:
            cached = _cache.get(identifier)
            if refresh or cached is None or time.time() - cached[0] > CACHE_SECONDS:
                try:
                    cached = (time.time(), build_index(identifier))
                except Exception as error:
                    logger.error("CVAT lookup of %s failed: %s", identifier, error)
                    return jsonify({"error": f"CVAT lookup failed: {error}"}), 502
                _cache[identifier] = cached
        result = cached[1]
        return jsonify({
            "configured": True,
            "experiment": identifier,
            "base_url": _public_base(),
            **result,
        })
