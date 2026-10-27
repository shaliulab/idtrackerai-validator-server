"""Data collector tab: fetch new recordings from the recording computer (cv2), propose and
adjust their idtrackerai settings, and check them with preprocess-sample.

Fetch: cv2 keeps its recordings in $COLLECTOR_REMOTE_ROOT (also served read-only over http
at $COLLECTOR_HTTP). Listing and copying go through ssh / rsync ($COLLECTOR_REMOTE_HOST, a
key without passphrase): the http listing has no sizes or times. Only what tracking needs is
copied: the files of the experiment folder (not its subfolders) and the chunks starting
before ZT `zt_limit` (30: the first 24 h and a buffer), skipping files modified in the last
`RECENT_S` (still being written). ZT0 is the experiment date at its `reference_hour` in the
metadata sheets (DEFAULT_REFERENCE_HOUR when the sheets have no row yet); a chunk starts at
created_utc + chunk * chunksize / framerate (metadata.yaml). One fetch at a time.

Settings: `roi-proposer propose` writes <date>.toml + landmarks.toml (arenas.toml for arena
experiments) marked with AI_MARK; the editor (a web version of idtrackerai v6's segmentation
app) rewrites them, the old file kept in toml_backups/, and converts <date>.toml to the
<date>.conf (idtrackerai v4) preprocess-sample and the tracking read.

preprocess-sample (idtrackeraiv4) samples frames of the chunks and writes the frames where
the segmentation does not find number_of_animals blobs to preprocess_sample.csv. It takes
hours, so it runs in its own session (it survives server restarts) and is found again by its
command line; the chunk it reads (its open .mp4) gives its progress.
"""
import datetime
import glob
import io
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import psutil
import yaml
from flask import jsonify, request, send_file

try:
    import tomllib
except ImportError:  # python < 3.11
    import tomli as tomllib

from idtrackerai_validator_server import rois
from idtrackerai_validator_server.jobs import start_job
from idtrackerai_validator_server.overview import sheet_files

logger = logging.getLogger(__name__)

REMOTE_HOST = os.environ.get("COLLECTOR_REMOTE_HOST", "cv2")
REMOTE_ROOT = os.environ.get("COLLECTOR_REMOTE_ROOT", "/home/vibflysleep/flyhostel_data/videos")
HTTP_URL = os.environ.get("COLLECTOR_HTTP", "http://10.43.205.5:8000")
ENVS = os.path.expanduser("~/mambaforge/envs")
ROI_PROPOSER = os.environ.get("ROI_PROPOSER", os.path.join(ENVS, "roi-proposer", "bin", "roi-proposer"))
IDTRACKERAI_V4_BIN = os.environ.get("IDTRACKERAI_V4_BIN", os.path.join(ENVS, "idtrackeraiv4", "bin"))
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]

ZT_LIMIT = 30
DEFAULT_REFERENCE_HOUR = 12
RECENT_S = 600                   # files modified more recently may still be written
FETCH_MARGIN = 50e9              # bytes a drive keeps free after a fetch
PREPROCESS_MAX_CHUNK = 300       # as roi-proposer
PREPROCESS_SAMPLES_PER_CHUNK = 20
AI_MARK = "# Proposed by roi-proposer (AI)"
TOML_FILES = ("{date_time}.toml", "landmarks.toml", "arenas.toml")
# key order of the tomls roi-proposer and idtrackerai's GUI write
TOML_ORDER = ["name", "video_paths", "intensity_ths", "area_ths", "tracking_intervals", "number_of_animals",
              "use_bkg", "check_segmentation", "resolution_reduction", "track_wo_identities", "roi_list",
              "exclusive_rois"]

EXPERIMENT_RE = re.compile(r"^(FlyHostel\d+)/(\d+)X/((\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2}))$")
# local folders the editor works on: experiments and the arenas cropped out of them
LOCAL_RE = re.compile(r"^(FlyHostel\d+)/(\d+)X/(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(/arenas/(\d+))?$")
CHUNK_RE = re.compile(r"^(\d{6})\.[A-Za-z0-9]+$")

_fetch_lock = threading.Lock()
_fetching = set()
_propose_lock = threading.Lock()


def _check_experiment(experiment, arenas=False):
    experiment = (experiment or "").strip().strip("/")
    if not (LOCAL_RE if arenas else EXPERIMENT_RE).match(experiment):
        raise ValueError(f"Not an experiment folder (FlyHostelN/NX/YYYY-MM-DD_HH-MM-SS"
                         f"{'[/arenas/K]' if arenas else ''}): {experiment!r}")
    return experiment


def date_time_of(experiment):
    """DATE_TIME of an experiment or of one of its arenas: the name of its tomls."""
    return LOCAL_RE.match(experiment).group(3)


def _ssh(script, timeout=120):
    result = subprocess.run(SSH + [REMOTE_HOST, "bash -s"], input=script, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"ssh {REMOTE_HOST} failed ({result.returncode}): {result.stderr.strip()[-500:]}")
    return result.stdout


# ---- reference hour (ZT0) from the metadata sheets ----

_sheet_cache = {"key": None, "hours": {}}


def reference_hours():
    """{(FlyHostelN, date_time): reference_hour} of the metadata sheet rows."""
    files, _ = sheet_files()
    key = tuple(sorted((path, os.path.getmtime(path)) for path in files.values() if os.path.exists(path)))
    if _sheet_cache["key"] == key:
        return _sheet_cache["hours"]
    hours = {}
    for path in files.values():
        try:
            df = pd.read_csv(path, dtype=str, usecols=lambda c: c in ("basedir", "reference_hour"))
        except Exception as error:
            logger.warning("Cannot read %s: %s", path, error)
            continue
        if "basedir" not in df.columns or "reference_hour" not in df.columns:
            continue
        for basedir, hour in zip(df["basedir"].fillna(""), df["reference_hour"]):
            match = re.search(r"(FlyHostel\d+)/\d+X/(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", basedir)
            try:
                hour = float(hour)
            except (TypeError, ValueError):
                continue
            if match and math.isfinite(hour):
                hours.setdefault(match.groups(), hour)
    _sheet_cache.update(key=key, hours=hours)
    return hours


# ---- what cv2 has ----

def remote_experiments():
    """Experiment folders of cv2 (slash form)."""
    out = _ssh(f"cd {REMOTE_ROOT} || exit 1\nfor d in FlyHostel*/*X/*/; do echo \"${{d%/}}\"; done\n")
    return sorted(line for line in out.splitlines() if EXPERIMENT_RE.match(line))


def remote_files(date):
    """{experiment: {"metadata": {...}, "files": {name: (size, mtime)}}} of the experiments of
    `date` (YYYY-MM-DD) on cv2, and cv2's clock."""
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date or ""):
        raise ValueError(f"Not a date: {date!r}")
    script = f"""cd {REMOTE_ROOT} || exit 1
echo "@@NOW $(date +%s)"
for d in FlyHostel*/*X/{date}_*/; do
  [ -d "$d" ] || continue
  d="${{d%/}}"
  echo "@@EXP $d"
  grep -E '^\\s*(chunksize|framerate|created_utc|imgshape):' "$d/metadata.yaml" 2>/dev/null | sed 's/^/@@MD /'
  find "$d" -maxdepth 1 -type f -printf '%f\\t%s\\t%T@\\n'
done
"""
    out = _ssh(script)
    now, experiments, current = time.time(), {}, None
    for line in out.splitlines():
        if line.startswith("@@NOW "):
            now = float(line.split()[1])
        elif line.startswith("@@EXP "):
            current = line[6:].strip()
            experiments[current] = {"metadata": {}, "files": {}}
        elif line.startswith("@@MD ") and current:
            field, _, value = line[5:].strip().partition(":")
            experiments[current]["metadata"][field.strip()] = value.strip().strip("'\"")
        elif current and "\t" in line:
            name, size, mtime = line.split("\t")
            experiments[current]["files"][name] = (int(size), float(mtime))
    return experiments, now


def plan(experiment, remote, now, zt_limit, root):
    """What to copy of one experiment: chunks before ZT zt_limit and the other top-level files."""
    hostel, _, date_time, date, hour, minute, second = EXPERIMENT_RE.match(experiment).groups()
    md = remote["metadata"]
    hours = reference_hours()
    reference_hour = hours.get((hostel, date_time))
    try:
        chunk_s = float(md["chunksize"]) / float(md["framerate"])
        created = datetime.datetime.fromisoformat(md["created_utc"])
    except (KeyError, ValueError, ZeroDivisionError):
        chunk_s = created = None
    zt0 = datetime.datetime.fromisoformat(date).replace(
        hour=0) + datetime.timedelta(hours=reference_hour if reference_hour is not None else DEFAULT_REFERENCE_HOUR)
    zt_start = (created - zt0).total_seconds() / 3600 if created else None

    folder = os.path.join(root, experiment)
    wanted, chunks, recent = [], set(), 0
    for name, (size, mtime) in sorted(remote["files"].items()):
        match = CHUNK_RE.match(name)
        chunk = int(match.group(1)) if match else None
        if chunk is not None:
            chunks.add(chunk)
            if zt_start is None or zt_start + chunk * chunk_s / 3600 >= zt_limit:
                continue
        if now - mtime < RECENT_S:
            recent += 1
            continue
        try:
            local_size = os.path.getsize(os.path.join(folder, name))
        except OSError:
            local_size = None
        wanted.append({"name": name, "size": size, "chunk": chunk, "here": local_size == size})

    wanted_chunks = sorted({f["chunk"] for f in wanted if f["chunk"] is not None})
    missing = [f for f in wanted if not f["here"]]
    last_chunk = max(chunks) if chunks else None
    recording = bool(remote["files"]) and now - max(m for _, m in remote["files"].values()) < RECENT_S
    return {
        "experiment": experiment, "flyhostel": hostel, "group_size": int(experiment.split("/")[1][:-1]),
        "date_time": date_time, "created_utc": md.get("created_utc"),
        "framerate": float(md["framerate"]) if md.get("framerate") else None,
        "chunk_minutes": round(chunk_s / 60, 2) if chunk_s else None,
        "reference_hour": reference_hour if reference_hour is not None else DEFAULT_REFERENCE_HOUR,
        "reference_hour_source": "sheet" if reference_hour is not None else "default",
        "zt_start": round(zt_start, 2) if zt_start is not None else None,
        "remote_chunks": len(chunks), "last_chunk": last_chunk,
        "remote_hours": round(len(chunks) * chunk_s / 3600, 1) if chunk_s else None,
        "recording": recording, "recent_files": recent,
        "wanted_chunks": len(wanted_chunks),
        "wanted_last_chunk": wanted_chunks[-1] if wanted_chunks else None,
        "wanted_bytes": sum(f["size"] for f in wanted),
        "here_files": sum(f["here"] for f in wanted), "wanted_files": len(wanted),
        "missing_bytes": sum(f["size"] for f in missing),
        "missing": [f["name"] for f in missing],
        "local": os.path.isdir(folder),
        "arenas": sorted(int(a) for a in os.listdir(os.path.join(folder, "arenas")) if a.isdigit())
        if os.path.isdir(os.path.join(folder, "arenas")) else [],
        "fetching": experiment in _fetching,
    }


def _drive_free(root, experiment):
    path = os.path.join(root, experiment.split("/")[0])
    return shutil.disk_usage(path).free if os.path.isdir(path) else None


def fetch(job, index, experiment, files, root):
    """rsync `files` of `experiment` from cv2 into root; progress from rsync's progress2."""
    folder = os.path.join(root, experiment)
    os.makedirs(folder, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(f"{experiment}/{name}" for name in files) + "\n")
        files_from = handle.name
    cmd = [
        "rsync", "-a", "--partial", "--keep-dirlinks", "--no-implied-dirs", "--info=progress2", "--no-inc-recursive",
        "-e", " ".join(SSH), f"--files-from={files_from}", f"{REMOTE_HOST}:{REMOTE_ROOT}/", f"{root}/",
    ]
    try:
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
            raise RuntimeError(f"rsync exited with {process.returncode}: {stderr.strip()[-800:]}")
    finally:
        os.unlink(files_from)


# ---- settings (tomls) ----

def experiment_folder(root, experiment):
    folder = os.path.join(root, _check_experiment(experiment, arenas=True))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"{experiment} is not here yet: fetch it first")
    return folder


def store_info(folder):
    with open(os.path.join(folder, "metadata.yaml")) as handle:
        store = (yaml.safe_load(handle) or {}).get("__store", {})
    chunks = sorted(int(os.path.basename(p)[:6]) for p in glob.glob(os.path.join(folder, "[0-9]" * 6 + ".mp4")))
    shape = store.get("imgshape") or [None, None]
    return {
        "chunksize": int(float(store["chunksize"])), "framerate": float(store["framerate"]),
        "height": shape[0], "width": shape[1], "chunks": chunks, "created_utc": store.get("created_utc"),
        "camera": store.get("camera-model"),
    }


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def toml_files(folder, date_time):
    out = []
    for pattern in TOML_FILES:
        name = pattern.format(date_time=date_time)
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            continue
        entry = {"name": name, "mtime": os.path.getmtime(path)}
        try:
            with open(path) as handle:
                text = handle.read()
            entry["text"] = text
            entry["ai_proposed"] = text.startswith(AI_MARK)
            entry["settings"] = _jsonable(tomllib.loads(text))
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        out.append(entry)
    return out


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value) if "'" in value or "\n" in value else f"'{value}'"
    if isinstance(value, list):
        if value and all(isinstance(v, str) for v in value):
            return "[\n" + "".join(f"    {_toml_value(v)},\n" for v in value) + "]"
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ValueError(f"Cannot write {value!r} to a toml")


def _number(value, name):
    if value in ("inf", None):
        return math.inf
    value = float(value)
    if math.isnan(value):
        raise ValueError(f"{name} is not a number")
    return int(value) if value.is_integer() else value


def clean_settings(settings):
    """Validated settings of the editor, in TOML_ORDER (unknown keys kept at the end)."""
    settings = dict(settings)
    lo, hi = settings["intensity_ths"]
    settings["intensity_ths"] = [_number(lo, "intensity_ths"), _number(hi, "intensity_ths")]
    if not 0 <= settings["intensity_ths"][0] <= settings["intensity_ths"][1] <= 255:
        raise ValueError(f"intensity_ths must be 0 ≤ low ≤ high ≤ 255: {settings['intensity_ths']}")
    lo, hi = settings["area_ths"]
    settings["area_ths"] = [_number(lo, "area_ths"), _number(hi, "area_ths")]
    if not 0 <= settings["area_ths"][0] <= settings["area_ths"][1]:
        raise ValueError(f"area_ths must be 0 ≤ low ≤ high: {settings['area_ths']}")
    settings["number_of_animals"] = int(settings["number_of_animals"])
    if settings["number_of_animals"] < 0:
        raise ValueError("number_of_animals must be ≥ 0")
    roi_list = settings.get("roi_list")
    if roi_list is not None:
        for label in roi_list:
            rois.roi_vertices(label)       # raises on a malformed ROI
        if not roi_list:
            raise ValueError("Add a region of interest or untick Regions of interest")
    else:
        settings.pop("roi_list", None)
        settings.pop("exclusive_rois", None)
    ordered = {k: settings[k] for k in TOML_ORDER if k in settings and settings[k] is not None}
    ordered.update({k: v for k, v in settings.items() if k not in ordered and v is not None})
    return ordered


def write_toml(folder, name, settings, was_ai):
    path = os.path.join(folder, name)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"# Edited in the idtrackerai validator on {stamp}"
    if was_ai:
        header += " (reviewed proposal of roi-proposer)"
    text = header + "\n" + "".join(f"{k} = {_toml_value(v)}\n" for k, v in settings.items())
    tomllib.loads(text)                     # what we write must read back
    if os.path.exists(path):
        backups = os.path.join(folder, "toml_backups")
        os.makedirs(backups, exist_ok=True)
        shutil.copy2(path, os.path.join(backups, f"{name}.{datetime.datetime.now():%Y%m%d-%H%M%S}"))
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(text)
    os.replace(tmp, path)
    return path


# arenas.toml (recordings split into N arenas, cropped by imgstore-crop-arenas): settings of
# the hand-made ones; one ellipse per arena, found by roi-proposer's detector when it finds
# exactly N, else laid out on a grid
ARENAS_DEFAULTS = {"intensity_ths": [0, 130], "area_ths": [50, "inf"], "tracking_intervals": "",
                   "number_of_animals": 0, "use_bkg": False, "check_segmentation": False,
                   "track_wo_identities": True, "exclusive_rois": False}
MAX_ARENAS = 16


def grid_arenas(n, width, height):
    """n circles on a rows x cols grid (rows ≤ cols), top-left to bottom-right."""
    rows = max(1, int(math.floor(math.sqrt(n))))
    cols = int(math.ceil(n / rows))
    cw, ch = width / cols, height / rows
    r = int(round(0.45 * min(cw, ch)))
    return [f"+ Ellipse {{'center': [{int(round((i % cols + 0.5) * cw))}, {int(round((i // cols + 0.5) * ch))}], "
            f"'axes': [{r}, {r}], 'angle': 0}}" for i in range(n)]


# run in the roi-proposer env (its detector needs cv2 ≥ 4.5's HOUGH_GRADIENT_ALT)
DETECT_ARENAS = """
import json, sys, cv2
from flyhostel.roi_proposer.rois import ARENAS_TOML_SCALE, detect_arenas, fit_ellipse
img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
print(json.dumps([fit_ellipse(a, scale=ARENAS_TOML_SCALE) for a in detect_arenas(img)]))
"""


def propose_arenas(frame, n):
    """(roi_list, source, detected) of n arenas in the frame."""
    detected = 0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "frame.png")
            cv2.imwrite(path, frame)
            result = subprocess.run([os.path.join(os.path.dirname(ROI_PROPOSER), "python"), "-c", DETECT_ARENAS, path],
                                    capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "failed")
        found = json.loads(result.stdout.strip().splitlines()[-1])
        detected = len(found)
        if detected == n:
            return found, "detected", detected
    except Exception as error:   # roi-proposer missing or detection failed: grid
        logger.warning("Arena detection failed: %s", error)
    height, width = frame.shape[:2]
    return grid_arenas(n, width, height), "grid", detected


def conf_convert(toml_path):
    conf_path = os.path.splitext(toml_path)[0] + ".conf"
    result = subprocess.run([os.path.join(IDTRACKERAI_V4_BIN, "conf-convert"), "-i", toml_path, "-o", conf_path],
                            capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"conf-convert failed: {(result.stderr or result.stdout).strip()[-500:]}")
    return conf_path


# ---- frames ----

class Frames:
    """Frames of the chunks of experiment folders (cv2 per chunk: the folder may not be an
    analyzed experiment), the last few decoded frames cached."""

    def __init__(self, n_captures=4, n_frames=16):
        self._caps = OrderedDict()   # (folder, chunk) -> [capture, next index]
        self._frames = OrderedDict()
        self._lock = threading.Lock()
        self._n_caps, self._n_frames = n_captures, n_frames

    def get(self, folder, frame_number, chunksize):
        key = (folder, frame_number)
        with self._lock:
            if key in self._frames:
                self._frames.move_to_end(key)
                return self._frames[key]
            chunk, index = divmod(frame_number, chunksize)
            entry = self._caps.get((folder, chunk))
            if entry is None:
                path = os.path.join(folder, f"{chunk:06d}.mp4")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Chunk {chunk} is not here")
                entry = [cv2.VideoCapture(path), 0]
                self._caps[(folder, chunk)] = entry
                while len(self._caps) > self._n_caps:
                    self._caps.popitem(last=False)[1][0].release()
            self._caps.move_to_end((folder, chunk))
            cap = entry[0]
            if entry[1] != index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            entry[1] = index + 1 if ok else -1
            if not ok:
                raise ValueError(f"Frame {frame_number} cannot be read")
            frame = frame[:, :, 0] if frame.ndim == 3 else frame
            self._frames[key] = frame
            while len(self._frames) > self._n_frames:
                self._frames.popitem(last=False)
            return frame


FRAMES = Frames()


# ---- preprocess-sample ----

def _npz_readable(path):
    try:
        with np.load(path) as npz:
            for key in npz.files:
                npz[key]
        return True
    except Exception:
        return False


def preprocess_chunks(folder):
    """First chunk, and the last one before PREPROCESS_MAX_CHUNK with a readable .npz (as roi-proposer)."""
    chunks = sorted(int(os.path.basename(f)[:6]) for f in glob.glob(os.path.join(folder, "[0-9]" * 6 + ".mp4"))
                    if os.path.exists(f[:-4] + ".npz"))
    for last in reversed([c for c in chunks if c < PREPROCESS_MAX_CHUNK]):
        if _npz_readable(os.path.join(folder, f"{last:06d}.npz")):
            return chunks[0], last
    raise FileNotFoundError(f"No chunk with a readable .npz before chunk {PREPROCESS_MAX_CHUNK}")


def preprocess_process(folder):
    """The preprocess-sample process checking this folder, if any (started here or by roi-proposer)."""
    store = os.path.join(os.path.realpath(folder), "metadata.yaml")
    stores = {store, os.path.join(folder, "metadata.yaml")}
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        cmdline = process.info["cmdline"] or []
        if any("preprocess-sample" in part for part in cmdline[:2]) and any(part in stores for part in cmdline):
            return process
    return None


def _arg(cmdline, name):
    return int(cmdline[cmdline.index(name) + 1]) if name in cmdline else None


def preprocess_status(folder):
    csv_path = os.path.join(folder, "preprocess_sample.csv")
    log_path = os.path.join(folder, "preprocess_sample.log")
    status = {"running": False, "csv": None, "log": None}
    process = preprocess_process(folder)
    if process is not None:
        try:
            cmdline = process.info["cmdline"]
            current = None
            for f in process.open_files():
                match = re.match(r"^(\d{6})\.mp4$", os.path.basename(f.path))
                if match:
                    current = int(match.group(1))
            status.update(running=True, pid=process.pid, started=process.info["create_time"],
                          start_chunk=_arg(cmdline, "--start-chunk"), end_chunk=_arg(cmdline, "--end-chunk"),
                          samples_per_chunk=_arg(cmdline, "--samples-per-chunk"), current_chunk=current)
        except psutil.Error:
            pass
    if os.path.exists(log_path):
        with open(log_path, errors="replace") as handle:
            lines = handle.read().splitlines()
        status["log"] = {"mtime": os.path.getmtime(log_path), "tail": lines[-40:]}
        total = next((line for line in reversed(lines) if line.startswith("Total problematic frames")), None)
        status["finished"] = total is not None and not status["running"]
        status["total_line"] = total
        error = next((line for line in reversed(lines) if re.match(r"^\w*(Error|Exception)\b", line)), None)
        status["error"] = error if not status["running"] and total is None else None
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["frame_number", "chunk", "video", "frame_idx", "n", "x", "y"])
        frames = []
        for frame_number, rows in df.groupby("frame_number", sort=True):
            first = rows.iloc[0]
            frames.append({
                "frame_number": int(frame_number), "chunk": int(first["chunk"]), "video": first["video"],
                "frame_idx": int(first["frame_idx"]), "n": int(first["n"]),
                "points": [[float(x), float(y)] for x, y in zip(rows["x"], rows["y"]) if pd.notna(x) and pd.notna(y)],
            })
        status["csv"] = {"mtime": os.path.getmtime(csv_path), "rows": len(df), "frames": frames}
    conf = os.path.join(folder, os.path.basename(folder) + ".conf")
    status["has_conf"] = os.path.exists(conf)
    if status["has_conf"]:
        try:
            with open(conf) as handle:
                status["number_of_animals"] = int(float(json.load(handle)["_number_of_animals"]["value"]))
        except Exception:
            pass
    return status


def start_preprocess(folder, start_chunk, end_chunk, samples_per_chunk):
    """end_chunk inclusive (preprocess-sample's --end-chunk is exclusive)."""
    cmd = [
        os.path.join(IDTRACKERAI_V4_BIN, "preprocess-sample"), "--store-path", os.path.join(folder, "metadata.yaml"),
        "--start-chunk", str(start_chunk), "--end-chunk", str(end_chunk + 1),
        "--samples-per-chunk", str(samples_per_chunk), "--output", os.path.join(folder, "preprocess_sample.csv"),
    ]
    with open(os.path.join(folder, "preprocess_sample.log"), "w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        # own session: keeps running when the server restarts
        process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    return process.pid


# ---- routes ----

def register_data_collector(app):
    root = os.environ["FLYHOSTEL_VIDEOS"]

    def bad_request(error):
        return jsonify({"error": str(error)}), 400

    @app.route("/api/collector/dates", methods=["GET"])
    def collector_dates():
        """Experiments of cv2 grouped by date, newest first."""
        try:
            experiments = remote_experiments()
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            return jsonify({"error": str(error)}), 502
        dates = {}
        for experiment in experiments:
            date = EXPERIMENT_RE.match(experiment).group(4)
            dates.setdefault(date, []).append({"experiment": experiment,
                                               "local": os.path.isdir(os.path.join(root, experiment))})
        return jsonify({
            "host": REMOTE_HOST, "remote_root": REMOTE_ROOT, "http": HTTP_URL,
            "dates": [{"date": d, "experiments": dates[d]} for d in sorted(dates, reverse=True)],
        })

    @app.route("/api/collector/date/<date>", methods=["GET"])
    def collector_date(date):
        """The experiments of `date` on cv2 with what a fetch would copy (?zt_limit=30)."""
        try:
            zt_limit = float(request.args.get("zt_limit", ZT_LIMIT))
            experiments, now = remote_files(date)
        except ValueError as error:
            return bad_request(error)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            return jsonify({"error": str(error)}), 502
        records = [plan(e, experiments[e], now, zt_limit, root) for e in sorted(experiments)]
        drives = {}
        for record in records:
            hostel = record["flyhostel"]
            if hostel not in drives:
                drives[hostel] = _drive_free(root, record["experiment"])
            record.pop("missing")
        return jsonify({"date": date, "zt_limit": zt_limit, "experiments": records, "free": drives,
                        "margin": FETCH_MARGIN, "fetching": sorted(_fetching)})

    @app.route("/api/collector/fetch", methods=["POST"])
    def collector_fetch():
        """{"experiments": [...], "zt_limit"}: job copying them from cv2, one item per experiment."""
        body = request.get_json(silent=True) or {}
        try:
            experiments = [_check_experiment(e) for e in body.get("experiments") or []]
            zt_limit = float(body.get("zt_limit", ZT_LIMIT))
        except ValueError as error:
            return bad_request(error)
        if not experiments:
            return bad_request("No experiment selected")
        if not _fetch_lock.acquire(blocking=False):
            return jsonify({"error": "A fetch is already running"}), 409
        try:
            dates = sorted({EXPERIMENT_RE.match(e).group(4) for e in experiments})
            listed, now = {}, time.time()
            for date in dates:
                found, now = remote_files(date)
                listed.update(found)
            plans = []
            for experiment in experiments:
                if experiment not in listed:
                    raise ValueError(f"{experiment} is not on {REMOTE_HOST}")
                plans.append(plan(experiment, listed[experiment], now, zt_limit, root))
            needed = {}
            for p in plans:
                needed[p["flyhostel"]] = needed.get(p["flyhostel"], 0) + p["missing_bytes"]
            for hostel, nbytes in needed.items():
                free = _drive_free(root, f"{hostel}/")
                if free is None:
                    raise ValueError(f"{os.path.join(root, hostel)} does not exist")
                if free - nbytes < FETCH_MARGIN:
                    raise ValueError(f"{hostel}'s drive has {free / 1e9:.0f} GB free, {nbytes / 1e9:.0f} GB needed "
                                     f"(at least {FETCH_MARGIN / 1e9:.0f} GB must stay free)")
        except Exception as error:
            _fetch_lock.release()
            status = 502 if isinstance(error, (RuntimeError, subprocess.TimeoutExpired)) else 400
            return jsonify({"error": str(error)}), status
        _fetching.update(experiments)

        def work(job):
            try:
                failed = []
                for index, p in enumerate(plans):
                    if not p["missing"]:
                        job.update(index, status="skipped", progress=1.0, error="already here")
                        continue
                    job.update(index, status="running")
                    try:
                        fetch(job, index, p["experiment"], p["missing"], root)
                        job.update(index, status="done", progress=1.0)
                    except Exception as error:
                        logger.exception("Fetching %s failed", p["experiment"])
                        job.update(index, status="failed", error=str(error))
                        failed.append(p["experiment"])
                    finally:
                        _fetching.discard(p["experiment"])
                job.result = {"experiments": experiments, "failed": failed}
                if failed:
                    raise RuntimeError(f"{len(failed)} experiment(s) failed: {', '.join(failed)}")
            finally:
                _fetching.difference_update(experiments)
                _fetch_lock.release()

        job = start_job(f"Fetch {len(plans)} experiment(s) from {REMOTE_HOST} (up to ZT{zt_limit:g})",
                        [p["experiment"] for p in plans], work)
        return jsonify(job.to_dict()), 202

    @app.route("/api/collector/settings", methods=["GET"])
    def collector_settings():
        """?experiment=: its tomls (text, settings, ai_proposed), the store and the preview."""
        try:
            folder = experiment_folder(root, request.args.get("experiment"))
            store = store_info(folder)
        except (ValueError, FileNotFoundError) as error:
            return bad_request(error)
        experiment = request.args.get("experiment").strip().strip("/")
        date_time = date_time_of(experiment)
        arena = LOCAL_RE.match(experiment).group(5)
        preview = os.path.join(folder, "rois_proposal.jpg")
        return jsonify({
            "experiment": experiment, "store": store, "files": toml_files(folder, date_time),
            "date_time": date_time, "arena": arena,
            "preview": os.path.getmtime(preview) if os.path.exists(preview) else None,
            "conf": os.path.exists(os.path.join(folder, date_time + ".conf")),
            # the NX of an arena folder is the whole recording's: 1 animal per arena by default (as roi-proposer)
            "group_size": None if arena else int(experiment.split("/")[1][:-1]),
        })

    @app.route("/api/collector/settings", methods=["POST"])
    def collector_save():
        """{"experiment", "file", "settings"}: writes the toml (old one in toml_backups/);
        <date>.toml is also converted to <date>.conf."""
        body = request.get_json(silent=True) or {}
        try:
            folder = experiment_folder(root, body.get("experiment"))
            experiment = body["experiment"].strip().strip("/")
            name = body.get("file")
            allowed = [p.format(date_time=date_time_of(experiment)) for p in TOML_FILES]
            if name not in allowed:
                raise ValueError(f"file must be one of {allowed}")
            settings = clean_settings(body.get("settings") or {})
        except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
            return bad_request(f"missing {error}" if isinstance(error, KeyError) else error)
        path = os.path.join(folder, name)
        was_ai = os.path.exists(path) and open(path).read().startswith(AI_MARK)
        path = write_toml(folder, name, settings, was_ai)
        response = {"written": path, "conf": None}
        if name == f"{date_time_of(experiment)}.toml":
            try:
                response["conf"] = conf_convert(path)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                response["conf_error"] = str(error)
        response["files"] = toml_files(folder, date_time_of(experiment))
        return jsonify(response)

    @app.route("/api/collector/propose", methods=["POST"])
    def collector_propose():
        """{"experiment", "force", "chunk"}: job running roi-proposer propose (no preprocess-sample)."""
        body = request.get_json(silent=True) or {}
        try:
            folder = experiment_folder(root, body.get("experiment"))
        except (ValueError, FileNotFoundError) as error:
            return bad_request(error)
        if not _propose_lock.acquire(blocking=False):
            return jsonify({"error": "roi-proposer is already running"}), 409
        cmd = [ROI_PROPOSER, "propose", folder, "--no-preprocess-sample"]
        if body.get("force"):
            cmd.append("--force")
        if body.get("chunk") not in (None, ""):
            cmd += ["--chunk", str(int(body["chunk"]))]

        def work(job):
            try:
                job.update(0, status="running", progress=0.1)
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                output = (result.stdout + result.stderr).strip()
                job.result = {"output": output, "returncode": result.returncode}
                if result.returncode != 0:
                    job.update(0, status="failed", error=output.splitlines()[-1] if output else "failed")
                    raise RuntimeError(output[-800:])
                job.update(0, status="done", progress=1.0)
            finally:
                _propose_lock.release()

        job = start_job(f"roi-proposer propose {body.get('experiment')}", [body.get("experiment")], work)
        return jsonify(job.to_dict()), 202

    @app.route("/api/collector/preview", methods=["GET"])
    def collector_preview():
        try:
            folder = experiment_folder(root, request.args.get("experiment"))
        except (ValueError, FileNotFoundError) as error:
            return bad_request(error)
        path = os.path.join(folder, "rois_proposal.jpg")
        if not os.path.exists(path):
            return jsonify({"error": "No rois_proposal.jpg"}), 404
        return send_file(path, mimetype="image/jpeg", max_age=0)

    @app.route("/api/collector/frame", methods=["GET"])
    def collector_frame():
        """?experiment=&frame_number=: JPEG of that frame of the folder's chunks."""
        try:
            folder = experiment_folder(root, request.args.get("experiment"))
            frame = FRAMES.get(folder, int(request.args["frame_number"]), store_info(folder)["chunksize"])
        except (ValueError, KeyError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return send_file(io.BytesIO(jpeg.tobytes()), mimetype="image/jpeg", max_age=0)

    @app.route("/api/collector/segment", methods=["POST"])
    def collector_segment():
        """{"experiment", "frame_number", "intensity_ths", "area_ths", "roi_list" (null = whole frame)}:
        contours and areas of the blobs idtrackerai would find."""
        body = request.get_json(silent=True) or {}
        try:
            folder = experiment_folder(root, body.get("experiment"))
            frame = FRAMES.get(folder, int(body["frame_number"]), store_info(folder)["chunksize"])
            intensity = [_number(v, "intensity_ths") for v in body["intensity_ths"]]
            area = [_number(v, "area_ths") for v in body["area_ths"]]
            source = {"rois": [], "_segmentation": {"min_threshold": intensity[0], "max_threshold": intensity[1],
                                                     "min_area": area[0], "max_area": area[1]}}
            for label in body.get("roi_list") or []:
                sign, shape, vertices = rois.roi_vertices(label)
                source["rois"].append({"sign": sign, "vertices": vertices.tolist()})
            if body.get("roi_list") is not None and not source["rois"]:
                return jsonify({"contours": [], "areas": []})
            contours, areas = rois.segment(frame, source)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
            return bad_request(error)
        return jsonify({"contours": contours, "areas": areas})

    @app.route("/api/collector/fit_ellipse", methods=["POST"])
    def collector_fit_ellipse():
        """{"points": [[x, y], ...] (≥ 5)}: the ellipse ROI idtrackerai's GUI makes of them."""
        points = np.asarray((request.get_json(silent=True) or {}).get("points") or [], dtype=np.float32)
        if points.ndim != 2 or len(points) < 5:
            return bad_request("An ellipse needs 5 points or more")
        (cx, cy), (w, h), angle = cv2.fitEllipse(points)
        return jsonify({"center": [round(cx), round(cy)], "axes": [round(w / 2), round(h / 2)], "angle": round(angle)})

    @app.route("/api/collector/arenas_proposal", methods=["POST"])
    def collector_arenas_proposal():
        """{"experiment", "frame_number", "n"}: settings of an arenas.toml splitting the recording
        into n arenas (nothing written: the editor saves them); source detected / grid."""
        body = request.get_json(silent=True) or {}
        try:
            experiment = body["experiment"].strip().strip("/")
            if LOCAL_RE.match(experiment) and LOCAL_RE.match(experiment).group(5):
                raise ValueError("arenas.toml belongs to the whole recording, not to an arena folder")
            folder = experiment_folder(root, experiment)
            n = int(body["n"])
            if not 2 <= n <= MAX_ARENAS:
                raise ValueError(f"n must be between 2 and {MAX_ARENAS}")
            frame_number = int(body["frame_number"])
            chunksize = store_info(folder)["chunksize"]
            frame = FRAMES.get(folder, frame_number, chunksize)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
            return bad_request(f"missing {error}" if isinstance(error, KeyError) else error)
        roi_list, source, detected = propose_arenas(frame, n)
        video = os.path.realpath(os.path.join(folder, f"{frame_number // chunksize:06d}.mp4"))
        settings = {"video_paths": [video], **ARENAS_DEFAULTS, "roi_list": roi_list}
        return jsonify({"settings": settings, "source": source, "detected": detected})

    @app.route("/api/collector/preprocess_sample", methods=["GET"])
    def collector_preprocess_status():
        try:
            folder = experiment_folder(root, request.args.get("experiment"))
        except (ValueError, FileNotFoundError) as error:
            return bad_request(error)
        status = preprocess_status(folder)
        status["arena"] = bool(LOCAL_RE.match(request.args["experiment"].strip().strip("/")).group(5))
        try:
            status["default_range"] = preprocess_chunks(folder)
        except FileNotFoundError as error:
            status["default_range"] = None
            status["range_error"] = str(error)
        status["samples_per_chunk_default"] = PREPROCESS_SAMPLES_PER_CHUNK
        return jsonify(status)

    @app.route("/api/collector/preprocess_sample", methods=["POST"])
    def collector_preprocess_start():
        """{"experiment", "start_chunk", "end_chunk" (inclusive), "samples_per_chunk"}: starts
        preprocess-sample in the background (409 if one checks this folder already)."""
        body = request.get_json(silent=True) or {}
        try:
            folder = experiment_folder(root, body.get("experiment"))
            if LOCAL_RE.match(body["experiment"].strip().strip("/")).group(5):
                raise ValueError("preprocess-sample does not run on arena folders: it reads <folder name>.conf")
            if not os.path.exists(os.path.join(folder, os.path.basename(folder) + ".conf")):
                raise ValueError(f"No {os.path.basename(folder)}.conf: save {os.path.basename(folder)}.toml first")
            if body.get("start_chunk") in (None, "") or body.get("end_chunk") in (None, ""):
                start, end = preprocess_chunks(folder)
            else:
                start, end = int(body["start_chunk"]), int(body["end_chunk"])
            samples = int(body.get("samples_per_chunk") or PREPROCESS_SAMPLES_PER_CHUNK)
            if not 0 <= start <= end or samples < 1:
                raise ValueError("Need 0 ≤ start chunk ≤ end chunk and samples per chunk ≥ 1")
        except (ValueError, FileNotFoundError) as error:
            return bad_request(error)
        if preprocess_process(folder) is not None:
            return jsonify({"error": "preprocess-sample is already running on this experiment"}), 409
        pid = start_preprocess(folder, start, end, samples)
        return jsonify({"pid": pid, "start_chunk": start, "end_chunk": end, "samples_per_chunk": samples})

    @app.route("/api/collector/preprocess_sample/stop", methods=["POST"])
    def collector_preprocess_stop():
        try:
            folder = experiment_folder(root, (request.get_json(silent=True) or {}).get("experiment"))
        except (ValueError, FileNotFoundError) as error:
            return bad_request(error)
        process = preprocess_process(folder)
        if process is None:
            return jsonify({"error": "preprocess-sample is not running on this experiment"}), 409
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()
        return jsonify({"stopped": process.pid})
