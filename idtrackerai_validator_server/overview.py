"""
overview.py  —  metadata of every experiment listed in $FLYHOSTEL_VIDEOS/index.txt,
for the frontend's Overview tab.

GET /api/overview returns one record per experiment (path-derived fields, a few
METADATA fields, whether validated tables exist) with one record per fly taken
from its `ethoscope_metadata` table. The fly columns are not fixed: every column
found in any experiment is returned, ordered by how many flies have a value.

The light cycle is not always in `light_cycle`: depending on the metadata sheet it
was written in `condition`, `experiment` or appended to `vial` ("sucrose vial LD").
`merge_light_cycle` gathers it into `light_cycle` (the database is not changed),
records where it came from in `light_cycle_source`, and lists every non-standard
case in the experiment's `light_cycle_flags`.

Reading ~500 databases takes ~10 s, so each one is cached until its mtime changes;
`?refresh=1` drops the cache.

Experiments not analyzed yet (folder in $FLYHOSTEL_VIDEOS but no database) are
taken from the metadata sheets: the CSV exports in $FLYHOSTEL_METADATA of every
sheet listed under SHEETS in the googleapi config. They come with `analyzed: False`
and their flies from the sheet rows.

POST /api/overview/reindex rewrites index.txt in a background job (jobs.py), with
the same command as by hand: find -L -maxdepth 4 -mindepth 4 -regex .*FlyHostel.*db -not -name index.db
"""
import io
import os
import re
import glob
import subprocess
import math
import sqlite3
import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import yaml
import pandas as pd
from flask import jsonify, request

from idtrackerai_validator_server.jobs import start_job

logger = logging.getLogger(__name__)

# Values meaning "not filled in" in the metadata sheets.
MISSING = {"", "nan", "none", "na", "n/a", "null"}
# Columns that are spreadsheet debris (" ", "_", "Unnamed: 3", "03-02") are dropped.
COLUMN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
# Per-experiment, path-derived or redundant with the experiment record.
SKIP_COLUMNS = {"Dropbox_link", "basedir", "flyhostel_number", "flyhostel_date", "flyhostel_time", "row_id"}
PATH_RE = re.compile(r"(FlyHostel\d+)/(\d+)X/(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")

# Where the light cycle may be written, in order of trust. Outside `light_cycle`
# the value is taken as a whole word, so "LD+2blob" and "normal food vial DD" count.
LIGHT_CYCLE_COLUMNS = ("light_cycle", "condition", "experiment", "vial")
LIGHT_CYCLES = {"LD", "DD", "LL"}
LIGHT_CYCLE_RE = re.compile(r"\b(LD|DD|LL)\b")

GOOGLEAPI_CONFIG = os.path.expanduser(os.environ.get("GOOGLEAPI_CONFIG", "~/.config/googleapi/config.yaml"))
METADATA_DIR = os.environ.get("FLYHOSTEL_METADATA", "/flyhostel_data/metadata")
# CSV export of a sheet: "ethoscope&flyhostel_metadata<version> - <sheet>.csv", latest version wins
SHEET_FILE_RE = re.compile(r"metadata(\d+) - (.+)\.csv$")
INDEX_COMMAND = ["find", "-L", "-maxdepth", "4", "-mindepth", "4", "-regex", ".*FlyHostel.*db", "-not", "-name", "index.db"]

_cache = {}                 # db path -> (mtime, record)
_scan_lock = threading.Lock()
_reindex_lock = threading.Lock()


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            value = int(value)
    text = str(value).strip()
    return None if text.lower() in MISSING else text


def _light_cycle_of(fly):
    """[(column, value, raw)] of every column holding a light cycle."""
    found = []
    for column in LIGHT_CYCLE_COLUMNS:
        raw = fly.get(column)
        if raw is None:
            continue
        if raw.upper() in LIGHT_CYCLES:
            found.append((column, raw.upper(), raw))
        elif column != "light_cycle":
            cycles = set(LIGHT_CYCLE_RE.findall(raw.upper()))
            if len(cycles) == 1:
                found.append((column, cycles.pop(), raw))
    return found


def merge_light_cycle(flies):
    """Set `light_cycle` and `light_cycle_source` of every fly; return the flags.

    A fly without a light cycle of its own takes the one of the other flies of
    the experiment, if they agree (they share the chamber).
    """
    flags = []
    raw_light_cycle = {}
    for i, fly in enumerate(flies):
        found = _light_cycle_of(fly)
        if fly.get("light_cycle") is not None and not found:
            raw_light_cycle[i] = fly["light_cycle"]          # unrecognised value: keep it
        if not found:
            continue
        column, value, raw = found[0]
        fly["light_cycle"] = value
        fly["light_cycle_source"] = column
        if column != "light_cycle":
            flags.append(f"in {column}" + ("" if raw.upper() == value else f" (\"{raw}\")"))
        others = {v for _, v, _ in found[1:]}
        if others - {value}:
            flags.append("conflict: " + ", ".join(f"{c} = \"{r}\"" for c, _, r in found))

    values = {fly["light_cycle"] for fly in flies if fly.get("light_cycle_source")}
    missing = [fly for fly in flies if not fly.get("light_cycle_source")]
    if len(values) > 1:
        flags.append("flies differ: " + ", ".join(sorted(values)))
    elif len(values) == 1 and missing:
        value = values.pop()
        for fly in missing:
            if fly.get("light_cycle") is None:
                fly["light_cycle"] = value
                fly["light_cycle_source"] = "other flies"
        flags.append(f"missing for {len(missing)} of {len(flies)} flies, filled from the others")
    for i, value in raw_light_cycle.items():
        flags.append(f"unrecognised light_cycle \"{value}\"")

    return sorted(set(flags), key=flags.index)


def _base_record(experiment):
    match = PATH_RE.search(experiment)
    return {
        "experiment": experiment,
        "flyhostel": match.group(1) if match else None,
        "group_size": int(match.group(2)) if match else None,
        "date": match.group(3) if match else None,
        "start_time": match.group(4).replace("-", ":") if match else None,
        "has_val": False,
        "duration_h": None,
        "flies": [],
        "light_cycle_flags": [],
        "analyzed": True,
        "error": None,
    }


def _fly_rows(df):
    """One dict per row of an ethoscope_metadata table / sheet, junk columns and empty values dropped."""
    df.columns = [str(c).strip() for c in df.columns]
    keep = [c for c in df.columns if COLUMN_RE.fullmatch(c) and c not in SKIP_COLUMNS]
    flies = []
    for _, row in df[keep].iterrows():
        fly = {}
        for column in keep:
            value = _clean(row[column])
            if value is not None:
                fly[column] = value
        flies.append(fly)
    return flies


def _read_experiment(root, relpath):
    path = os.path.join(root, relpath)
    record = _base_record(os.path.dirname(os.path.normpath(relpath)))
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5) as conn:
            fields = dict(conn.execute(
                "SELECT field, value FROM METADATA WHERE field IN "
                "('ethoscope_metadata', 'chunks', 'chunksize', 'framerate')"
            ).fetchall())
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        record["has_val"] = "IDENTITY_VAL" in tables

        try:
            first, last = (int(c) for c in fields["chunks"].split(","))
            seconds = (last - first + 1) * float(fields["chunksize"]) / float(fields["framerate"])
            record["duration_h"] = round(seconds / 3600, 1)
        except (KeyError, ValueError, ZeroDivisionError):
            pass

        if fields.get("ethoscope_metadata"):
            df = pd.read_csv(io.StringIO(fields["ethoscope_metadata"]), index_col=0, dtype=str)
            record["flies"] = _fly_rows(df)
            record["light_cycle_flags"] = merge_light_cycle(record["flies"])
    except Exception as error:     # a broken database must not hide the others
        record["error"] = str(error)
    return record


def sheet_files():
    """{sheet: csv path} of the sheets listed in the googleapi config, latest export of each."""
    with open(GOOGLEAPI_CONFIG) as handle:
        sheets = yaml.safe_load(handle).get("SHEETS") or []
    latest = {}
    for path in glob.glob(os.path.join(METADATA_DIR, "*.csv")):
        match = SHEET_FILE_RE.search(os.path.basename(path))
        if match and match.group(2) in sheets:
            version = int(match.group(1))
            if match.group(2) not in latest or version > latest[match.group(2)][0]:
                latest[match.group(2)] = (version, path)
    return {sheet: latest[sheet][1] for sheet in sheets if sheet in latest}, \
        [sheet for sheet in sheets if sheet not in latest]


def _has_database(folder):
    """Same rule as the index command: a *db file in the folder other than index.db."""
    return any(os.path.basename(p) != "index.db" for p in glob.glob(os.path.join(folder, "*db")))


def scan_not_analyzed(root, known):
    """Experiments of the metadata sheets with a folder in `root` but no database.

    `known`: experiments already read from index.txt. An experiment in several sheets
    takes its flies from the first one (config order) and lists them all in `sheets`.
    Returns (records, warnings).
    """
    files, missing = sheet_files()
    warnings = [f"no CSV export of sheet \"{sheet}\" in {METADATA_DIR}" for sheet in missing]
    records = {}
    for sheet, path in files.items():
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as error:
            warnings.append(f"could not read {os.path.basename(path)}: {error}")
            continue
        if "basedir" not in df.columns:
            warnings.append(f"{os.path.basename(path)} has no basedir column")
            continue
        experiments = df["basedir"].fillna("").map(lambda b: (m := PATH_RE.search(b)) and m.group(0))
        for experiment, rows in df.groupby(experiments, sort=False):
            if not experiment or experiment in known:
                continue
            if experiment in records:
                records[experiment]["sheets"].append(sheet)
                continue
            folder = os.path.join(root, experiment)
            if not os.path.isdir(folder) or _has_database(folder):
                continue
            record = _base_record(experiment)
            record.update(analyzed=False, sheets=[sheet], flies=_fly_rows(rows.copy()))
            record["light_cycle_flags"] = merge_light_cycle(record["flies"])
            records[experiment] = record
    return list(records.values()), warnings


def rebuild_index(root):
    """Rewrite index.txt; returns the number of databases listed. The old file is only
    replaced once find has produced a list."""
    result = subprocess.run(INDEX_COMMAND, cwd=root, capture_output=True, text=True, timeout=600)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"find listed no database (exit {result.returncode}): {result.stderr.strip()[:500]}")
    if result.returncode != 0:      # e.g. a broken link or an unreadable folder: keep what was found
        logger.warning("find exited with %s: %s", result.returncode, result.stderr.strip()[:2000])
    tmp = os.path.join(root, "index.txt.tmp")
    with open(tmp, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(tmp, os.path.join(root, "index.txt"))
    return len(lines)


def scan(root, refresh=False):
    with open(os.path.join(root, "index.txt")) as handle:
        paths = [line.strip() for line in handle if line.strip()]

    with _scan_lock:
        if refresh:
            _cache.clear()

        def load(relpath):
            path = os.path.join(root, relpath)
            try:
                mtime = os.path.getmtime(path)
            except OSError as error:
                return {"experiment": os.path.dirname(os.path.normpath(relpath)), "flies": [],
                        "light_cycle_flags": [], "error": str(error)}
            cached = _cache.get(path)
            if cached and cached[0] == mtime:
                return cached[1]
            record = _read_experiment(root, relpath)
            _cache[path] = (mtime, record)
            return record

        with ThreadPoolExecutor(max_workers=8) as pool:
            records = list(pool.map(load, paths))

    warnings = []
    try:
        pending, warnings = scan_not_analyzed(root, {record["experiment"] for record in records})
        records += pending
    except Exception as error:     # the analyzed experiments are still worth showing
        logger.exception("Could not read the metadata sheets")
        warnings.append(f"could not read the metadata sheets: {error}")

    coverage = Counter(column for record in records for fly in record["flies"] for column in fly)
    return {
        "experiments": records,
        "warnings": warnings,
        # every fly-level column, the most often filled in first
        "fly_columns": [{"name": name, "count": count} for name, count in coverage.most_common()],
        "n_flies": sum(len(record["flies"]) for record in records),
    }


def register_overview(app):

    @app.route("/api/overview", methods=["GET"])
    def get_overview():
        root = os.environ["FLYHOSTEL_VIDEOS"]
        try:
            return jsonify(scan(root, refresh=request.args.get("refresh") == "1"))
        except Exception as error:
            logger.exception("Could not build the overview")
            return jsonify({"error": str(error)}), 500

    @app.route("/api/overview/reindex", methods=["POST"])
    def reindex():
        """Start a job rewriting $FLYHOSTEL_VIDEOS/index.txt (one at a time)."""
        root = os.environ["FLYHOSTEL_VIDEOS"]
        if not _reindex_lock.acquire(blocking=False):
            return jsonify({"error": "index.txt is already being rebuilt"}), 409

        def work(job):
            try:
                job.update(0, status="running", progress=0.5)
                count = rebuild_index(root)
                job.update(0, status="done", progress=1.0, count=count)
            finally:
                _reindex_lock.release()

        try:
            job = start_job("Rebuild index.txt", ["index.txt"], work)
        except Exception:
            _reindex_lock.release()
            raise
        return jsonify(job.to_dict())
