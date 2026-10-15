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
"""
import io
import os
import re
import math
import sqlite3
import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from flask import jsonify, request

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

_cache = {}                 # db path -> (mtime, record)
_scan_lock = threading.Lock()


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


def _read_experiment(root, relpath):
    path = os.path.join(root, relpath)
    experiment = os.path.dirname(os.path.normpath(relpath))
    match = PATH_RE.search(experiment)
    record = {
        "experiment": experiment,
        "flyhostel": match.group(1) if match else None,
        "group_size": int(match.group(2)) if match else None,
        "date": match.group(3) if match else None,
        "start_time": match.group(4).replace("-", ":") if match else None,
        "has_val": False,
        "duration_h": None,
        "flies": [],
        "light_cycle_flags": [],
        "error": None,
    }
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
            df.columns = [str(c).strip() for c in df.columns]
            keep = [c for c in df.columns if COLUMN_RE.fullmatch(c) and c not in SKIP_COLUMNS]
            for _, row in df[keep].iterrows():
                fly = {}
                for column in keep:
                    value = _clean(row[column])
                    if value is not None:
                        fly[column] = value
                record["flies"].append(fly)
            record["light_cycle_flags"] = merge_light_cycle(record["flies"])
    except Exception as error:     # a broken database must not hide the others
        record["error"] = str(error)
    return record


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

    coverage = Counter(column for record in records for fly in record["flies"] for column in fly)
    return {
        "experiments": records,
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
