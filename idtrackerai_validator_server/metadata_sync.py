"""
metadata_sync.py  —  refresh an experiment's `ethoscope_metadata` from the Google sheets.

Two steps, so nothing is written before it has been looked at:

POST /api/metadata/preview {"experiment"}
    Job running `download_flyhostel_metadata` (googleapi, from flyhostel) for the
    experiment's FlyHostel number and date (and `--arena` for an arena of a
    multi-arena recording, FlyHostel<N>/<X>X/<date>/arenas/<n>). Its result is the new metadata (every row,
    with the row now in the database for the same identity), the columns that change,
    and checks: one row per animal, flyhostel_number / flyhostel_date / flyhostel_time
    matching the experiment path, identities present and unique, genotype filled in,
    and the light cycle encoding (overview.merge_light_cycle).

POST /api/metadata/predownload {"experiments"}
    Job downloading the metadata of many experiments at once (each sheet once, see
    download_metadata_batch.py) and keeping it in memory: the preview of those then
    uses it instead of downloading (`"fresh": true` downloads again). Its result, and
    GET /api/metadata/predownloaded, give per experiment a status for the table:
    same / minor (only non-key columns change) / filled (key values only fill empty
    cells) / changes / error (a failed check) / failed (download).

POST /api/metadata/download_sheets
    Job running `download_metadata_sheets` (googleapi, next to the downloader): the
    tabs listed under SHEETS in the googleapi config, as CSV, into $FLYHOSTEL_METADATA,
    where the Overview reads the experiments not analyzed yet. Result: the files saved.

POST /api/metadata/apply {"experiment", "job_id", "force"}
    Writes the previewed CSV to METADATA.ethoscope_metadata. Refused when a check
    failed unless force. The old value is first saved to
    <experiment>/metadata_backups/ethoscope_metadata_<timestamp>.csv, and so is
    <experiment>/metadata.csv, which is replaced too when it exists (flyhostel uses
    it instead of downloading when the database is made again).

The database is read again on every request (tracking, overview), so the change
shows without restarting the server.
"""
import io
import re
import os
import json
import time
import sqlite3
import logging
import tempfile
import threading
import subprocess

import pandas as pd
from flask import jsonify, request
from flyhostel.utils import get_number_of_animals

from idtrackerai_validator_server.jobs import start_job, _jobs
from idtrackerai_validator_server.experiments import to_flat
from idtrackerai_validator_server.overview import PATH_RE, METADATA_DIR, _clean, _fly_rows, merge_light_cycle

logger = logging.getLogger(__name__)

DOWNLOADER = os.environ.get(
    "DOWNLOAD_FLYHOSTEL_METADATA",
    "/home/vibflysleep/mambaforge/envs/rapids-23.04/bin/download_flyhostel_metadata",
)
DOWNLOAD_TIMEOUT = 300          # s; an expired Google token would otherwise wait for a browser login
# Batch download (download_metadata_batch.py) with the Python of the downloader's environment.
BATCH_PYTHON = os.path.join(os.path.dirname(DOWNLOADER), "python")
BATCH_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_metadata_batch.py")
BATCH_MARKER = "@@PROGRESS "
# Whole sheets as CSV (the browser's File > Download), for the Overview's not-analyzed experiments.
SHEETS_DOWNLOADER = os.path.join(os.path.dirname(DOWNLOADER), "download_metadata_sheets")
SAVED_RE = re.compile(r"Saved (.+\.csv)\s*$", re.M)
_sheets_lock = threading.Lock()
BATCH_TIMEOUT = 3600
# experiment -> {"csv", "downloaded_at", "status", ...}: the last download of its metadata,
# used by the preview instead of downloading again. In memory: lost on restart.
_predownloaded = {}
_predownload_lock = threading.Lock()
# Shown first in the preview, when present.
ARENA_PATH_RE = re.compile(PATH_RE.pattern + r"(?:/arenas/(\d+))?")
KEY_COLUMNS = ["flyhostel_number", "flyhostel_date", "flyhostel_time", "number_of_animals",
               "identity", "wing", "genotype", "sex", "light_cycle"]


def _parse(experiment):
    """Fields of an experiment folder, FlyHostel<N>/<X>X/<date> or an arena of one
    (FlyHostel<N>/<X>X/<date>/arenas/<n>, the sheet rows of that arena)."""
    match = ARENA_PATH_RE.fullmatch(experiment or "")
    if not match:
        raise ValueError(f"Not an experiment path: {experiment!r}")
    arena = int(match.group(5)) if match.group(5) else None
    animals = int(match.group(2))
    if arena is not None:
        # the NX of the folder counts the whole hostel; the arena's are in its idtrackerai
        # conf or database (None when it has neither yet)
        try:
            animals = int(get_number_of_animals(to_flat(experiment)))
        except Exception as error:
            logger.warning("Number of animals of %s unknown: %s", experiment, error)
            animals = None
    return {
        "hostel": int(match.group(1)[len("FlyHostel"):]),
        "animals": animals,
        "date": match.group(3),
        "time": match.group(4),
        "arena": arena,
    }


def _paths(experiment):
    folder = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], experiment)
    return folder, os.path.join(folder, to_flat(experiment) + ".db")


def _read_current(dbfile):
    with sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True, timeout=5) as conn:
        row = conn.execute("SELECT value FROM METADATA WHERE field = 'ethoscope_metadata'").fetchone()
    return row[0] if row and row[0] else ""


def _rows(csv_text):
    """[{column: cleaned str}] of a metadata CSV (first column = index)."""
    if not csv_text.strip():
        return [], []
    df = pd.read_csv(io.StringIO(csv_text), index_col=0, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    rows = [{c: _clean(row[c]) for c in df.columns} for _, row in df.iterrows()]
    return rows, list(df.columns)


def check_metadata(experiment, rows, columns):
    """[{level: ok|warning|error, text}] of the new metadata."""
    info = _parse(experiment)
    checks = []
    add = lambda level, text: checks.append({"level": level, "text": text})

    n = len(rows)
    if info["animals"] is None:
        add("warning", f"{n} row{'s' if n != 1 else ''}, number of animals of the arena unknown")
    else:
        add("ok" if n == info["animals"] else "error",
            f"{n} row{'s' if n != 1 else ''} for {info['animals']} animal{'s' if info['animals'] != 1 else ''} ({info['animals']}X)")
    if n == 0:
        add("error", "no row of the sheets has this flyhostel_number and flyhostel_date"
            + (f" (and arena {info['arena']})" if info["arena"] is not None else ""))
        return checks

    def column_check(column, expected, level, label):
        if column not in columns:
            add("warning", f"no {column} column")
            return
        values = [r.get(column) for r in rows]
        wrong = sorted({str(v) for v in values if v != expected})
        if wrong:
            add(level, f"{label}: {', '.join(wrong)} (expected {expected})")
        else:
            add("ok", f"{label} {expected}")

    column_check("flyhostel_number", str(info["hostel"]), "error", "flyhostel_number")
    column_check("flyhostel_date", info["date"], "error", "flyhostel_date")
    column_check("flyhostel_time", info["time"], "warning", "flyhostel_time")
    if info["animals"] is not None:
        column_check("number_of_animals", str(info["animals"]), "warning", "number_of_animals")
    if info["arena"] is not None:
        column_check("arena", str(info["arena"]), "error", "arena")

    if "identity" not in columns:
        add("error" if info["animals"] > 1 else "warning", "no identity column")
    else:
        identities = [r.get("identity") for r in rows]
        missing = sum(i is None for i in identities)
        present = [i for i in identities if i is not None]
        duplicated = sorted({i for i in present if present.count(i) > 1})
        if missing:
            add("error" if info["animals"] > 1 else "warning", f"identity missing in {missing} row(s)")
        if duplicated:
            add("error", f"identity repeated: {', '.join(duplicated)}")
        # idtrackerai's identities: 0 for a fly alone, 1..N in a group
        expected = {0} if info["animals"] == 1 else set(range(1, info["animals"] + 1)) if info["animals"] else None
        try:
            numbers = {int(float(i)) for i in present}
        except ValueError:
            numbers = None
        unexpected = expected is not None and not missing and not duplicated and numbers != expected
        if unexpected:
            add("error", f"identities {', '.join(sorted(present, key=lambda v: (len(v), v)))} "
                         f"(expected {', '.join(map(str, sorted(expected)))} for {info['animals']} "
                         f"animal{'s' if info['animals'] > 1 else ''})")
        elif not missing and not duplicated:
            add("ok", f"identities {', '.join(sorted(present, key=lambda v: (len(v), v)))}")

    no_genotype = sum(r.get("genotype") is None for r in rows)
    add("warning" if no_genotype else "ok",
        f"genotype missing in {no_genotype} row(s)" if no_genotype
        else "genotype: " + ", ".join(sorted({r["genotype"] for r in rows})))
    if "wing" in columns:
        add("ok", "wing: " + ", ".join(str(r.get("wing")) for r in rows))

    flies = [{k: v for k, v in r.items() if v is not None} for r in rows]
    flags = merge_light_cycle(flies)
    cycles = sorted({f["light_cycle"] for f in flies if f.get("light_cycle")})
    if flags:
        add("warning", f"light cycle {', '.join(cycles) or '?'}, non-standard encoding: {'; '.join(flags)}")
    elif cycles:
        add("ok", f"light cycle {', '.join(cycles)} in light_cycle")
    else:
        add("warning", "no light cycle found")
    return checks


def _pair(new_rows, old_rows):
    """The old row of each new row: same identity when identities are unique, else same position."""
    def by_identity(rows):
        ids = [r.get("identity") for r in rows]
        return None if None in ids or len(set(ids)) != len(ids) else dict(zip(ids, rows))
    new_ids, old_ids = by_identity(new_rows), by_identity(old_rows)
    if new_ids is not None and old_ids is not None:
        return [old_ids.get(r.get("identity")) for r in new_rows]
    return [old_rows[i] if i < len(old_rows) else None for i in range(len(new_rows))]


def _download(experiment, workdir):
    """CSV text written by download_flyhostel_metadata for one experiment ("" if none)."""
    info = _parse(experiment)
    if not os.path.exists(DOWNLOADER):
        raise FileNotFoundError(f"{DOWNLOADER} not found (set DOWNLOAD_FLYHOSTEL_METADATA)")
    output = os.path.join(workdir, "metadata.csv")
    cmd = [DOWNLOADER, "--metadata", output, "--flyhostel-id", str(info["hostel"]),
           "--date-time", f"{info['date']}_{info['time']}"]
    if info["animals"] is not None:
        cmd += ["--number-of-animals", str(info["animals"])]
    if info["arena"] is not None:
        cmd += ["--arena", str(info["arena"])]
    try:
        result = subprocess.run(cmd, cwd=workdir, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=DOWNLOAD_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"download_flyhostel_metadata took more than {DOWNLOAD_TIMEOUT} s "
                           "(is the Google token in ~/.config/googleapi still valid?)")
    if result.returncode != 0:
        raise RuntimeError(f"download_flyhostel_metadata failed: {(result.stderr or result.stdout).strip()[-1000:]}")
    return open(output, encoding="utf8").read() if os.path.exists(output) else ""


def compare(experiment, csv_text):
    """The preview of writing `csv_text`: new rows next to the database's, changes, checks."""
    _, dbfile = _paths(experiment)
    new_rows, columns = _rows(csv_text)
    old_csv = _read_current(dbfile)
    old_rows, old_columns = _rows(old_csv)
    previous = _pair(new_rows, old_rows)
    changed = [c for c in dict.fromkeys(columns + old_columns)
               if len(new_rows) != len(old_rows)
               or any((r.get(c) if r else None) != (p.get(c) if p else None) for r, p in zip(new_rows, previous))]
    return {
        "experiment": experiment,
        "rows": new_rows,
        "previous": previous,
        "n_previous": len(old_rows),
        "columns": columns,
        "key_columns": [c for c in KEY_COLUMNS if c in columns or c in old_columns],
        "changed_columns": changed,
        "unchanged": csv_text.strip() == old_csv.strip(),
        "checks": check_metadata(experiment, new_rows, columns),
    }


def _status(result, downloaded_at):
    """What the experiments table shows about a downloaded CSV. "changes": rows added or
    removed, or a key value changes or disappears; "filled": key values only appear where
    the database had none (e.g. light_cycle, added to the sheets after most databases
    were made); "minor": only other columns change."""
    levels = [c["level"] for c in result["checks"]]
    rows_change = len(result["rows"]) != result["n_previous"]
    # key column -> "changed" (a value differs or is gone) or "filled" (only empty cells get a value)
    key_changes = {}
    for row, old in zip(result["rows"], result["previous"]):
        for column in KEY_COLUMNS:
            new_value, old_value = row.get(column), (old or {}).get(column)
            if new_value != old_value:
                kind = "filled" if old is not None and old_value is None else "changed"
                if key_changes.get(column) != "changed":
                    key_changes[column] = kind
    if "error" in levels:
        status = "error"
    elif result["unchanged"] or not result["changed_columns"]:
        status = "same"
    elif rows_change or "changed" in key_changes.values():
        status = "changes"
    else:
        status = "filled" if key_changes else "minor"
    return {
        "downloaded_at": downloaded_at,
        "status": status,
        "key_changes": [c for c, kind in key_changes.items() if kind == "changed"] + (["rows"] if rows_change else []),
        "key_filled": [c for c, kind in key_changes.items() if kind == "filled"],
        "errors": [c["text"] for c in result["checks"] if c["level"] == "error"],
        "warnings": levels.count("warning"),
        "changed_columns": result["changed_columns"],
    }


def preview(experiment, job, fresh=False):
    """Preview from the predownloaded CSV, or downloaded now when there is none or `fresh`."""
    _, dbfile = _paths(experiment)
    if not os.path.exists(dbfile):
        raise FileNotFoundError(f"{dbfile} does not exist")
    job.update(0, status="running", progress=0.1)
    cached = None if fresh else _predownloaded.get(experiment)
    if cached:
        csv_text, downloaded_at = cached["csv"], cached["downloaded_at"]
    else:
        job.workdir = tempfile.mkdtemp(prefix="metadata_")
        csv_text, downloaded_at = _download(experiment, job.workdir), time.time()
    job.update(0, progress=0.8)

    result = compare(experiment, csv_text)
    result.update(downloaded_at=downloaded_at, predownloaded=cached is not None)
    _predownloaded[experiment] = {"csv": csv_text, "downloaded_at": downloaded_at, **_status(result, downloaded_at)}
    job.pending = {"experiment": experiment, "csv": csv_text}
    job.result = result
    logger.info("New metadata of %s:\n%s", experiment,
                pd.DataFrame(result["rows"], columns=result["key_columns"] or None).to_string()
                if result["rows"] else "(no rows)")
    job.update(0, status="done", progress=1.0)


def predownload(experiments, job):
    """Download the sheets once and keep the metadata of every experiment in _predownloaded."""
    if not os.path.exists(BATCH_PYTHON):
        raise FileNotFoundError(f"{BATCH_PYTHON} not found (python next to download_flyhostel_metadata)")
    job.workdir = tempfile.mkdtemp(prefix="metadata_batch_")
    todo = []
    for experiment in experiments:
        info = _parse(experiment)
        todo.append({"experiment": experiment, "hostel": info["hostel"], "date_time": f"{info['date']}_{info['time']}",
                     "arena": info["arena"]})

    job.update(0, status="running", progress=0.0)
    process = subprocess.Popen(
        [BATCH_PYTHON, BATCH_SCRIPT], cwd=job.workdir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    timer = threading.Timer(BATCH_TIMEOUT, process.kill)
    timer.start()
    failed = {}
    try:
        process.stdin.write(json.dumps({"outdir": job.workdir, "experiments": todo}))
        process.stdin.close()
        stderr = []
        threading.Thread(target=lambda: stderr.extend(process.stderr), daemon=True).start()
        for line in process.stdout:
            if not line.startswith(BATCH_MARKER):
                continue
            step = json.loads(line[len(BATCH_MARKER):])
            if step["error"]:
                failed[step["experiment"]] = step["error"]
            job.update(0, progress=step["done"] / max(1, step["total"]))
        process.wait()
    finally:
        timer.cancel()
    if process.returncode != 0:
        raise RuntimeError(f"metadata download stopped (exit {process.returncode}): {''.join(stderr).strip()[-1000:]}")

    downloaded_at = time.time()
    summary = {}
    for experiment in experiments:
        if experiment in failed:
            summary[experiment] = {"downloaded_at": downloaded_at, "status": "failed", "errors": [failed[experiment]],
                                   "warnings": 0, "changed_columns": [], "key_changes": [], "key_filled": []}
            continue
        path = os.path.join(job.workdir, experiment.replace("/", "_") + ".csv")
        csv_text = open(path, encoding="utf8").read() if os.path.exists(path) else ""
        try:
            status = _status(compare(experiment, csv_text), downloaded_at)
        except Exception as error:     # e.g. a database that cannot be read
            status = {"downloaded_at": downloaded_at, "status": "failed", "errors": [str(error)],
                      "warnings": 0, "changed_columns": [], "key_changes": [], "key_filled": []}
        else:
            _predownloaded[experiment] = {"csv": csv_text, **status}
        summary[experiment] = status
    job.result = {"summary": summary}
    job.update(0, status="done", progress=1.0)


def download_sheets(job):
    """Run download_metadata_sheets into METADATA_DIR; job.result = {"files", "output"}."""
    if not os.path.exists(SHEETS_DOWNLOADER):
        raise FileNotFoundError(f"{SHEETS_DOWNLOADER} not found (googleapi too old?)")
    job.update(0, status="running", progress=0.2)
    try:
        result = subprocess.run([SHEETS_DOWNLOADER, "--output", METADATA_DIR], stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"download_metadata_sheets took more than {DOWNLOAD_TIMEOUT} s "
                           "(is the Google token in ~/.config/googleapi still valid?)")
    output = (result.stderr or "") + (result.stdout or "")
    if result.returncode != 0:
        raise RuntimeError(f"download_metadata_sheets failed: {output.strip()[-1000:]}")
    files = SAVED_RE.findall(output)
    logger.info("Metadata sheets downloaded to %s: %s", METADATA_DIR, files)
    job.result = {"files": files, "output": METADATA_DIR}
    job.update(0, status="done", progress=1.0, count=len(files))


def predownloaded_status():
    return {e: {k: v for k, v in entry.items() if k != "csv"} for e, entry in _predownloaded.items()}


def apply(experiment, job_id, force=False):
    job = _jobs.get(str(job_id))
    pending = getattr(job, "pending", None)
    if job is None or pending is None or pending["experiment"] != experiment:
        raise LookupError("No preview of this experiment to write: download it again")
    errors = [c["text"] for c in job.result["checks"] if c["level"] == "error"]
    if errors and not force:
        raise PermissionError("Checks failed: " + "; ".join(errors))

    folder, dbfile = _paths(experiment)
    backups = os.path.join(folder, "metadata_backups")
    os.makedirs(backups, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = os.path.join(backups, f"ethoscope_metadata_{stamp}.csv")
    with open(backup, "w", encoding="utf8") as handle:
        handle.write(_read_current(dbfile))

    with sqlite3.connect(dbfile, timeout=30) as conn:     # one transaction
        conn.execute("DELETE FROM METADATA WHERE field = 'ethoscope_metadata'")
        conn.execute("INSERT INTO METADATA (field, value) VALUES (?, ?)", ("ethoscope_metadata", pending["csv"]))

    metadata_csv = os.path.join(folder, "metadata.csv")
    replaced_csv = False
    if os.path.exists(metadata_csv):
        os.replace(metadata_csv, os.path.join(backups, f"metadata_{stamp}.csv"))
        with open(metadata_csv, "w", encoding="utf8") as handle:
            handle.write(pending["csv"])
        replaced_csv = True

    job.pending = None                  # written once
    if experiment in _predownloaded:
        _predownloaded[experiment].update(status="same", errors=[], changed_columns=[], key_changes=[], key_filled=[])
    logger.info("ethoscope_metadata of %s updated from the sheets (backup %s)", experiment, backup)
    return {"experiment": experiment, "backup": backup, "metadata_csv": replaced_csv,
            "rows": len(job.result["rows"]), "forced": bool(errors)}


def register_metadata_sync(app):

    @app.route("/api/metadata/preview", methods=["POST"])
    def metadata_preview():
        body = request.get_json(silent=True) or {}
        experiment, fresh = body.get("experiment"), bool(body.get("fresh"))
        try:
            _parse(experiment)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        cached = experiment in _predownloaded and not fresh
        job = start_job(f"Metadata of {experiment}", ["predownloaded" if cached else "download from the sheets"],
                        lambda job: preview(experiment, job, fresh=fresh))
        return jsonify(job.to_dict())

    @app.route("/api/metadata/predownload", methods=["POST"])
    def metadata_predownload():
        """{"experiments": [...]}: download the metadata of all of them, for instant previews."""
        experiments = list(dict.fromkeys((request.get_json(silent=True) or {}).get("experiments") or []))
        try:
            for experiment in experiments:
                _parse(experiment)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        if not experiments:
            return jsonify({"error": "No experiment to download"}), 400
        if not _predownload_lock.acquire(blocking=False):
            return jsonify({"error": "The metadata is already being downloaded"}), 409

        def work(job):
            try:
                predownload(experiments, job)
            finally:
                _predownload_lock.release()

        try:
            job = start_job(f"Download sheet metadata of {len(experiments)} experiments", ["sheets"], work)
        except Exception:
            _predownload_lock.release()
            raise
        return jsonify(job.to_dict())

    @app.route("/api/metadata/download_sheets", methods=["POST"])
    def metadata_download_sheets():
        """Download the SHEETS tabs as CSV into $FLYHOSTEL_METADATA (one at a time)."""
        if not _sheets_lock.acquire(blocking=False):
            return jsonify({"error": "The metadata sheets are already being downloaded"}), 409

        def work(job):
            try:
                download_sheets(job)
            finally:
                _sheets_lock.release()

        try:
            job = start_job("Download metadata sheets", ["sheets"], work)
        except Exception:
            _sheets_lock.release()
            raise
        return jsonify(job.to_dict())

    @app.route("/api/metadata/predownloaded", methods=["GET"])
    def metadata_predownloaded():
        return jsonify(predownloaded_status())

    @app.route("/api/metadata/apply", methods=["POST"])
    def metadata_apply():
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(apply(body.get("experiment"), body.get("job_id"), bool(body.get("force"))))
        except (LookupError, PermissionError) as error:
            return jsonify({"error": str(error)}), 409
        except Exception as error:
            logger.exception("Could not write the metadata")
            return jsonify({"error": str(error)}), 500
