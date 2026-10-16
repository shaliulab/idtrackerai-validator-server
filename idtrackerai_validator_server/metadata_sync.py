"""
metadata_sync.py  —  refresh an experiment's `ethoscope_metadata` from the Google sheets.

Two steps, so nothing is written before it has been looked at:

POST /api/metadata/preview {"experiment"}
    Job running `download_flyhostel_metadata` (googleapi, from flyhostel) for the
    experiment's FlyHostel number and date. Its result is the new metadata (every row,
    with the row now in the database for the same identity), the columns that change,
    and checks: one row per animal, flyhostel_number / flyhostel_date / flyhostel_time
    matching the experiment path, identities present and unique, genotype filled in,
    and the light cycle encoding (overview.merge_light_cycle).

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
import os
import time
import sqlite3
import logging
import tempfile
import subprocess

import pandas as pd
from flask import jsonify, request

from idtrackerai_validator_server.jobs import start_job, _jobs
from idtrackerai_validator_server.overview import PATH_RE, _clean, _fly_rows, merge_light_cycle

logger = logging.getLogger(__name__)

DOWNLOADER = os.environ.get(
    "DOWNLOAD_FLYHOSTEL_METADATA",
    "/home/vibflysleep/mambaforge/envs/rapids-23.04/bin/download_flyhostel_metadata",
)
DOWNLOAD_TIMEOUT = 300          # s; an expired Google token would otherwise wait for a browser login
# Shown first in the preview, when present.
KEY_COLUMNS = ["flyhostel_number", "flyhostel_date", "flyhostel_time", "number_of_animals",
               "identity", "wing", "genotype", "sex", "light_cycle"]


def _parse(experiment):
    match = PATH_RE.fullmatch(experiment or "")
    if not match:
        raise ValueError(f"Not an experiment path: {experiment!r}")
    return {
        "hostel": int(match.group(1)[len("FlyHostel"):]),
        "animals": int(match.group(2)),
        "date": match.group(3),
        "time": match.group(4),
    }


def _paths(experiment):
    folder = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], experiment)
    return folder, os.path.join(folder, experiment.replace("/", "_") + ".db")


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
    add("ok" if n == info["animals"] else "error",
        f"{n} row{'s' if n != 1 else ''} for {info['animals']} animal{'s' if info['animals'] != 1 else ''} ({info['animals']}X)")
    if n == 0:
        add("error", "no row of the sheets has this flyhostel_number and flyhostel_date")
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
    column_check("number_of_animals", str(info["animals"]), "warning", "number_of_animals")

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
        if not missing and not duplicated:
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


def preview(experiment, job):
    info = _parse(experiment)
    folder, dbfile = _paths(experiment)
    if not os.path.exists(dbfile):
        raise FileNotFoundError(f"{dbfile} does not exist")
    if not os.path.exists(DOWNLOADER):
        raise FileNotFoundError(f"{DOWNLOADER} not found (set DOWNLOAD_FLYHOSTEL_METADATA)")

    job.update(0, status="running", progress=0.1)
    job.workdir = tempfile.mkdtemp(prefix="metadata_")
    output = os.path.join(job.workdir, "metadata.csv")
    cmd = [DOWNLOADER, "--metadata", output, "--flyhostel-id", str(info["hostel"]),
           "--date-time", f"{info['date']}_{info['time']}", "--number-of-animals", str(info["animals"])]
    try:
        result = subprocess.run(cmd, cwd=job.workdir, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=DOWNLOAD_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"download_flyhostel_metadata took more than {DOWNLOAD_TIMEOUT} s "
                           "(is the Google token in ~/.config/googleapi still valid?)")
    if result.returncode != 0:
        raise RuntimeError(f"download_flyhostel_metadata failed: {(result.stderr or result.stdout).strip()[-1000:]}")
    job.update(0, progress=0.8)

    csv_text = open(output, encoding="utf8").read() if os.path.exists(output) else ""
    new_rows, columns = _rows(csv_text)
    old_csv = _read_current(dbfile)
    old_rows, old_columns = _rows(old_csv)
    previous = _pair(new_rows, old_rows)

    changed = [c for c in dict.fromkeys(columns + old_columns)
               if len(new_rows) != len(old_rows)
               or any((r.get(c) if r else None) != (p.get(c) if p else None) for r, p in zip(new_rows, previous))]
    key = [c for c in KEY_COLUMNS if c in columns or c in old_columns]
    job.pending = {"experiment": experiment, "csv": csv_text}
    job.result = {
        "experiment": experiment,
        "rows": new_rows,
        "previous": previous,
        "n_previous": len(old_rows),
        "columns": columns,
        "key_columns": key,
        "changed_columns": changed,
        "unchanged": csv_text.strip() == old_csv.strip(),
        "checks": check_metadata(experiment, new_rows, columns),
    }
    logger.info("New metadata of %s:\n%s", experiment,
                pd.DataFrame(new_rows, columns=key or None).to_string() if new_rows else "(no rows)")
    job.update(0, status="done", progress=1.0)


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
    logger.info("ethoscope_metadata of %s updated from the sheets (backup %s)", experiment, backup)
    return {"experiment": experiment, "backup": backup, "metadata_csv": replaced_csv,
            "rows": len(job.result["rows"]), "forced": bool(errors)}


def register_metadata_sync(app):

    @app.route("/api/metadata/preview", methods=["POST"])
    def metadata_preview():
        experiment = (request.get_json(silent=True) or {}).get("experiment")
        try:
            _parse(experiment)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        job = start_job(f"Metadata of {experiment}", ["download from the sheets"],
                        lambda job: preview(experiment, job))
        return jsonify(job.to_dict())

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
