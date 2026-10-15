"""Routes that help write `<basedir>/flyhostel/validation/validation.csv`.

The CSV is read by flyhostel (`flyhostel/data/human_validation/cvat/validation_csv.py`)
when the validated tables are rebuilt. Its `fragment` is the fragment of the raw
ROI_0 table (fragment ids restart in every chunk) and `local_identity` is the
local identity within the chunk of the row, so every row must stay in one chunk.

    GET  /api/validation/frame/<n>   raw and validated blobs of frame n, the extent
                                     of each raw fragment in its chunk and the
                                     identity -> local_identity map of the chunk
    GET  /api/validation/lid_map     ?first=<fn>&last=<fn>: that map for every chunk
                                     between two frames
    GET  /api/validation/csv         current validation.csv of the loaded experiment
    POST /api/validation/csv         {"experiment", "entries": [{"comment", "rows"}]}:
                                     appends the rows of each entry under its comment
                                     (a copy of the previous file is kept next to it)
"""
import logging
import math
import os
import shutil
import sqlite3
import time
from threading import Lock

from flask import jsonify, request

from idtrackerai_validator_server.backend import generate_database_filename

logger = logging.getLogger(__name__)

# Column order of new files. Existing files keep theirs.
CSV_COLUMNS = [
    "frame_number", "first_frame_number", "last_frame_number",
    "x", "y", "fragment", "local_identity", "by_identity", "replace",
]
BACKUP_DIR = "validation_csv_backups"

_write_lock = Lock()


def validation_csv_path(experiment):
    return os.path.join(
        os.environ["FLYHOSTEL_VIDEOS"], experiment, "flyhostel", "validation", "validation.csv"
    )


def _connect(experiment):
    dbfile = generate_database_filename(experiment)
    return sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True)


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _chunksize(conn):
    row = conn.execute("SELECT value FROM METADATA WHERE field = 'chunksize'").fetchone()
    return int(float(row[0]))


def _blobs(conn, frame_number, suffix):
    rows = conn.execute(
        f"""SELECT r.in_frame_index, r.x, r.y, r.fragment, i.local_identity, i.identity
            FROM ROI_0{suffix} r LEFT JOIN IDENTITY{suffix} i
              ON i.frame_number = r.frame_number AND i.in_frame_index = r.in_frame_index
            WHERE r.frame_number = ? ORDER BY r.in_frame_index""",
        (frame_number,),
    ).fetchall()
    keys = ["in_frame_index", "x", "y", "fragment", "local_identity", "identity"]
    out = []
    for row in rows:
        blob = dict(zip(keys, row))
        blob["x"] = None if blob["x"] is None else round(blob["x"])
        blob["y"] = None if blob["y"] is None else round(blob["y"])
        if blob["fragment"] is not None:
            blob["fragment"] = int(blob["fragment"])
        out.append(blob)
    return out


def _lid_maps(conn, first_chunk, last_chunk):
    """{chunk: {identity: local_identity}} from the validated concatenation, which
    is what the last validation produced, else the raw one."""
    table = "CONCATENATION_VAL" if "CONCATENATION_VAL" in _tables(conn) else "CONCATENATION"
    maps = {}
    for chunk, identity, local_identity in conn.execute(
        f"SELECT chunk, identity, local_identity FROM {table} WHERE chunk BETWEEN ? AND ?",
        (first_chunk, last_chunk),
    ):
        if identity is None or local_identity is None:
            continue
        maps.setdefault(int(chunk), {})[int(identity)] = int(local_identity)
    return maps, table


def _format(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _check_rows(rows, chunksize):
    """Normalised copies of the posted rows; ValueError with the reason otherwise."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("no rows to write")
    out = []
    for i, row in enumerate(rows, 1):
        try:
            frame_number = int(row["frame_number"])
            local_identity = int(row["local_identity"])
            replace = bool(row["replace"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"row {i}: frame_number, local_identity and replace are required")
        first, last = row.get("first_frame_number"), row.get("last_frame_number")
        if (first is None) != (last is None):
            raise ValueError(f"row {i}: give both first_frame_number and last_frame_number, or neither")
        if first is not None:
            first, last = int(first), int(last)
            if not first <= frame_number <= last:
                raise ValueError(f"row {i}: frame_number must lie in [first_frame_number, last_frame_number]")
            if first // chunksize != last // chunksize:
                raise ValueError(f"row {i}: frames {first}-{last} span more than one chunk")
        fragment = row.get("fragment")
        x, y = row.get("x"), row.get("y")
        if fragment is None and (x is None or y is None):
            raise ValueError(f"row {i}: needs a fragment or a position (x, y)")
        out.append({
            "frame_number": frame_number,
            "first_frame_number": first,
            "last_frame_number": last,
            "x": None if x is None else int(round(float(x))),
            "y": None if y is None else int(round(float(y))),
            "fragment": None if fragment is None else int(fragment),
            "local_identity": local_identity,
            "by_identity": bool(row.get("by_identity", False)),
            "replace": replace,
        })
    return out


def _append(path, entries):
    """Append each entry's rows (dicts keyed by CSV_COLUMNS) under its `# comment` lines.

    The file keeps its column order. Columns it lacks are added with NaN for the
    rows already there, which flyhostel reads as the default it used before
    (single frame for first/last_frame_number, by_identity=True).
    """
    lines = []
    if os.path.exists(path):
        with open(path) as f:
            lines = f.read().splitlines()

    header_at = next(
        (i for i, line in enumerate(lines) if line.strip() and not line.lstrip().startswith("#")),
        None,
    )
    if header_at is None:
        header = list(CSV_COLUMNS)
        lines.append(",".join(header))
    else:
        header = [c.strip() for c in lines[header_at].split(",")]
        missing = [c for c in CSV_COLUMNS if c not in header]
        if missing:
            header += missing
            lines[header_at] = ",".join(header)
            pad = ",NaN" * len(missing)
            for i in range(header_at + 1, len(lines)):
                if lines[i].strip() and not lines[i].lstrip().startswith("#"):
                    lines[i] += pad

    for comment, rows in entries:
        lines.append("")
        for text in (comment or "").splitlines():
            if text.strip():
                lines.append(f"# {text.strip()}")
        for row in rows:
            lines.append(",".join(_format(row.get(c)) for c in header))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def register_validation_csv(app, get_selected_experiment):

    def _experiment():
        experiment = get_selected_experiment()
        if not experiment:
            raise LookupError("no experiment loaded")
        return experiment

    @app.route("/api/validation/frame/<int:frame_number>", methods=["GET"])
    def validation_frame(frame_number):
        try:
            experiment = _experiment()
        except LookupError as error:
            return jsonify({"error": str(error)}), 503
        with _connect(experiment) as conn:
            chunksize = _chunksize(conn)
            chunk = frame_number // chunksize
            first, last = chunk * chunksize, (chunk + 1) * chunksize - 1
            raw = _blobs(conn, frame_number, "")
            val = _blobs(conn, frame_number, "_VAL") if "ROI_0_VAL" in _tables(conn) else None

            fragments = sorted({b["fragment"] for b in raw if b["fragment"] is not None})
            extents = {}
            if fragments:
                marks = ",".join("?" * len(fragments))
                for fragment, f0, f1 in conn.execute(
                    f"""SELECT fragment, MIN(frame_number), MAX(frame_number) FROM ROI_0
                        WHERE frame_number BETWEEN ? AND ? AND fragment IN ({marks})
                        GROUP BY fragment""",
                    (first, last, *fragments),
                ):
                    extents[int(fragment)] = [int(f0), int(f1)]
            for blob in raw:
                blob["fragment_frames"] = extents.get(blob["fragment"])

            maps, source = _lid_maps(conn, chunk, chunk)

        return jsonify({
            "experiment": experiment,
            "frame_number": frame_number,
            "chunk": chunk,
            "chunksize": chunksize,
            "chunk_frames": [first, last],
            "raw": raw,
            "val": val,
            "lid_by_identity": maps.get(chunk, {}),
            "lid_source": source,
        })

    @app.route("/api/validation/lid_map", methods=["GET"])
    def validation_lid_map():
        try:
            experiment = _experiment()
            first = int(request.args["first"])
            last = int(request.args["last"])
        except LookupError as error:
            return jsonify({"error": str(error)}), 503
        except (KeyError, ValueError):
            return jsonify({"error": "first and last frame numbers required"}), 400
        with _connect(experiment) as conn:
            chunksize = _chunksize(conn)
            maps, source = _lid_maps(conn, first // chunksize, last // chunksize)
        return jsonify({"chunksize": chunksize, "maps": maps, "lid_source": source})

    @app.route("/api/validation/csv", methods=["GET"])
    def validation_csv_get():
        try:
            experiment = _experiment()
        except LookupError as error:
            return jsonify({"error": str(error)}), 503
        path = validation_csv_path(experiment)
        text = None
        if os.path.exists(path):
            with open(path) as f:
                text = f.read()
        return jsonify({"experiment": experiment, "path": path, "exists": text is not None, "text": text})

    @app.route("/api/validation/csv", methods=["POST"])
    def validation_csv_append():
        try:
            experiment = _experiment()
        except LookupError as error:
            return jsonify({"error": str(error)}), 503
        data = request.get_json(silent=True) or {}
        if data.get("experiment") != experiment:
            return jsonify({"error": f"the loaded experiment is {experiment}, not {data.get('experiment')}"}), 409
        try:
            with _connect(experiment) as conn:
                chunksize = _chunksize(conn)
            entries = data.get("entries")
            if not isinstance(entries, list) or not entries:
                raise ValueError("no entries to write")
            entries = [(e.get("comment"), _check_rows(e.get("rows"), chunksize)) for e in entries]
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        n_rows = sum(len(rows) for _, rows in entries)

        path = validation_csv_path(experiment)
        backup = None
        with _write_lock:
            if os.path.exists(path):
                backup_dir = os.path.join(os.path.dirname(path), BACKUP_DIR)
                os.makedirs(backup_dir, exist_ok=True)
                backup = os.path.join(backup_dir, f"validation_{time.strftime('%Y%m%d-%H%M%S')}.csv")
                shutil.copy2(path, backup)
            _append(path, entries)
        logger.info("Appended %d rows to %s (backup %s)", n_rows, path, backup)
        return jsonify({"path": path, "backup": backup, "rows": n_rows})
