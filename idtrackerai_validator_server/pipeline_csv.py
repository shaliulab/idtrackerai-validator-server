"""
pipeline_csv.py  —  rows of the idtrackerai pipeline's experiment list (idtrackerai.csv) edited from the app.

`$IDTRACKERAI_CSV` (default `$NEXTFLOW_PIPELINES/idtrackerai_v3/idtrackerai.csv`) is a plain comma-separated
file without header or quoting, one experiment per line:

    /flyhostel_data/videos/FlyHostel1/3X/2026-11-22_17-00-00,1,OK
    <basedir>,<mark>,<note>

idtrackerai.nf / segmentation.nf / splitter.nf read it with `splitCsv()` and take the rows whose second column
is exactly `1` (basedir trimmed, trailing / dropped); the basedir is the key of every join, so an experiment
listed twice with 1 fails the run, and each selected basedir must hold metadata.yaml, *.conf, *.pfs, index.db
and landmarks.toml (`checkIfExists`: one missing file stops the whole run). By hand the mark is 1 (next run),
NEXT / LATER / ? (planned), or the date the run was made (written after it); the note is mostly OK. Blank
lines and 2-column rows exist; the file is not in git (`.gitignore` of idtrackerai_v3) and nothing else
writes it (idtrackerai has no run_queue.sh).

Edits are line based: untouched lines keep their bytes. Each change is previewed (diff + `sig` of the file
read) and applied only if the file still has that sig, under an flock on the CSV (as monitor.requeue_rows and
run_queue.sh), written to a temporary file + os.replace (mode kept), after a timestamped copy in
`$IDTRACKERAI_CSV_BACKUPS` (last `KEEP_BACKUPS`), and logged to `$IDTRACKERAI_CSV_LOG` (who, when, address,
the lines before / after).

Who may write (`may_edit`): a browser on the server (loopback), or a login user listed in
`$VALIDATOR_PIPELINE_USERS` (comma separated; default the login of `VALIDATOR_USERNAME`; `*` = anyone who
reached the app, logged in or not). Reading is open to every logged-in user.
"""
import datetime
import glob
import hashlib
import ipaddress
import json
import logging
import os
import re
import shutil
import tempfile

from flask import jsonify, request, session

from idtrackerai_validator_server import lab_paths
from idtrackerai_validator_server.oscompat import IS_WINDOWS, lock, unlock

logger = logging.getLogger(__name__)

KEEP_BACKUPS = 50
PROBLEM_CHUNKS_WARN = 0.2      # share of the sampled chunks with problematic frames worth a warning
MARK_RE = re.compile(r"^[^,\r\n]*$")
SELECTED = "1"
MARKS = ["1", "NEXT", "LATER", "?"]   # offered by the client; any text without comma is accepted
REQUIRED = [("metadata.yaml", "metadata.yaml"), ("*.conf", "settings (.conf)"), ("*.pfs", "camera .pfs"),
            ("index.db", "index.db"), ("landmarks.toml", "landmarks.toml")]
LOCAL_RE = re.compile(r"^(FlyHostel\d+)/(\d+)X/(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(/arenas/(\d+))?$")


class Conflict(Exception):
    """The file changed since the preview (or the row is not what the client saw)."""


# ---- paths -------------------------------------------------------------------------------------------

def csv_path():
    return lab_paths.IDTRACKERAI_CSV


def videos_root():
    return os.environ.get("FLYHOSTEL_VIDEOS", "")


def check_experiment(experiment):
    experiment = (experiment or "").strip().strip("/")
    if not LOCAL_RE.match(experiment):
        raise ValueError(f"Not an experiment folder (FlyHostelN/NX/YYYY-MM-DD_HH-MM-SS[/arenas/K]): {experiment!r}")
    return experiment


def basedir_of(experiment, root=None):
    """The basedir written in the CSV: $FLYHOSTEL_VIDEOS/<experiment>, as the rows written by hand."""
    root = (root if root is not None else videos_root()).rstrip("/")
    return f"{root}/{experiment}"


def norm(basedir):
    """Key of a basedir as the pipeline compares them: trimmed, without trailing /."""
    return basedir.strip().rstrip("/")


# ---- reading -----------------------------------------------------------------------------------------

def read_text(path=None):
    path = path or csv_path()
    with open(path, "rb") as handle:
        data = handle.read()
    return data


def sig_of(data):
    return hashlib.sha256(data).hexdigest()[:16]


def split_lines(text):
    """Lines without their terminator, and whether the file ends with one ("\\n"; \\r kept in the line)."""
    if text == "":
        return [], True
    ends = text.endswith("\n")
    lines = text.split("\n")
    if ends:
        lines = lines[:-1]
    return lines, ends


def join_lines(lines, ends):
    return "\n".join(lines) + ("\n" if ends and lines else "")


def parse_line(line):
    fields = line.rstrip("\r").split(",")
    return {"basedir": norm(fields[0]), "mark": fields[1].strip() if len(fields) > 1 else None,
            "note": ",".join(fields[2:]).strip() if len(fields) > 2 else None}


def rows(text):
    """Every non-blank line: {line (1-based), text, basedir, mark, note}."""
    lines, _ = split_lines(text)
    out = []
    for n, line in enumerate(lines):
        if not line.strip():
            continue
        out.append({"line": n + 1, "text": line, **parse_line(line)})
    return out


def same_place(a, b):
    """Do two basedirs name the same folder? Literally, or through the FlyHostel<N> links."""
    if norm(a) == norm(b):
        return True
    try:
        return os.path.realpath(norm(a)) == os.path.realpath(norm(b))
    except OSError:
        return False


def rows_of(text, basedir):
    return [r for r in rows(text) if same_place(r["basedir"], basedir)]


# ---- checks ------------------------------------------------------------------------------------------

def _date_conf(folder):
    """The conf preprocess-sample reads: <date>.conf, or in arenas/K K.conf (copied from <date>.conf)."""
    return os.path.join(folder, os.path.basename(folder) + ".conf")


def _preprocess(folder):
    """What preprocess-sample left: finished, running, problematic frames / chunks of the sampled range."""
    log_path = os.path.join(folder, "preprocess_sample.log")
    csv_file = os.path.join(folder, "preprocess_sample.csv")
    info = {"log": os.path.exists(log_path), "csv": os.path.exists(csv_file), "finished": False,
            "running": False, "total_line": None, "problem_frames": None, "problem_chunks": None,
            "sampled_chunks": None, "mtime": None}
    try:
        from idtrackerai_validator_server import data_collector
        process = data_collector.preprocess_process(folder)
        info["running"] = process is not None
    except Exception:       # data_collector left out (compat.MISSING), psutil missing, ...
        pass
    if info["log"]:
        with open(log_path, errors="replace") as handle:
            lines = handle.read().splitlines()
        total = next((line for line in reversed(lines) if line.startswith("Total problematic frames")), None)
        info["total_line"] = total
        info["finished"] = total is not None and not info["running"]
        m = re.search(r"(\d+)", total or "")
        info["problem_frames"] = int(m.group(1)) if m else None
        ranges = [l for l in lines if re.search(r"--start-chunk|start_chunk", l)]
        info["command"] = ranges[-1][-300:] if ranges else None
    if info["csv"]:
        info["mtime"] = os.path.getmtime(csv_file)
        chunks = set()
        with open(csv_file, errors="replace") as handle:
            next(handle, None)
            for line in handle:
                parts = line.split(",")
                if len(parts) > 1 and parts[1].strip().isdigit():
                    chunks.add(int(parts[1]))
        info["problem_chunks"] = len(chunks)
        sampled = sorted(int(os.path.basename(f)[:6]) for f in glob.glob(os.path.join(folder, "[0-9]" * 6 + ".npz")))
        sampled = [c for c in sampled if c < 300]
        info["sampled_chunks"] = len(sampled) or None
    return info


def checks(experiment, mark, root=None, text=None, adding=True):
    """[{level: ok / warn / error, what, detail}] of adding / marking an experiment, plus facts for the client.
    An error blocks a row marked 1 (the run would stop); with another mark it is only a warning."""
    root = root if root is not None else videos_root()
    folder = os.path.join(root, experiment)
    out = []
    selected = (mark or "").strip() == SELECTED

    def add(level, what, detail=""):
        if level == "error" and not selected:
            level = "warn"
        out.append({"level": level, "what": what, "detail": detail})

    if not os.path.isdir(folder):
        add("error", "Folder", f"{folder} does not exist")
        return out, {}
    add("ok", "Folder", folder)
    for pattern, label in REQUIRED:
        found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder, pattern)))
        if found:
            add("ok", label, ", ".join(found))
        else:
            add("error", label, f"no {pattern} in the folder: the pipeline stops the whole run (checkIfExists)")
    conf = _date_conf(folder)
    arena = LOCAL_RE.match(experiment).group(5)
    date_time = LOCAL_RE.match(experiment).group(3)
    toml = os.path.join(folder, date_time + ".toml")
    date_conf = os.path.join(folder, date_time + ".conf")
    if os.path.exists(conf):
        add("ok", "Settings", os.path.basename(conf))
        if os.path.exists(toml) and os.path.getmtime(toml) > os.path.getmtime(date_conf if arena else conf) + 1:
            add("warn", "Settings", f"{os.path.basename(toml)} is newer than its .conf: save the settings again")
    else:
        add("error", "Settings", f"no {os.path.basename(conf)} (save the segmentation settings in the Data collector)")
    pre = _preprocess(folder)
    if pre["running"]:
        add("warn", "preprocess-sample", "still running")
    elif not pre["log"] and not pre["csv"]:
        add("warn", "preprocess-sample", "never run on this folder")
    elif not pre["finished"]:
        add("warn", "preprocess-sample", "did not finish (no 'Total problematic frames' line in its log)")
    else:
        detail = pre["total_line"]
        if pre["problem_chunks"] is not None and pre["sampled_chunks"]:
            share = pre["problem_chunks"] / pre["sampled_chunks"]
            detail += f"; {pre['problem_chunks']} of ~{pre['sampled_chunks']} sampled chunks with problems"
            add("warn" if share > PROBLEM_CHUNKS_WARN else "ok", "preprocess-sample", detail)
        else:
            add("ok", "preprocess-sample", detail)
        if pre["mtime"] and os.path.exists(conf) and os.path.getmtime(conf) > pre["mtime"] + 1:
            add("warn", "preprocess-sample", "the settings changed after it ran: run it again")
    if os.path.isdir(os.path.join(folder, "arenas")) and not arena:
        n = len(glob.glob(os.path.join(folder, "arenas", "*", "")))
        add("warn", "Arenas", f"this recording has {n} arena folder(s): they are usually added one by one (…/arenas/K)")
    dbs = [os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.db"))
           if os.path.basename(p) != "index.db" and os.path.getsize(p) > 0]
    if dbs:
        add("warn", "Database", f"already analyzed ({', '.join(dbs)}): a run analyzes the chunks not done yet")
    if text is not None:
        listed = rows_of(text, basedir_of(experiment, root))
        if listed and adding:
            add("warn", "Already listed", "line " + ", ".join(str(r["line"]) for r in listed) + ": update it instead")
        others = [r for r in rows(text) if r["mark"] == SELECTED and not same_place(r["basedir"], basedir_of(experiment, root))]
        if selected and others:
            add("ok", "Next run", f"{len(others)} other row(s) marked 1 will run with it")
    return out, {"preprocess": pre, "arena": arena}


# ---- edits -------------------------------------------------------------------------------------------

def _clean(value, what):
    value = (value or "").strip()
    if not MARK_RE.match(value):
        raise ValueError(f"{what} cannot hold a comma or a line break")
    return value


def make_line(basedir, mark, note):
    mark, note = _clean(mark, "Mark"), _clean(note, "Note")
    if not mark:
        raise ValueError("Mark is required (1 = taken by the next run)")
    return ",".join([norm(basedir), mark] + ([note] if note else []))


def apply_op(text, op, root=None):
    """New text of an operation on `text`: {"op": "add", "experiment", "mark", "note"} appends a row
    (refused when the experiment is listed), {"op": "update" | "remove", "line", "expect", ...} changes /
    deletes that line if it still reads `expect`. Returns (new text, [diff lines])."""
    lines, ends = split_lines(text)
    kind = op.get("op")
    experiment = check_experiment(op.get("experiment"))
    basedir = basedir_of(experiment, root)
    if kind == "add":
        listed = rows_of(text, basedir)
        if listed:
            raise Conflict(f"{experiment} is already listed (line {listed[0]['line']}): update that row instead")
        new = make_line(basedir, op.get("mark"), op.get("note"))
        if lines and not ends:
            ends = True             # the last line gets its terminator before the new one
        lines = lines + [new]
        return join_lines(lines, True), [{"line": len(lines), "old": None, "new": new}]
    if kind not in ("update", "remove"):
        raise ValueError(f"unknown operation {kind!r}")
    try:
        n = int(op.get("line")) - 1
    except (TypeError, ValueError):
        raise ValueError("line is required")
    if not (0 <= n < len(lines)) or lines[n] != op.get("expect"):
        raise Conflict(f"line {n + 1} changed since it was read: read the file again")
    if not same_place(parse_line(lines[n])["basedir"], basedir):
        raise ValueError(f"line {n + 1} is not a row of {experiment}")
    old = lines[n]
    if kind == "remove":
        del lines[n]
        return join_lines(lines, ends), [{"line": n + 1, "old": old, "new": None}]
    # update keeps the basedir as written and whatever follows the note's column untouched
    keep = parse_line(old)["basedir"]
    new = make_line(keep, op.get("mark"), op.get("note"))
    if old.endswith("\r"):
        new += "\r"
    lines[n] = new
    return join_lines(lines, ends), [{"line": n + 1, "old": old, "new": new}]


# ---- writing -----------------------------------------------------------------------------------------

def _backup(path, data):
    folder = lab_paths.IDTRACKERAI_CSV_BACKUPS
    os.makedirs(folder, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    target = os.path.join(folder, f"{os.path.basename(path)}.{stamp}")
    with open(target, "wb") as handle:
        handle.write(data)
    old = sorted(glob.glob(os.path.join(folder, os.path.basename(path) + ".*")))
    for extra in old[:-KEEP_BACKUPS]:
        try:
            os.remove(extra)
        except OSError:
            pass
    return target


def _log(entry):
    path = lab_paths.IDTRACKERAI_CSV_LOG
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def write(op, sig, who=None, address=None, path=None, root=None):
    """Applies `op` if the file still has `sig`: under flock, backup, temp + os.replace, audit line.
    Returns {"diff", "sig", "backup"}."""
    path = path or csv_path()
    for _ in range(5):
        # the lock is on the CSV itself (run_queue.sh, monitor.requeue_rows, `flock idtrackerai.csv`);
        # Windows locks are mandatory, so there a .lock next to it
        with open(path + ".lock" if IS_WINDOWS else path, "a") as handle:
            lock(handle)
            try:
                # replaced (by us or another writer) while we waited: lock the file now at the path
                if not IS_WINDOWS and os.fstat(handle.fileno()).st_ino != os.stat(path).st_ino:
                    continue
                data = read_text(path)
                if sig_of(data) != sig:
                    raise Conflict("idtrackerai.csv changed since the preview: check the change again")
                text = data.decode("utf-8", errors="surrogateescape")
                new_text, diff = apply_op(text, op, root=root)
                backup = _backup(path, data)
                mode = os.stat(path).st_mode & 0o7777
                fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", dir=os.path.dirname(path))
                try:
                    with os.fdopen(fd, "wb") as out:
                        out.write(new_text.encode("utf-8", errors="surrogateescape"))
                        out.flush()
                        os.fsync(out.fileno())
                    os.chmod(tmp, mode)
                    os.replace(tmp, path)
                except BaseException:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    raise
                new_sig = sig_of(new_text.encode("utf-8", errors="surrogateescape"))
                _log({"t": datetime.datetime.now().isoformat(timespec="seconds"), "user": who, "address": address,
                      "file": path, "op": op.get("op"), "experiment": op.get("experiment"), "diff": diff,
                      "sig_before": sig, "sig_after": new_sig, "backup": backup})
                logger.info("idtrackerai.csv: %s %s by %s (%s)", op.get("op"), op.get("experiment"), who, address)
                return {"diff": diff, "sig": new_sig, "backup": backup}
            finally:
                unlock(handle)
    raise Conflict("idtrackerai.csv kept being replaced: try again")


# ---- permission --------------------------------------------------------------------------------------

def _loopback(address):
    try:
        return ipaddress.ip_address((address or "").split("%")[0]).is_loopback
    except ValueError:
        return False


def current_user():
    if session.get("user"):
        return session["user"]
    auth = request.authorization
    if auth is not None and auth.type == "basic":
        return auth.username       # auth.require_login checked it
    return None


def allowed_users():
    raw = lab_paths.PIPELINE_USERS
    if raw is None:
        raw = os.environ.get("VALIDATOR_USERNAME") or ""
    return [u.strip() for u in raw.split(",") if u.strip()]


def may_edit():
    """(allowed, why): a browser on the server, or a login user of $VALIDATOR_PIPELINE_USERS (`*`: anyone)."""
    users = allowed_users()
    user = current_user()
    if _loopback(request.remote_addr):
        return True, "browser on the server"
    if "*" in users:
        return True, "VALIDATOR_PIPELINE_USERS=*"
    if user and user in users:
        return True, f"login {user}"
    if user:
        return False, f"{user} is not in VALIDATOR_PIPELINE_USERS"
    return False, "sign in (or use a browser on the server) to change idtrackerai.csv"


# ---- routes ------------------------------------------------------------------------------------------

def register_pipeline_csv(app):

    def info(experiment, mark=SELECTED):
        path = csv_path()
        exists = os.path.exists(path)
        data = read_text(path) if exists else b""
        text = data.decode("utf-8", errors="surrogateescape")
        found, facts = checks(experiment, mark, text=text)
        allowed, why = may_edit()
        listed = rows_of(text, basedir_of(experiment))
        return {
            "file": path, "exists": exists, "sig": sig_of(data), "experiment": experiment,
            "basedir": basedir_of(experiment), "rows": listed, "checks": found, **facts,
            "marks": MARKS, "default": {"mark": SELECTED, "note": "OK"},
            "selected": [r for r in rows(text) if r["mark"] == SELECTED],
            "n_rows": len(rows(text)), "can_edit": allowed, "permission": why,
        }

    @app.route("/api/collector/idtrackerai_csv", methods=["GET"])
    def pipeline_csv_get():
        """?experiment=[&mark=]: the experiment's rows in idtrackerai.csv, the checks of adding it, the file's
        sig, the rows marked 1, and whether this browser may change it."""
        try:
            experiment = check_experiment(request.args.get("experiment"))
            return jsonify(info(experiment, request.args.get("mark") or SELECTED))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except OSError as error:
            return jsonify({"error": f"cannot read {csv_path()}: {error}"}), 500

    def _op():
        body = request.get_json(silent=True) or {}
        op = {k: body.get(k) for k in ("op", "experiment", "mark", "note", "line", "expect")}
        op["experiment"] = check_experiment(op["experiment"])
        return op, body.get("sig")

    @app.route("/api/collector/idtrackerai_csv/preview", methods=["POST"])
    def pipeline_csv_preview():
        """{"op": add / update / remove, "experiment", "mark", "note", "line", "expect"}: the diff the change
        would make, the checks (with the new mark) and the sig to send back with /apply. Nothing written."""
        try:
            op, _ = _op()
            data = read_text() if os.path.exists(csv_path()) else b""
            text = data.decode("utf-8", errors="surrogateescape")
            _, diff = apply_op(text, op)
            found, _ = checks(op["experiment"], op.get("mark") if op["op"] != "remove" else "", text=text,
                              adding=op["op"] == "add")
            blocking = [c for c in found if c["level"] == "error"] if op["op"] != "remove" else []
            allowed, why = may_edit()
            return jsonify({"diff": diff, "sig": sig_of(data), "checks": found, "blocking": len(blocking),
                            "can_edit": allowed, "permission": why})
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except Conflict as error:
            return jsonify({"error": str(error), "conflict": True}), 409

    @app.route("/api/collector/idtrackerai_csv/apply", methods=["POST"])
    def pipeline_csv_apply():
        """The same body plus the preview's `sig`: written if the file did not change since (409 otherwise),
        refused (400) when a row marked 1 fails a check the pipeline would stop on, 403 without permission."""
        allowed, why = may_edit()
        if not allowed:
            return jsonify({"error": f"Not allowed to change idtrackerai.csv: {why}"}), 403
        try:
            op, sig = _op()
            if not os.path.exists(csv_path()):
                return jsonify({"error": f"{csv_path()} does not exist"}), 400
            if op["op"] != "remove":
                found, _ = checks(op["experiment"], op.get("mark"))
                errors = [c for c in found if c["level"] == "error"]
                if errors:
                    return jsonify({"error": "; ".join(f"{c['what']}: {c['detail']}" for c in errors),
                                    "checks": found}), 400
            result = write(op, sig, who=current_user() or why, address=request.remote_addr)
            return jsonify({**result, **info(op["experiment"])})
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except Conflict as error:
            return jsonify({"error": str(error), "conflict": True}), 409
