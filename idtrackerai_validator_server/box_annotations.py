"""Box annotations: validation of multi-fly experiments in the app (the "Annotate" tab), CVAT kept as fallback.

A person checks the frames where tracking could be wrong and leaves, on each, one box per fly
labelled with its local identity (1..N, the labels of the CVAT tasks), or a tag (FMB,
NEW-FRAGMENT, CROSSING, the CVAT tag vocabulary of flyhostel's cross_machine_human). Boxes are
kept per experiment in a sidecar database, <basedir>/flyhostel/validation/validator_annotations.db:

    boxes   (frame_number, x0, y0, x1, y1, label)     label "" = not assigned (ignored, as UNASSIGNED in CVAT)
    frames  (frame_number, author, updated, n_boxes)   frames saved by a person: the annotated frames
    log     (t, author, frame_number, boxes JSON)      every save, for audit

Coordinates are those of the original frame (no CVAT resize). flyhostel reads the database with
`integrate-human-annotations --source validator` (flyhostel/data/human_validation/cvat/validator_annotations.py),
which turns each box into the rectangle contour CVAT exports, so the integration is the same.

A frame not saved yet is proposed, when `yolov7-cvat propose` ran on the experiment ("Propose" button: the
detections and identity solver of the CVAT pipeline's `predict --assign` + `complete`, written to the tables
proposal_frames / proposals / proposal_runs of the same database), from those proposals: boxes with the solver's
identity, a note and its confidence, and the frame's flags (what a person has to check, the CVAT issues). Otherwise
from the machine tables (ROI_0 + IDENTITY, never _VAL: the integration crosses the boxes with the machine data):
one box per row, the bounding box of the segmented blob holding its centroid (at least MIN_BOX px) labelled with its
local identity (0 → not assigned).

Frames to check: the failed scenes of the validation pipeline's scene_qc.csv (make-identogram +
auto-annotate-qc, scenes ≤ MAX_SCENE_LENGTH frames as export-images), else the runs of bad frames of
the QC tab's cached result.

Routes
    GET    /api/annotate/scenes              scenes to check, annotated frames, labels, the database, the frames
                                             still with problems after the last integration (validator_problem_frames.csv)
    GET    /api/annotate/frame/<n>           saved boxes or the proposal (?proposal=1: always; =tracking: idtrackerai's,
                                             not the solver's), machine rows, local id → identity, flags
    POST   /api/annotate/frame/<n>           {"boxes": [{x0, y0, x1, y1, label}]} saves the frame
    DELETE /api/annotate/frame/<n>           the frame is no longer annotated
    POST   /api/annotate/propose             job running `yolov7-cvat propose` (YOLO + identity solver) on the scenes
    POST   /api/annotate/integrate           {"mode": "integrate" | "save", "start_zt", "end_zt"}: job running
                                             flyhostel's integrate-human-annotations / save-human-annotations
"""
import csv
import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time

import cv2
import numpy as np
from flask import jsonify, request, session

from idtrackerai_validator_server import qc
from idtrackerai_validator_server.experiments import to_flat
from idtrackerai_validator_server.jobs import _jobs, start_job

logger = logging.getLogger(__name__)

DB_NAME = "validator_annotations.db"
TAGS = ["FMB", "NEW-FRAGMENT", "CROSSING"]
MIN_BOX = 50               # px: a proposal covers at least the square flyhostel uses for YOLO rows
PAD = 4                    # px around the segmented blob
MAX_SCENE_LENGTH = 5000    # as flyhostel's export-images
VALIDATION_ENV = os.path.expanduser(os.environ.get("FLYHOSTEL_VALIDATION_ENV", "~/mambaforge/envs/rapids-23.04"))
YOLOV7_CVAT = os.path.expanduser(os.environ.get("YOLOV7_CVAT", "~/mambaforge/envs/yolov7/bin/yolov7-cvat"))

_lock = threading.Lock()
_integrating = {}          # experiment -> job id
_proposing = {}            # experiment -> job id


def _validation_folder(basedir):
    return os.path.join(basedir, "flyhostel", "validation")


def _connect(basedir, create=False):
    path = os.path.join(_validation_folder(basedir), DB_NAME)
    if not create and not os.path.exists(path):
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS boxes (frame_number INTEGER, x0 REAL, y0 REAL, x1 REAL, y1 REAL, label TEXT);
        CREATE INDEX IF NOT EXISTS boxes_fn ON boxes (frame_number);
        CREATE TABLE IF NOT EXISTS frames (frame_number INTEGER PRIMARY KEY, author TEXT, updated REAL, n_boxes INTEGER);
        CREATE TABLE IF NOT EXISTS log (t REAL, author TEXT, frame_number INTEGER, boxes TEXT);
        CREATE TABLE IF NOT EXISTS proposal_frames (frame_number INTEGER PRIMARY KEY, flags TEXT);
        CREATE TABLE IF NOT EXISTS proposals (frame_number INTEGER, x0 REAL, y0 REAL, x1 REAL, y1 REAL, label TEXT,
                                              note TEXT, confidence TEXT);
        CREATE INDEX IF NOT EXISTS proposals_fn ON proposals (frame_number);
        CREATE TABLE IF NOT EXISTS proposal_runs (t REAL, model TEXT, scenes INTEGER, frames INTEGER, summary TEXT);
    """)
    return con


def _machine_rows(dbfile, frame_number):
    con = sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT r.in_frame_index, r.x, r.y, r.fragment, r.modified, i.local_identity, i.identity "
            "FROM ROI_0 r LEFT JOIN IDENTITY i ON i.frame_number = r.frame_number AND i.in_frame_index = r.in_frame_index "
            "WHERE r.frame_number = ? ORDER BY r.in_frame_index", (frame_number,)).fetchall()
        return [{"in_frame_index": a, "x": x, "y": y, "fragment": f, "modified": m or 0,
                 "local_identity": lid, "identity": ident} for a, x, y, f, m, lid, ident in rows]
    finally:
        con.close()


def _concatenation(dbfile, chunk):
    """local identity -> identity in that chunk"""
    con = sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT local_identity, identity FROM CONCATENATION WHERE chunk = ?", (chunk,)).fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        con.close()
    return {int(lid): int(ident) for lid, ident in rows if lid is not None and ident is not None}


def propose(rows, contours, width, height):
    """One box per machine row: the bounding box of the blob holding its centroid (else the
    nearest within 20 px), at least MIN_BOX px wide and high, labelled with the local identity."""
    polys = [np.asarray(c, dtype=np.float32).reshape((-1, 1, 2)) for c in contours if len(c) >= 3]
    boxes = []
    for row in rows:
        x, y = float(row["x"]), float(row["y"])
        best, best_d = None, -20.0
        for poly in polys:
            d = cv2.pointPolygonTest(poly, (x, y), True)
            if d > best_d:
                best, best_d = poly, d
        if best is not None:
            bx, by, bw, bh = cv2.boundingRect(best.astype(np.int32))
            x0, y0, x1, y1 = bx - PAD, by - PAD, bx + bw + PAD, by + bh + PAD
        else:
            x0, y0, x1, y1 = x, y, x, y
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_w, half_h = max(x1 - x0, MIN_BOX) / 2, max(y1 - y0, MIN_BOX) / 2
        lid = row["local_identity"]
        boxes.append({
            "x0": max(0.0, cx - half_w), "y0": max(0.0, cy - half_h),
            "x1": min(float(width), cx + half_w), "y1": min(float(height), cy + half_h),
            "label": str(int(lid)) if lid else "",
        })
    return boxes


def check_boxes(boxes, n_animals):
    """Boxes as stored ({x0, y0, x1, y1, label}), or ValueError."""
    labels = {str(i) for i in range(1, n_animals + 1)} | set(TAGS) | {""}
    out, seen = [], set()
    for box in boxes:
        try:
            x0, y0, x1, y1 = (float(box[k]) for k in ("x0", "y0", "x1", "y1"))
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"bad box {box!r}")
        label = str(box.get("label") or "")
        if label not in labels:
            raise ValueError(f"unknown label {label!r} (local identities 1..{n_animals}, {', '.join(TAGS)})")
        if label.isdigit():
            if label in seen:
                raise ValueError(f"local identity {label} is on two boxes")
            seen.add(label)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise ValueError("a box is too small")
        out.append({"x0": round(x0, 1), "y0": round(y0, 1), "x1": round(x1, 1), "y1": round(y1, 1), "label": label})
    return out


def read_scenes(basedir, experiment, n_animals):
    """[{start, length, reason}] of the frames to check, and where they come from."""
    path = os.path.join(_validation_folder(basedir), "scene_qc.csv")
    if os.path.exists(path):
        scenes = []
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("pass") == "True":
                    continue
                length = int(float(row["length"]))
                if length > MAX_SCENE_LENGTH:
                    continue
                reasons = []
                if row.get("all_valid_ids") == "False":
                    reasons.append("ids missing")
                if row.get("maintains_id") in ("False", "0"):
                    reasons.append("id changes")
                if float(row.get("broken") or 0) > 0:
                    reasons.append("broken")
                if float(row.get("gap_n_frames") or 0) > 0:
                    reasons.append("gap")
                if float(row.get("between_chunks") or 0) > 0:
                    reasons.append("between chunks")
                if float(row.get("n_failed_fragments") or 0) > 0:
                    reasons.append(f"{int(float(row['n_failed_fragments']))} failed fragments")
                scenes.append({"start": int(float(row["scene_start"])), "length": length, "reason": ", ".join(reasons)})
        scenes.sort(key=lambda s: s["start"])
        return scenes, {"kind": "scene_qc", "path": path}
    # the QC tab's runs of bad frames, if it ran on the machine tables
    dbfiles = [f for f in os.listdir(basedir) if f.endswith(".db") and f != "index.db"]
    if dbfiles:
        dbfile = os.path.join(basedir, dbfiles[0])
        cached = qc._read_cache(qc._cache_path(experiment, "IDENTITY", n_animals), os.path.getmtime(dbfile))
        if cached:
            runs = cached.get("result", cached).get("runs", [])
            scenes = [{"start": int(r[0]), "length": int(r[2]), "reason": f"min {r[3]} flies identified"} for r in runs]
            scenes.sort(key=lambda s: s["start"])
            return scenes, {"kind": "qc", "path": None}
    return [], {"kind": None, "path": None}


def read_problems(basedir):
    """Frames annotated here that still had a problem after the last integration
    (flyhostel's validator_problem_frames.csv): {mtime, frames: [{frame_number, problems}]} or None."""
    path = os.path.join(_validation_folder(basedir), "validator_problem_frames.csv")
    if not os.path.exists(path):
        return None
    frames = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            frames.setdefault(int(row["frame_number"]), []).append(row["problem"])
    return {"mtime": os.path.getmtime(path),
            "frames": [{"frame_number": fn, "problems": p} for fn, p in sorted(frames.items())]}


def _author():
    return session.get("user") or os.environ.get("USER") or "anonymous"


def register_box_annotations(app, get_state):
    """get_state() -> None (no database) or a dict with experiment, basedir, dbfile, chunksize,
    n_animals, contours(frame_number) -> list, size() -> (width, height)."""

    def state_or_error():
        state = get_state()
        if state is None:
            return None, (jsonify({"error": "load an experiment with a tracking database"}), 400)
        return state, None

    @app.route("/api/annotate/scenes", methods=["GET"])
    def annotate_scenes():
        state, error = state_or_error()
        if error:
            return error
        scenes, source = read_scenes(state["basedir"], state["experiment"], state["n_animals"])
        annotated, flagged, proposed, run = [], [], 0, None
        con = _connect(state["basedir"])
        if con is not None:
            with con:
                annotated = [{"frame_number": fn, "author": a, "updated": u, "n_boxes": n}
                             for fn, a, u, n in con.execute("SELECT frame_number, author, updated, n_boxes FROM frames ORDER BY frame_number")]
                flagged = [fn for (fn,) in con.execute(
                    "SELECT frame_number FROM proposal_frames WHERE flags != '[]' ORDER BY frame_number")]
                proposed = con.execute("SELECT COUNT(*) FROM proposal_frames").fetchone()[0]
                last = con.execute("SELECT t, model, scenes, frames, summary FROM proposal_runs ORDER BY t DESC LIMIT 1").fetchone()
                if last:
                    run = {"t": last[0], "model": last[1], "scenes": last[2], "frames": last[3], "summary": json.loads(last[4] or "{}")}
            con.close()
        return jsonify({
            "experiment": state["experiment"], "n_animals": state["n_animals"], "chunksize": state["chunksize"],
            "labels": [str(i) for i in range(1, state["n_animals"] + 1)], "tags": TAGS,
            "scenes": scenes, "source": source, "annotated": annotated,
            "database": os.path.join(_validation_folder(state["basedir"]), DB_NAME),
            "integrating": _integrating.get(state["experiment"]),
            "problems": read_problems(state["basedir"]),
            "flagged": flagged, "proposed_frames": proposed, "proposal_run": run,
            "proposing": _proposing.get(state["experiment"]),
        })

    @app.route("/api/annotate/frame/<int:frame_number>", methods=["GET"])
    def annotate_frame(frame_number):
        state, error = state_or_error()
        if error:
            return error
        chunk = frame_number // state["chunksize"]
        rows = _machine_rows(state["dbfile"], frame_number)
        saved, meta, solver, flags = None, None, None, []
        want = request.args.get("proposal")
        con = _connect(state["basedir"])
        if con is not None:
            with con:
                meta = con.execute("SELECT author, updated FROM frames WHERE frame_number = ?", (frame_number,)).fetchone()
                if meta:
                    saved = [{"x0": a, "y0": b, "x1": c, "y1": d, "label": lab} for a, b, c, d, lab in con.execute(
                        "SELECT x0, y0, x1, y1, label FROM boxes WHERE frame_number = ? ORDER BY rowid", (frame_number,))]
                pf = con.execute("SELECT flags FROM proposal_frames WHERE frame_number = ?", (frame_number,)).fetchone()
                if pf:
                    flags = json.loads(pf[0] or "[]")
                    solver = [{"x0": a, "y0": b, "x1": c, "y1": d, "label": lab or "", "note": note or "", "confidence": conf}
                              for a, b, c, d, lab, note, conf in con.execute(
                                  "SELECT x0, y0, x1, y1, label, note, confidence FROM proposals WHERE frame_number = ? ORDER BY rowid",
                                  (frame_number,))]
            con.close()
        proposal, source = None, None
        if solver is not None and (saved is None or want) and want != "tracking":
            proposal, source = solver, "solver"
        elif saved is None or want:
            source = "tracking"
            width, height = state["size"]()
            try:
                contours = state["contours"](frame_number)
            except Exception as exc:
                logger.warning("No contours for frame %s: %s", frame_number, exc)
                contours = []
            proposal = propose(rows, contours, width, height)
        return jsonify({
            "frame_number": frame_number, "chunk": chunk, "source": source if proposal is not None else "saved",
            "flags": flags, "has_solver": solver is not None,
            "saved": saved is not None, "boxes": proposal if proposal is not None else saved,
            "author": meta[0] if meta else None, "updated": meta[1] if meta else None,
            "machine": rows, "concatenation": _concatenation(state["dbfile"], chunk),
        })

    @app.route("/api/annotate/frame/<int:frame_number>", methods=["POST"])
    def annotate_save(frame_number):
        state, error = state_or_error()
        if error:
            return error
        body = request.get_json(silent=True) or {}
        if body.get("experiment") and to_flat(body["experiment"]) != to_flat(state["experiment"]):
            return jsonify({"error": "another experiment is loaded"}), 409
        try:
            boxes = check_boxes(body.get("boxes") or [], state["n_animals"])
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        author, now = _author(), time.time()
        with _lock:
            con = _connect(state["basedir"], create=True)
            with con:
                con.execute("DELETE FROM boxes WHERE frame_number = ?", (frame_number,))
                con.executemany("INSERT INTO boxes VALUES (?, ?, ?, ?, ?, ?)",
                                [(frame_number, b["x0"], b["y0"], b["x1"], b["y1"], b["label"]) for b in boxes])
                con.execute("INSERT OR REPLACE INTO frames VALUES (?, ?, ?, ?)", (frame_number, author, now, len(boxes)))
                con.execute("INSERT INTO log VALUES (?, ?, ?, ?)", (now, author, frame_number, json.dumps(boxes)))
            con.close()
        return jsonify({"frame_number": frame_number, "saved": True, "boxes": boxes, "author": author, "updated": now})

    @app.route("/api/annotate/frame/<int:frame_number>", methods=["DELETE"])
    def annotate_delete(frame_number):
        state, error = state_or_error()
        if error:
            return error
        author, now = _author(), time.time()
        with _lock:
            con = _connect(state["basedir"])
            if con is not None:
                with con:
                    con.execute("DELETE FROM boxes WHERE frame_number = ?", (frame_number,))
                    con.execute("DELETE FROM frames WHERE frame_number = ?", (frame_number,))
                    con.execute("INSERT INTO log VALUES (?, ?, ?, ?)", (now, author, frame_number, "null"))
                con.close()
        return jsonify({"frame_number": frame_number, "saved": False})

    @app.route("/api/annotate/propose", methods=["POST"])
    def annotate_propose():
        state, error = state_or_error()
        if error:
            return error
        experiment = state["experiment"]
        running = _jobs.get(str(_proposing.get(experiment)))
        if running is not None and running.status == "running":
            return jsonify({"error": "proposals of this experiment are being made", "job": running.to_dict()}), 409
        if not os.path.exists(YOLOV7_CVAT):
            return jsonify({"error": f"{YOLOV7_CVAT} not found (YOLOV7_CVAT)"}), 500
        if not os.path.exists(os.path.join(_validation_folder(state["basedir"]), "scene_qc.csv")):
            return jsonify({"error": "no flyhostel/validation/scene_qc.csv: run make-identogram and auto-annotate-qc first"}), 400
        command = [YOLOV7_CVAT, "propose", "--experiment", to_flat(experiment),
                   "--videos", os.environ["FLYHOSTEL_VIDEOS"],
                   "--db", os.path.join(_validation_folder(state["basedir"]), DB_NAME)]

        def work(job):
            job.update(0, status="running")
            lines = []
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                if line.startswith("@@PROGRESS"):
                    job.update(0, progress=min(0.99, float(line.split()[1])))
                    continue
                lines.append(line.rstrip("\n"))
                del lines[:-200]
            code = process.wait()
            summary = next((json.loads(l) for l in reversed(lines) if l.startswith("{")), None)
            job.result = {"command": " ".join(command), "exit_code": code, "tail": lines[-40:], "summary": summary}
            if code != 0:
                job.update(0, status="failed", error=f"exit code {code}")
                raise RuntimeError(f"yolov7-cvat propose ended with exit code {code}")
            job.update(0, status="done", progress=1.0)

        job = start_job(f"YOLO + identity proposals of {to_flat(experiment)}", [to_flat(experiment)], work)
        _proposing[experiment] = job.id
        return jsonify(job.to_dict()), 202

    @app.route("/api/annotate/integrate", methods=["POST"])
    def annotate_integrate():
        state, error = state_or_error()
        if error:
            return error
        body = request.get_json(silent=True) or {}
        mode = body.get("mode", "integrate")
        if mode not in ("integrate", "save"):
            return jsonify({"error": "mode is integrate or save"}), 400
        experiment = state["experiment"]
        running = _jobs.get(str(_integrating.get(experiment)))
        if running is not None and running.status == "running":
            return jsonify({"error": "an integration of this experiment is running", "job": running.to_dict()}), 409
        executable = os.path.join(VALIDATION_ENV, "bin", "integrate-human-annotations" if mode == "integrate" else "save-human-annotations")
        if not os.path.exists(executable):
            return jsonify({"error": f"{executable} not found (FLYHOSTEL_VALIDATION_ENV)"}), 500
        folder = _validation_folder(state["basedir"])
        command = [executable, "--experiment", to_flat(experiment), "--folder", folder]
        if mode == "integrate":
            command += ["--source", "validator"]
        range_args = []
        for key, flag in (("start_zt", "--start-zt"), ("end_zt", "--end-zt")):
            if body.get(key) not in (None, ""):
                range_args += [flag, str(float(body[key]))]
        command += range_args or ["--frames-from-annotation"]
        if mode == "save" and not os.path.exists(os.path.join(folder, "IDENTITY_VAL.feather")):
            return jsonify({"error": "integrate first: no IDENTITY_VAL.feather in the validation folder"}), 400

        def work(job):
            job.update(0, status="running")
            env = dict(os.environ, PATH=os.path.join(VALIDATION_ENV, "bin") + os.pathsep + os.environ.get("PATH", ""))
            lines = []
            process = subprocess.Popen(command, cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                lines.append(line.rstrip("\n"))
                del lines[:-200]
                match = re.search(r"(\d+)%\|", line)
                if match:
                    job.update(0, progress=min(0.99, int(match.group(1)) / 100))
            code = process.wait()
            job.result = {"mode": mode, "command": " ".join(command), "exit_code": code, "tail": lines[-60:]}
            if code != 0:
                job.update(0, status="failed", error=f"exit code {code}")
                raise RuntimeError(f"{os.path.basename(executable)} ended with exit code {code}")
            job.update(0, status="done", progress=1.0)

        title = ("Integrate the boxes of " if mode == "integrate" else "Write the validated tables of ") + to_flat(experiment)
        job = start_job(title, [to_flat(experiment)], work)
        _integrating[experiment] = job.id
        return jsonify(job.to_dict()), 202
