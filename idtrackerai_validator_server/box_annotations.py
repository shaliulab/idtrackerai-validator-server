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
auto-annotate-qc, as vsc-scripts' manual/validation/main.sh, made by Prepare scenes; scenes ≤ MAX_SCENE_LENGTH frames
as export-images; the QC tab's runs of bad frames are not used). Plus the unsure chunk boundaries (`unsure_boundaries`): the integration links each
fly of a chunk to the next by its nearest centroid (flyhostel's make_identity_table), which can go wrong when a fly
is missing on either side, jumps, or another fly is almost as close; those boundaries are added as scenes of
2 × BOUNDARY_CONTEXT frames (or their reason appended to a scene that already covers them).

Routes
    GET    /api/annotate/scenes              scenes to check, annotated frames, labels, the database, the frames
                                             still with problems after the last integration (validator_problem_frames.csv)
    GET    /api/annotate/frame/<n>           saved boxes or the proposal (?proposal=1: always; =tracking: idtrackerai's,
                                             not the solver's), machine rows, local id → identity, flags
    POST   /api/annotate/frame/<n>           {"boxes": [{x0, y0, x1, y1, label}]} saves the frame
    DELETE /api/annotate/frame/<n>           the frame is no longer annotated
    POST   /api/annotate/propose             job running `yolov7-cvat propose` (YOLO + identity solver) on the scenes
    POST   /api/annotate/integrate           {"mode": "integrate" | "save", "start_zt", "end_zt", "range"}: job running
                                             flyhostel's integrate-human-annotations / save-human-annotations. Without
                                             ZT, integrate covers every tracked chunk (--fn-interval), or the annotated
                                             frames with "range": "annotated"; save writes what was integrated
    POST   /api/annotate/integrate/stop      kills the running integration / write of the loaded experiment
"""
import csv
import glob
import json
import logging
import os
import re
import sqlite3
import subprocess
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from flask import jsonify, request, session

from idtrackerai_validator_server import links
from idtrackerai_validator_server.compat import get_number_of_animals
from idtrackerai_validator_server.experiments import is_validated, to_flat
from idtrackerai_validator_server.jobs import _jobs, start_job
from idtrackerai_validator_server.exchange_bundle import bundle_info, read_bundle_scenes

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
_preparing = {}            # experiment -> job id
_integration_procs = {}    # experiment -> Popen of the running integrate / save
_stopped = set()           # job ids stopped by a person

# Chunk boundaries the integration may link wrongly (flyhostel's make_identity_table: each fly of the last frame of
# chunk c goes, in local identity order, to the nearest free fly of the first frame of c + 1). Measured on
# FlyHostel1_3X_2026-08-28: the chosen fly is ≤ 13 px away, the next candidate ≥ 76 px further.
BOUNDARY_JUMP_PX = 30      # chosen fly further than this: the fly moved a lot or the link is wrong
BOUNDARY_MARGIN_PX = 50    # second candidate less than this further than the chosen one: ambiguous
BOUNDARY_CONTEXT = 3       # frames shown on each side of the boundary
_boundary_cache = {}       # dbfile -> (mtime, n_animals, [(frame, reason)])
_boundary_running = set()  # dbfiles being checked in the background
PREPARE_VERSION = "v4.0"  # validation_version.txt, as vsc-scripts/manual/validation/main.sh
TIME_WINDOW_LENGTH = 0.002   # make-identogram, as main.sh
PREPARE_JOBS = int(os.environ.get("PREPARE_N_JOBS", "10"))   # auto-annotate-qc cores


def _validation_folder(basedir):
    return os.path.join(basedir, "flyhostel", "validation")


def sidecar_path(basedir):
    return os.path.join(_validation_folder(basedir), DB_NAME)


def _connect(basedir, create=False):
    path = sidecar_path(basedir)
    if not create and not os.path.exists(path):
        return None
    return connect_path(path)


def connect_path(path):
    """An annotations database (the sidecar or a copy of it, gdrive_exchange.py), made / upgraded to the schema"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
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
    # track: id of the track a box belongs to (drawn with "New track": carried to the next frame until hidden), or NULL
    if "track" not in [r[1] for r in con.execute("PRAGMA table_info(boxes)")]:
        con.execute("ALTER TABLE boxes ADD COLUMN track TEXT")
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


def box_fragments(boxes, rows):
    """idtrackerai fragment of each box: the machine row whose centroid is inside the box, nearest its centre,
    one row per box (as the client's fragmentsOf) -> [fragment or None]. Tag boxes (FMB, NEW-FRAGMENT, CROSSING)
    take no row. A row inside a box with a local identity that got no row of its own is one blob holding two
    flies (a crossing): its fragment is given to neither box, or a label given to one fly would be carried
    through the fragment to the other."""
    def inside(b, r):
        return b["x0"] <= r["x"] <= b["x1"] and b["y0"] <= r["y"] <= b["y1"]

    def is_lid(b):
        return str(b.get("label") or "").isdigit()

    pairs = []
    for i, b in enumerate(boxes):
        if b.get("label") in TAGS:
            continue
        for j, r in enumerate(rows):
            if r.get("fragment") is None or not inside(b, r):
                continue
            pairs.append((((r["x"] - (b["x0"] + b["x1"]) / 2) ** 2 + (r["y"] - (b["y0"] + b["y1"]) / 2) ** 2), i, j))
    pairs.sort()
    row_of, used = [None] * len(boxes), set()
    for _, i, j in pairs:
        if row_of[i] is None and j not in used:
            row_of[i] = j
            used.add(j)
    shared = set()
    for i, b in enumerate(boxes):
        if row_of[i] is None and is_lid(b):
            for k, j in enumerate(row_of):
                if j is not None and inside(b, rows[j]):
                    shared.update((i, k))
                    break
    return [None if j is None or i in shared else int(rows[j]["fragment"]) for i, j in enumerate(row_of)]


def tag_fragments(boxes, rows):
    """Fragments a NEW-FRAGMENT tag splits in this frame: per tag box, the machine row whose centroid is inside it,
    nearest its centre (flyhostel's integration gives that fly a new fragment from the tagged frame to the end of the
    chunk)."""
    out = set()
    for b in boxes:
        if b.get("label") != "NEW-FRAGMENT":
            continue
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
        inside = [r for r in rows if r.get("fragment") is not None
                  and b["x0"] <= r["x"] <= b["x1"] and b["y0"] <= r["y"] <= b["y1"]]
        if inside:
            r = min(inside, key=lambda r: (r["x"] - cx) ** 2 + (r["y"] - cy) ** 2)
            out.add(int(r["fragment"]))
    return out


def segment_key(fragment, frame_number, splits):
    """Key of the stretch of a fragment holding that frame: "<fragment>" up to its first NEW-FRAGMENT tag, then
    "<fragment>@<frame of the last tag at or before it>". Labels are carried within a stretch only.
    splits: {fragment: [frames]}"""
    if fragment is None:
        return None
    start = max((f for f in splits.get(fragment, ()) if f <= frame_number), default=None)
    return str(fragment) if start is None else f"{fragment}@{start}"


def _saved_chunk(con, dbfile, chunk, chunksize, skip=None):
    """Saved boxes of a chunk ({frame: [box]}, but `skip`) and the machine rows of those frames"""
    lo, hi = chunk * chunksize, (chunk + 1) * chunksize
    boxes = {}
    for fn, x0, y0, x1, y1, label, track in con.execute(
            "SELECT b.frame_number, b.x0, b.y0, b.x1, b.y1, b.label, b.track FROM boxes b JOIN frames f ON f.frame_number = b.frame_number "
            "WHERE b.frame_number >= ? AND b.frame_number < ? ORDER BY b.frame_number, b.rowid", (lo, hi)):
        if fn != skip:
            boxes.setdefault(fn, []).append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "label": label or "",
                                             **({"track": track} if track else {})})
    rows = {}
    if boxes:
        mcon = sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True)
        try:
            frames = sorted(boxes)
            for k in range(0, len(frames), 500):
                part = frames[k:k + 500]
                for fn, x, y, frag in mcon.execute(
                        f"SELECT frame_number, x, y, fragment FROM ROI_0 WHERE frame_number IN ({','.join('?' * len(part))})", part):
                    rows.setdefault(fn, []).append({"x": x, "y": y, "fragment": frag})
        finally:
            mcon.close()
    return boxes, rows


def _splits(boxes, rows):
    """{fragment: [frames]} of the NEW-FRAGMENT tags of these saved frames"""
    splits = {}
    for fn in sorted(boxes):
        for frag in tag_fragments(boxes[fn], rows.get(fn, [])):
            splits.setdefault(frag, []).append(fn)
    return splits


_chunk_labels = {}   # (sidecar, dbfile, chunk) -> (key, ({segment key: [(frame, label)] sorted}, splits))


def chunk_fragment_labels(con, dbfile, chunk, chunksize, sidecar):
    """The local identities a person gave in the saved frames of a chunk, per stretch of idtrackerai fragment
    (`segment_key`: the integration spreads each through its fragment, split by NEW-FRAGMENT tags):
    ({key: [(frame_number, label)]}, splits {fragment: [frames]}). Cached until a frame of the chunk is saved or
    unsaved."""
    lo, hi = chunk * chunksize, (chunk + 1) * chunksize
    key = con.execute("SELECT COUNT(*), MAX(updated) FROM frames WHERE frame_number >= ? AND frame_number < ?", (lo, hi)).fetchone()
    cache_key = (sidecar, dbfile, chunk)
    hit = _chunk_labels.get(cache_key)
    if hit and hit[0] == key:
        return hit[1]
    boxes, rows = _saved_chunk(con, dbfile, chunk, chunksize)
    splits = _splits(boxes, rows)
    labels = {}
    for fn, bs in boxes.items():
        for b, frag in zip(bs, box_fragments(bs, rows.get(fn, []))):
            if frag is not None and b["label"].isdigit():
                labels.setdefault(segment_key(frag, fn, splits), []).append((fn, b["label"]))
    _chunk_labels[cache_key] = (key, (labels, splits))
    return labels, splits


def inherit_labels(boxes, rows, labels, frame_number, splits=None):
    """Boxes of a proposal whose fragment a person labelled in another saved frame of the chunk take that
    label (from the nearest saved frame before, else after), marked inherited = that frame; a local
    identity already on another box swaps with it."""
    if not labels:
        return boxes
    out = [dict(b) for b in boxes]
    splits = dict(splits or {})
    here = tag_fragments(out, rows)     # this frame's own tags (a proposal has none)
    for frag in here:
        splits[frag] = sorted(set(splits.get(frag, [])) | {frame_number})
    for i, frag in enumerate(box_fragments(out, rows)):
        frag = segment_key(frag, frame_number, splits)
        if frag is None or frag not in labels:
            continue
        before = [(fn, lab) for fn, lab in labels[frag] if fn < frame_number]
        after = [(fn, lab) for fn, lab in labels[frag] if fn > frame_number]
        if not before and not after:
            continue
        fn, label = before[-1] if before else after[0]
        old = out[i]["label"]
        if old != label:
            for j, b in enumerate(out):
                if j != i and b["label"] == label:
                    b["label"] = old if old.isdigit() else ""
                    b.pop("inherited", None)
        out[i]["label"] = label
        out[i]["inherited"] = fn
    return out


def relabel_by_fragment(boxes, fragments, mapping):
    """Boxes with the labels of `mapping` (segment key -> local identity) on the boxes of those fragment stretches
    (`fragments`: their keys); a local identity already on another box swaps with it (as in the app).
    -> (boxes, changed)"""
    out = [dict(b) for b in boxes]
    changed = False
    for frag, label in mapping.items():
        if frag not in fragments:
            continue
        i = fragments.index(frag)
        old = out[i]["label"]
        if old == label:
            continue
        for j, b in enumerate(out):
            if j != i and b["label"] == label:
                b["label"] = old if old.isdigit() else ""
        out[i]["label"] = label
        changed = True
    return out, changed


def relabel_chunk(con, dbfile, chunk, chunksize, mapping, skip, author, skip_tags=()):
    """`mapping` (segment key -> local identity) written into every saved frame of the chunk (but `skip`) holding a
    box of those fragment stretches. skip_tags: the fragments the NEW-FRAGMENT tags of frame `skip` split, as the
    person has it now (maybe not saved yet). -> frames changed"""
    boxes, rows = _saved_chunk(con, dbfile, chunk, chunksize, skip=skip)
    if not boxes:
        return []
    splits = _splits(boxes, rows)
    for frag in skip_tags:
        splits[frag] = sorted(set(splits.get(frag, [])) | {skip})
    now, changed = time.time(), []
    for fn, bs in boxes.items():
        keys = [segment_key(f, fn, splits) for f in box_fragments(bs, rows.get(fn, []))]
        new, did = relabel_by_fragment(bs, keys, mapping)
        if not did:
            continue
        con.execute("DELETE FROM boxes WHERE frame_number = ?", (fn,))
        con.executemany("INSERT INTO boxes (frame_number, x0, y0, x1, y1, label, track) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [(fn, b["x0"], b["y0"], b["x1"], b["y1"], b["label"], b.get("track")) for b in new])
        con.execute("UPDATE frames SET author = ?, updated = ? WHERE frame_number = ?", (author, now, fn))
        con.execute("INSERT INTO log VALUES (?, ?, ?, ?)", (now, author, fn, json.dumps(new)))
        changed.append(fn)
    return sorted(changed)


def fragment_events(dbfile, frame_number, chunksize):
    """idtrackerai fragments starting / ending at this frame (not present in the previous / next frame of the
    chunk): where a track was lost or picked up, so where identities switch"""
    con = sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True)
    try:
        frags = {}
        for fn, frag in con.execute("SELECT frame_number, fragment FROM ROI_0 WHERE frame_number BETWEEN ? AND ?",
                                    (frame_number - 1, frame_number + 1)):
            if frag is not None:
                frags.setdefault(fn, set()).add(int(frag))
    finally:
        con.close()
    here = frags.get(frame_number, set())
    chunk = frame_number // chunksize
    starts = sorted(here - frags.get(frame_number - 1, set())) if (frame_number - 1) // chunksize == chunk else []
    ends = sorted(here - frags.get(frame_number + 1, set())) if (frame_number + 1) // chunksize == chunk else []
    return {"starts": starts, "ends": ends}


_start_hours = {}


def _clock(dbfile, frame_number):
    """Seconds since 00:00 of the recording's first day on the clock of the folder names (UTC) at that frame,
    or None: imgstore's frame_time (STORE_INDEX, ms) counts from the o'clock hour the recording started in
    (METADATA date_time floored to the hour), as the Group viewer's ZT."""
    try:
        with sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True, timeout=5) as conn:
            if dbfile not in _start_hours:
                value = conn.execute("SELECT value FROM METADATA WHERE field = 'date_time'").fetchone()
                start = int(float(value[0]))
                _start_hours[dbfile] = (start - start % 3600) % 86400
            row = conn.execute("SELECT frame_time FROM STORE_INDEX WHERE frame_number = ?", (frame_number,)).fetchone()
    except (sqlite3.Error, TypeError, ValueError) as error:
        logger.warning("No time for frame %s: %s", frame_number, error)
        return None
    return None if row is None or row[0] is None else _start_hours[dbfile] + row[0] / 1000


def _clocks(dbfile, frame_numbers):
    """_clock of many frames in few queries: {frame_number: seconds} (frames without time left out)."""
    frames = sorted(set(int(f) for f in frame_numbers))
    if not frames:
        return {}
    first = _clock(dbfile, frames[0])     # also caches the start hour
    if first is None:
        return {}
    out = {}
    try:
        with sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True, timeout=5) as conn:
            for k in range(0, len(frames), 900):
                part = frames[k:k + 900]
                rows = conn.execute(
                    f"SELECT frame_number, frame_time FROM STORE_INDEX WHERE frame_number IN ({','.join('?' * len(part))})",
                    part)
                out.update({fn: _start_hours[dbfile] + t / 1000 for fn, t in rows if t is not None})
    except sqlite3.Error as error:
        logger.warning("No times for the scenes: %s", error)
    return out


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
    nearest within 20 px), at least MIN_BOX px wide and high, labelled with the local identity.
    Flies touching make one blob holding several centroids: each of them gets instead a box centred
    on its centroid, of the median size of the frame's single-fly boxes."""
    polys = [np.asarray(c, dtype=np.float32).reshape((-1, 1, 2)) for c in contours if len(c) >= 3]
    found = []
    for row in rows:
        x, y = float(row["x"]), float(row["y"])
        best, best_d = None, -20.0
        for k, poly in enumerate(polys):
            d = cv2.pointPolygonTest(poly, (x, y), True)
            if d > best_d:
                best, best_d = k, d
        found.append((row, x, y, best))
    holders = {}
    for _, _, _, k in found:
        if k is not None:
            holders[k] = holders.get(k, 0) + 1

    def blob_box(k):
        bx, by, bw, bh = cv2.boundingRect(polys[k].astype(np.int32))
        return bx - PAD, by - PAD, bx + bw + PAD, by + bh + PAD

    single = [blob_box(k) for k, n in holders.items() if n == 1]
    if single:
        fly_w = float(np.median([x1 - x0 for x0, _, x1, _ in single]))
        fly_h = float(np.median([y1 - y0 for _, y0, _, y1 in single]))
    else:
        fly_w = fly_h = float(MIN_BOX)
    boxes = []
    for row, x, y, k in found:
        if k is not None and holders[k] == 1:
            x0, y0, x1, y1 = blob_box(k)
        elif k is not None:
            x0, y0, x1, y1 = x - fly_w / 2, y - fly_h / 2, x + fly_w / 2, y + fly_h / 2
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
    """Boxes as stored ({x0, y0, x1, y1, label[, track]}), or ValueError."""
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
        box_out = {"x0": round(x0, 1), "y0": round(y0, 1), "x1": round(x1, 1), "y1": round(y1, 1), "label": label}
        track = box.get("track")
        if track:
            if not isinstance(track, str) or len(track) > 64:
                raise ValueError(f"bad track {track!r}")
            box_out["track"] = track
        out.append(box_out)
    return out


BUNDLE_ERROR = ("this is a minimal bundle from the Drive exchange (the scene frames only): scenes, proposals and the "
                "integration are made by the publisher on the whole experiment")


def read_scenes(basedir, experiment, n_animals):
    """[{start, length, reason}] of the frames to check, and where they come from."""
    # a minimal bundle of the Drive exchange: the scenes as listed where it was made (boundaries included)
    if bundle_info(basedir) is not None:
        scenes = read_bundle_scenes(basedir)
        if scenes is not None:
            return scenes, {"kind": "bundle", "path": os.path.join(_validation_folder(basedir), "scenes.json")}
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
                # the machine QC index starts a scene 2 frames before its first failure: a failure at frame 0
                # gives a scene at -2, whose frames before 0 do not exist
                start = int(float(row["scene_start"]))
                if start < 0:
                    length += start
                    start = 0
                if length <= 0:
                    continue
                scenes.append({"start": start, "length": length, "reason": ", ".join(reasons)})
        scenes.sort(key=lambda s: s["start"])
        return scenes, {"kind": "scene_qc", "path": path}
    # only the validation pipeline's scenes (Prepare scenes): the QC tab's runs of bad frames are not used
    return [], {"kind": None, "path": None}


def tracked_frame_range(dbfile):
    """(first, last) frame number with rows in ROI_0, or (None, None). Separate queries, so both use the index."""
    with sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True) as conn:
        first = conn.execute("SELECT MIN(frame_number) FROM ROI_0").fetchone()[0]
        last = conn.execute("SELECT MAX(frame_number) FROM ROI_0").fetchone()[0]
    return first, last


def link_boundary(before, after, n_animals):
    """Reasons ([] if none) why the integration's link of one chunk boundary may be wrong.

    before / after: [(local_identity, x, y)] of the last frame of a chunk / the first of the next (machine tables).
    Same greedy matching as flyhostel's make_identity_table: in local identity order, the nearest free fly.
    """
    reasons = []
    lids_before = sorted(row for row in before if row[0])
    lids_after = [row for row in after if row[0]]
    for name, rows, all_rows in (("last frame", lids_before, before), ("first frame", lids_after, after)):
        lids = [row[0] for row in rows]
        if len(lids) != n_animals or len(set(lids)) != len(lids) or len(all_rows) != len(rows):
            reasons.append(f"{len(set(lids))} of {n_animals} flies identified in the {name}")
    used = set()
    for lid, x, y in lids_before:
        candidates = sorted((float(np.hypot(x - ax, y - ay)), alid) for alid, ax, ay in lids_after if alid not in used)
        if not candidates:
            break
        distance, chosen = candidates[0]
        used.add(chosen)
        if distance > BOUNDARY_JUMP_PX:
            reasons.append(f"fly {lid} → {chosen} {distance:.0f} px away")
        if len(candidates) > 1 and candidates[1][0] - distance < BOUNDARY_MARGIN_PX:
            reasons.append(f"fly {lid}: {chosen} at {distance:.0f} px, {candidates[1][1]} at {candidates[1][0]:.0f} px")
    return reasons


def unsure_boundaries(dbfile, chunksize, n_animals):
    """[(first frame of the chunk after, reasons)] of the tracked chunk boundaries the integration may link wrongly.
    ~20 ms; cached per database mtime."""
    if n_animals < 2:
        return []
    mtime = os.path.getmtime(dbfile)
    cached = _boundary_cache.get(dbfile)
    if cached and cached[0] == mtime and cached[1] == n_animals:
        return cached[2]
    first, last = tracked_frame_range(dbfile)
    out = []
    if first is not None:
        boundaries = [c * chunksize for c in range(first // chunksize + 1, last // chunksize + 1)]
        frames = [f for b in boundaries for f in (b - 1, b)]
        rows = {}
        with sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True) as conn:
            for i in range(0, len(frames), 500):
                part = frames[i:i + 500]
                for fn, lid, x, y in conn.execute(
                        "SELECT r.frame_number, i.local_identity, r.x, r.y FROM ROI_0 r JOIN IDENTITY i "
                        "ON i.frame_number = r.frame_number AND i.in_frame_index = r.in_frame_index "
                        f"WHERE r.frame_number IN ({','.join('?' * len(part))})", part):
                    rows.setdefault(fn, []).append((int(lid or 0), float(x), float(y)))
        for b in boundaries:
            reasons = link_boundary(rows.get(b - 1, []), rows.get(b, []), n_animals)
            if reasons:
                out.append((b, reasons))
    _boundary_cache[dbfile] = (mtime, n_animals, out)
    return out


def unsure_boundaries_async(dbfile, chunksize, n_animals):
    """The cached unsure_boundaries, or None while they are checked in a background thread
    (7-50 s on a database the HDD has not read lately; ~10 ms once its pages are cached)."""
    cached = _boundary_cache.get(dbfile)
    try:
        if cached and cached[0] == os.path.getmtime(dbfile) and cached[1] == n_animals:
            return cached[2]
    except OSError:
        return []
    with _lock:
        if dbfile in _boundary_running:
            return None
        _boundary_running.add(dbfile)

    def work():
        try:
            unsure_boundaries(dbfile, chunksize, n_animals)
        except Exception:
            logger.exception("Could not check the chunk boundaries of %s", dbfile)
            _boundary_cache[dbfile] = (os.path.getmtime(dbfile), n_animals, [])
        finally:
            with _lock:
                _boundary_running.discard(dbfile)

    threading.Thread(target=work, name="boundaries", daemon=True).start()
    return None


def add_boundary_scenes(scenes, boundaries, chunksize):
    """Scenes plus one per unsure boundary; a scene already holding both sides of a boundary gets its reason."""
    scenes = [dict(sc) for sc in scenes]
    for b, reasons in boundaries:
        reason = f"chunk {b // chunksize - 1}→{b // chunksize} link: " + "; ".join(reasons)
        covering = next((sc for sc in scenes if sc["start"] <= b - 1 and b < sc["start"] + sc["length"]), None)
        if covering is not None:
            covering["reason"] = "; ".join(r for r in (covering.get("reason"), reason) if r)
            covering["boundary"] = b
        else:
            scenes.append({"start": max(0, b - BOUNDARY_CONTEXT), "length": 2 * BOUNDARY_CONTEXT,
                           "reason": reason, "boundary": b})
    scenes.sort(key=lambda sc: sc["start"])
    return scenes


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


# --- validation queue: analyzed experiments without the validated tables, with this person's progress ---
QUEUE_FILE = os.environ.get("ANNOTATE_QUEUE_FILE",
                            os.path.expanduser("~/.config/idtrackerai-validator/annotate_queue.json"))
QUEUE_ORDER = {"in_progress": 0, "to_do": 1, "no_scenes": 2, "annotated": 3}
_progress_cache = {}          # dbfile -> (key, progress)
_progress_lock = threading.Lock()


def _is_mine(author, me):
    """A save of this person: their own, or the labels they carried to other frames ("<me> (fragments of frame N)")"""
    return bool(author) and (author == me or author.startswith(me + " ("))


def _mtime_size(path):
    try:
        stat = os.stat(path)
        return stat.st_mtime, stat.st_size
    except OSError:
        return None


def experiment_progress(basedir, experiment, n_animals, me):
    """Scenes of an experiment (scene_qc.csv; the chunk boundaries are left out:
    checking them reads the database) and the frames saved in its sidecar: per scene done = every frame
    saved (by anyone), `mine` = frames this person saved. Cached per sidecar / scene_qc.csv mtime and size."""
    sidecar = sidecar_path(basedir)
    key = (_mtime_size(sidecar), _mtime_size(os.path.join(_validation_folder(basedir), "scene_qc.csv")), me)
    with _progress_lock:
        cached = _progress_cache.get(basedir)
    if cached and cached[0] == key:
        return cached[1]
    scenes, source = read_scenes(basedir, experiment, n_animals)
    saved, mine, last_mine = set(), 0, None
    if key[0] is not None:
        try:
            con = sqlite3.connect(f"file:{sidecar}?mode=ro", uri=True, timeout=5)
            try:
                for fn, author, updated in con.execute("SELECT frame_number, author, updated FROM frames"):
                    saved.add(fn)
                    if _is_mine(author, me):
                        mine += 1
                        if updated is not None and (last_mine is None or updated > last_mine):
                            last_mine = updated
            finally:
                con.close()
        except sqlite3.Error as error:
            logger.warning("Cannot read %s: %s", sidecar, error)
    done = sum(all(fn in saved for fn in range(sc["start"], sc["start"] + sc["length"])) for sc in scenes)
    if not scenes:
        status = "no_scenes"
    elif done == len(scenes):
        status = "annotated"
    elif mine:
        status = "in_progress"
    else:
        status = "to_do"
    progress = {"status": status, "scenes": len(scenes), "scenes_done": done, "scene_source": source["kind"],
                "annotated": len(saved), "mine": mine, "last_mine": last_mine}
    with _progress_lock:
        _progress_cache[basedir] = (key, progress)
    return progress


def read_hidden(me):
    try:
        with open(QUEUE_FILE) as handle:
            return set(json.load(handle).get("hidden", {}).get(me, []))
    except (OSError, ValueError):
        return set()


def write_hidden(me, experiment, hidden):
    try:
        with open(QUEUE_FILE) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = {}
    names = set(data.setdefault("hidden", {}).get(me, []))
    (names.add if hidden else names.discard)(experiment)
    data["hidden"][me] = sorted(names)
    os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)
    tmp = QUEUE_FILE + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(data, handle, indent=1)
    os.replace(tmp, QUEUE_FILE)
    return names


def validation_queue(root, me):
    """Every database of index.txt (analyzed) without the validated tables (experiments.VALIDATED_TABLES),
    with its number of animals and this person's progress. ~25 s cold on the HDDs (sqlite_master of
    every database), then cached per file."""
    index = os.path.join(root, "index.txt")
    try:
        with open(index) as handle:
            paths = [line.strip() for line in handle if line.strip()]
    except OSError:
        return [], 0
    hidden = read_hidden(me)
    from idtrackerai_validator_server.gdrive_exchange import read_local    # it imports this module

    def one(relpath):
        path = os.path.normpath(os.path.join(root, relpath))
        try:
            stat = os.stat(path)
        except OSError:
            return None
        if stat.st_size == 0:
            return None
        validated = is_validated(path, stat)
        if validated:
            return None
        experiment = os.path.basename(path)[:-len(".db")] if path.endswith(".db") else os.path.basename(path)
        basedir = os.path.dirname(path)
        try:
            n_animals = get_number_of_animals(experiment)
        except Exception:     # unreadable conf: still listed
            n_animals = None
        item = {"experiment": experiment, "relpath": os.path.relpath(basedir, root), "n_animals": n_animals,
                "date": (m.group(1) if (m := re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", experiment)) else None),
                "db_mtime": stat.st_mtime, "hidden": experiment in hidden, "unreadable": validated is None}
        # Drive exchange of this copy (flyhostel/validation/exchange.json): published from here, or downloaded
        local = read_local(basedir) or {}
        item["exchange"] = {k: local.get(k) for k in ("role", "kind", "published_at", "downloaded_at", "publisher")} \
            if local.get("role") else None
        try:
            item.update(experiment_progress(basedir, experiment, n_animals or 1, me))
        except Exception as error:
            logger.warning("Validation queue: cannot read the scenes of %s: %s", experiment, error)
            item.update({"status": "no_scenes", "scenes": 0, "scenes_done": 0, "annotated": 0, "mine": 0,
                         "last_mine": None, "error": str(error)})
        return item

    with ThreadPoolExecutor(max_workers=16) as pool:
        items = [item for item in pool.map(one, paths) if item is not None]
    items.sort(key=lambda it: (QUEUE_ORDER[it["status"]], -(it["last_mine"] or 0), it["date"] or "", it["experiment"]))
    return items, len(paths)


# --- validation queue: metadata of one experiment, shown on hover ---
# per-fly identifiers and bookkeeping, never shown; the rest is shown when it takes more than one value in the sheets
META_SKIP = {"Dropbox_link", "basedir", "row_id", "flyhostel_number", "flyhostel_date", "flyhostel_time", "comment",
             "region_id", "machine_name", "date", "identity", "fly_no", "meta", "reason", "node", "physical_arena",
             "side", "fly_opposite", "GBX_opposite", "number_of_animals", "tube"}
META_FIRST = ("genotype", "sex", "light_cycle", "age_min", "age_max", "incubator", "GBX", "reference_hour")
_varying = {"key": None, "columns": set()}
_meta_cache = {}              # db path -> (mtime, meta)
_meta_lock = threading.Lock()


def varying_columns():
    """Columns of the metadata sheets (latest export of each) taking at least two values, minus META_SKIP;
    cached per file mtime. Empty set (= show every column) when the sheets cannot be read."""
    try:
        from idtrackerai_validator_server.overview import _clean, sheet_files
        files = sorted(sheet_files()[0].values())
        key = tuple((f, os.path.getmtime(f)) for f in files)
    except Exception as error:
        logger.warning("Validation queue: cannot read the metadata sheets: %s", error)
        return set()
    with _meta_lock:
        if _varying["key"] == key:
            return _varying["columns"]
    values = {}
    for path in files:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                for column, value in row.items():
                    value = _clean(value)
                    if column and value is not None:
                        values.setdefault(column, set()).add(value)
    columns = {c for c, v in values.items() if len(v) > 1 and c not in META_SKIP}
    with _meta_lock:
        _varying.update(key=key, columns=columns)
    return columns


def experiment_meta(root, experiment):
    """Fly metadata of an analyzed experiment (its database's ethoscope_metadata, read as the Overview does:
    light cycle merged): per column [[value, flies]], key columns first, then the ones that vary in the sheets."""
    from idtrackerai_validator_server.experiments import to_relpath
    from idtrackerai_validator_server.overview import _read_experiment
    relpath = os.path.join(to_relpath(experiment), experiment + ".db")
    path = os.path.join(root, relpath)
    mtime = os.path.getmtime(path)
    with _meta_lock:
        cached = _meta_cache.get(path)
    if cached and cached[0] == mtime:
        record = cached[1]
    else:
        record = _read_experiment(root, relpath)
        with _meta_lock:
            _meta_cache[path] = (mtime, record)
    varying = varying_columns()
    counts = {}
    for fly in record["flies"]:
        for column, value in fly.items():
            if column in META_SKIP or column == "arena" or (varying and column not in varying and column not in META_FIRST):
                continue
            counts.setdefault(column, {}).setdefault(value, 0)
            counts[column][value] += 1
    order = [c for c in META_FIRST if c in counts] + sorted(c for c in counts if c not in META_FIRST)
    return {"experiment": experiment, "flies": len(record["flies"]), "arena": record.get("arena"),
            "group_size": record.get("group_size"), "duration_h": record.get("duration_h"), "error": record.get("error"),
            "columns": [[c, sorted(counts[c].items(), key=lambda kv: (-kv[1], kv[0]))] for c in order]}


# --- what the scene proposer looked at: the tracking it read, the database's tracked range, its scenes ---
_input_cache = {}            # feather path -> ((mtime, size), (first, last))


def _feather_range(path):
    """(first, last) frame_number of the tracking data make-identogram wrote and read
    (flyhostel/validation/<experiment>_tracking_data.feather); cached per mtime / size; None without pyarrow."""
    key = _mtime_size(path)
    cached = _input_cache.get(path)
    if cached and cached[0] == key:
        return cached[1]
    try:
        import pyarrow.compute as pc
        import pyarrow.feather as pf
        column = pf.read_table(path, columns=["frame_number"], memory_map=True)["frame_number"]
        bounds = pc.min_max(column)
        result = (int(bounds["min"].as_py()), int(bounds["max"].as_py())) if len(column) else None
    except Exception as error:
        logger.warning("Cannot read the frame range of %s: %s", path, error)
        result = None
    _input_cache[path] = (key, result)
    return result


def scene_coverage(basedir, dbfile, scenes, chunksize):
    """Frames (with their clock) the scene proposer covered: `input` = the tracking data it read, `tracked` = ROI_0's
    range now, `recorded` = the store's index.db, `scenes` = first / last scene of scene_qc.csv, `scene_qc` = when it was made; `untracked_chunks`:
    chunks tracked after the proposer ran (Prepare scenes again)."""
    folder = _validation_folder(basedir)
    feathers = sorted(glob.glob(os.path.join(folder, "*_tracking_data.feather")), key=os.path.getmtime)
    rng = _feather_range(feathers[-1]) if feathers else None
    try:
        tracked = tracked_frame_range(dbfile)
    except sqlite3.Error:
        tracked = (None, None)
    first_scene = scenes[0]["start"] if scenes else None
    last_scene = max((sc["start"] for sc in scenes), default=None)
    frames = [f for f in (*(rng or ()), *tracked, first_scene, last_scene) if f is not None]
    clocks = _clocks(dbfile, frames)

    def span(a, b):
        return None if a is None or b is None else {"first": a, "last": b, "first_clock": clocks.get(a), "last_clock": clocks.get(b)}

    # the recording: the store's own index.db (frame_time on the same clock as the database's STORE_INDEX)
    recorded = None
    try:
        with sqlite3.connect(f"file:{os.path.join(basedir, 'index.db')}?mode=ro", uri=True, timeout=5) as conn:
            first = conn.execute("SELECT frame_number, frame_time FROM frames ORDER BY frame_number LIMIT 1").fetchone()
            last = conn.execute("SELECT frame_number, frame_time FROM frames ORDER BY frame_number DESC LIMIT 1").fetchone()
        if first and last and dbfile in _start_hours:
            recorded = {"first": first[0], "last": last[0],
                        "first_clock": _start_hours[dbfile] + first[1] / 1000 if first[1] is not None else None,
                        "last_clock": _start_hours[dbfile] + last[1] / 1000 if last[1] is not None else None}
    except sqlite3.Error as error:
        logger.warning("Cannot read the recording's index.db: %s", error)
    scene_qc = os.path.join(folder, "scene_qc.csv")
    missing = None
    # older make-identogram versions left out the last chunk: one chunk short is not a sign the scenes are old
    if rng and tracked[1] is not None and chunksize and tracked[1] // chunksize > rng[1] // chunksize + 1:
        missing = [int(rng[1] // chunksize) + 1, int(tracked[1] // chunksize)]
    return {"input": span(*rng) if rng else None,
            "input_path": feathers[-1] if feathers else None,
            "input_mtime": os.path.getmtime(feathers[-1]) if feathers else None,
            "tracked": span(*tracked), "recorded": recorded, "scenes": span(first_scene, last_scene),
            "scene_qc_mtime": os.path.getmtime(scene_qc) if os.path.exists(scene_qc) else None,
            "untracked_chunks": missing,
            "outdated": scenes_outdated(basedir, dbfile, chunksize)}


def scenes_outdated(basedir, dbfile, chunksize):
    """Why scene_qc.csv may be older than the tracking, or None. Quick signs only (the content check of every chunk takes
    minutes): the tracking was concatenated again (idtrackerai/concatenation-overlap.csv, rewritten by every idtrackerai.nf
    run) after the tracking data the scenes were made from was exported, or the database is tracked more than one chunk
    beyond it (older make-identogram versions left out the last chunk: one chunk short is not a sign) or not as far."""
    folder = _validation_folder(basedir)
    scene_qc = os.path.join(folder, "scene_qc.csv")
    if not os.path.exists(scene_qc):
        return None
    feathers = sorted(glob.glob(os.path.join(folder, "*_tracking_data.feather")), key=os.path.getmtime)
    made = os.path.getmtime(feathers[-1]) if feathers else os.path.getmtime(scene_qc)
    overlap = os.path.join(basedir, "idtrackerai", "concatenation-overlap.csv")
    if os.path.exists(overlap) and os.path.getmtime(overlap) > made + 60:
        return (f"the tracking was redone on {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(overlap)))}, "
                f"after the scenes were made ({time.strftime('%Y-%m-%d %H:%M', time.localtime(made))})")
    rng = _feather_range(feathers[-1]) if feathers else None
    if rng and chunksize:
        try:
            last = tracked_frame_range(dbfile)[1]
        except sqlite3.Error:
            last = None
        # further than the scenes' input (beyond the one-chunk shortfall), or not as far (tracked again over a shorter range)
        if last is not None and (last // chunksize > rng[1] // chunksize + 1 or last // chunksize < rng[1] // chunksize):
            return (f"the database is tracked up to chunk {int(last // chunksize)}, the scenes were made from the tracking "
                    f"up to chunk {int(rng[1] // chunksize)}")
    return None


def _running_job(job_id):
    job = _jobs.get(str(job_id)) if job_id is not None else None
    return job.to_dict() if job is not None and job.status == "running" else None


def _author():
    """The login user, else the annotator name of the Drive exchange tab (gdrive_exchange.py), else $USER"""
    if session.get("user"):
        return session["user"]
    from idtrackerai_validator_server.gdrive_exchange import read_config
    return read_config()["annotator"] or os.environ.get("USER") or "anonymous"


def register_box_annotations(app, get_state):
    """get_state() -> None (no database) or a dict with experiment, basedir, dbfile, chunksize,
    n_animals, contours(frame_number) -> list, size() -> (width, height)."""

    def state_or_error():
        state = get_state()
        if state is None:
            return None, (jsonify({"error": "load an experiment with a tracking database"}), 400)
        return state, None

    @app.route("/api/annotate/link", methods=["GET"])
    def annotate_link():
        """?frame_number=N[&experiment=E][&tab=annotate|group_viewer]: link opening E (default the loaded one)
        at frame N, with the address this browser used (links.frame_url)."""
        experiment = request.args.get("experiment")
        if not experiment:
            state = get_state()
            if state is None:
                return jsonify({"error": "no experiment given or loaded"}), 400
            experiment = state["experiment"]
        try:
            frame_number = int(request.args["frame_number"])
            url = links.frame_url(experiment, frame_number, tab=request.args.get("tab", "annotate"),
                                  base_url=request.host_url)
        except (KeyError, ValueError) as error:
            return jsonify({"error": f"frame_number (integer) and tab: {error}"}), 400
        return jsonify({"url": url, "experiment": to_flat(experiment), "frame_number": frame_number})

    @app.route("/api/annotate/scenes", methods=["GET"])
    def annotate_scenes():
        state, error = state_or_error()
        if error:
            return error
        scenes, source = read_scenes(state["basedir"], state["experiment"], state["n_animals"])
        bundle = source["kind"] == "bundle"
        # the range the proposer looked at (before the chunk boundaries are added: only scene_qc.csv's scenes)
        coverage = None if bundle or source["kind"] != "scene_qc" else \
            scene_coverage(state["basedir"], state["dbfile"], scenes, state["chunksize"])
        # a bundle's database holds only the scene frames: its boundaries are in scenes.json already
        boundaries = [] if bundle else unsure_boundaries_async(state["dbfile"], state["chunksize"], state["n_animals"])
        scenes = add_boundary_scenes(scenes, boundaries or [], state["chunksize"])
        # boundaries None: still being checked, the client asks again
        source = {**source, "boundaries": None if boundaries is None else len(boundaries)}
        clocks = _clocks(state["dbfile"], [sc["start"] for sc in scenes])
        for sc in scenes:
            sc["clock"] = clocks.get(sc["start"])
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
            "integrating": _running_job(_integrating.get(state["experiment"])),
            "problems": read_problems(state["basedir"]),
            "flagged": flagged, "proposed_frames": proposed, "proposal_run": run,
            "proposing": _proposing.get(state["experiment"]),
            "preparing": _preparing.get(state["experiment"]),
            "bundle": bundle,
            "coverage": coverage,
        })

    @app.route("/api/annotate/queue", methods=["GET"])
    def annotate_queue():
        """The validation queue (validation_queue) of the person asking: `me`, `items`, `analyzed`."""
        me = _author()
        items, analyzed = validation_queue(os.environ["FLYHOSTEL_VIDEOS"], me)
        return jsonify({"me": me, "items": items, "analyzed": analyzed})

    @app.route("/api/annotate/queue/meta", methods=["GET"])
    def annotate_queue_meta():
        """?experiment=<flat>: experiment_meta, shown when hovering its row of the validation queue"""
        experiment = to_flat(request.args.get("experiment", ""))
        if not experiment or "/" in experiment or ".." in experiment:
            return jsonify({"error": "experiment required"}), 400
        try:
            return jsonify(experiment_meta(os.environ["FLYHOSTEL_VIDEOS"], experiment))
        except OSError:
            return jsonify({"error": f"no database for {experiment}"}), 404

    @app.route("/api/annotate/queue/hide", methods=["POST"])
    def annotate_queue_hide():
        """{"experiment", "hidden": bool}: take it out of this person's queue (or back in)"""
        body = request.get_json(silent=True) or {}
        if not body.get("experiment"):
            return jsonify({"error": "experiment required"}), 400
        hidden = write_hidden(_author(), to_flat(body["experiment"]), bool(body.get("hidden", True)))
        return jsonify({"hidden": sorted(hidden)})

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
                    saved = [{"x0": a, "y0": b, "x1": c, "y1": d, "label": lab, **({"track": tr} if tr else {})}
                             for a, b, c, d, lab, tr in con.execute(
                                 "SELECT x0, y0, x1, y1, label, track FROM boxes WHERE frame_number = ? ORDER BY rowid", (frame_number,))]
                pf = con.execute("SELECT flags FROM proposal_frames WHERE frame_number = ?", (frame_number,)).fetchone()
                if pf:
                    flags = json.loads(pf[0] or "[]")
                    solver = [{"x0": a, "y0": b, "x1": c, "y1": d, "label": lab or "", "note": note or "", "confidence": conf}
                              for a, b, c, d, lab, note, conf in con.execute(
                                  "SELECT x0, y0, x1, y1, label, note, confidence FROM proposals WHERE frame_number = ? ORDER BY rowid",
                                  (frame_number,))]
            con.close()
        proposal, source = None, None
        labels, splits = {}, {}
        con = _connect(state["basedir"])
        if con is not None:
            with con:
                labels, splits = chunk_fragment_labels(con, state["dbfile"], chunk, state["chunksize"], os.path.join(_validation_folder(state["basedir"]), DB_NAME))
            con.close()
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
        if proposal is not None:
            proposal = inherit_labels(proposal, rows, labels, frame_number, splits)
        return jsonify({
            "frame_number": frame_number, "chunk": chunk, "source": source if proposal is not None else "saved",
            "flags": flags, "has_solver": solver is not None,
            "saved": saved is not None, "boxes": proposal if proposal is not None else saved,
            "author": meta[0] if meta else None, "updated": meta[1] if meta else None,
            "machine": rows, "concatenation": _concatenation(state["dbfile"], chunk),
            "clock": _clock(state["dbfile"], frame_number),
            "fragment_events": fragment_events(state["dbfile"], frame_number, state["chunksize"]),
            "fragment_splits": {str(f): fs for f, fs in splits.items()},
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
                con.executemany("INSERT INTO boxes (frame_number, x0, y0, x1, y1, label, track) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                [(frame_number, b["x0"], b["y0"], b["x1"], b["y1"], b["label"], b.get("track")) for b in boxes])
                con.execute("INSERT OR REPLACE INTO frames VALUES (?, ?, ?, ?)", (frame_number, author, now, len(boxes)))
                con.execute("INSERT INTO log VALUES (?, ?, ?, ?)", (now, author, frame_number, json.dumps(boxes)))
            con.close()
        return jsonify({"frame_number": frame_number, "saved": True, "boxes": boxes, "author": author, "updated": now})

    @app.route("/api/annotate/relabel_fragments", methods=["POST"])
    def annotate_relabel_fragments():
        """{"experiment", "frame_number", "mapping": {segment key: local identity}, "tags": [fragments]}: the labels a
        person gave those fragment stretches ("<fragment>" or "<fragment>@<frame>", split by NEW-FRAGMENT tags; `tags`:
        the fragments this frame's tags split) in that frame, written into every saved frame of its chunk holding them
        (swapping as in the app)."""
        state, error = state_or_error()
        if error:
            return error
        body = request.get_json(silent=True) or {}
        if body.get("experiment") and to_flat(body["experiment"]) != to_flat(state["experiment"]):
            return jsonify({"error": "another experiment is loaded"}), 409
        try:
            frame_number = int(body["frame_number"])
            mapping = {}
            for f, label in (body.get("mapping") or {}).items():
                frag, _, start = str(f).partition("@")
                mapping[f"{int(frag)}@{int(start)}" if start else str(int(frag))] = str(label)
            tags = [int(f) for f in body.get("tags") or []]
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "frame_number and mapping {fragment: local identity} needed"}), 400
        valid = {str(i) for i in range(1, state["n_animals"] + 1)}
        if not mapping or any(label not in valid for label in mapping.values()):
            return jsonify({"error": f"local identities are 1..{state['n_animals']}"}), 400
        chunk = frame_number // state["chunksize"]
        with _lock:
            con = _connect(state["basedir"], create=True)
            with con:
                changed = relabel_chunk(con, state["dbfile"], chunk, state["chunksize"], mapping, frame_number,
                                        f"{_author()} (fragments of frame {frame_number})", skip_tags=tags)
            con.close()
        return jsonify({"chunk": chunk, "mapping": mapping, "changed": changed})

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
        if bundle_info(state["basedir"]) is not None:
            return jsonify({"error": BUNDLE_ERROR}), 400
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

    @app.route("/api/annotate/prepare", methods=["POST"])
    def annotate_prepare():
        """Job making flyhostel/validation/scene_qc.csv: make-identogram then auto-annotate-qc, the first steps of
        vsc-scripts/manual/validation/main.sh (one job item each). An existing scene_qc.csv needs {"force": true}."""
        state, error = state_or_error()
        if error:
            return error
        if bundle_info(state["basedir"]) is not None:
            return jsonify({"error": BUNDLE_ERROR}), 400
        body = request.get_json(silent=True) or {}
        experiment = state["experiment"]
        running = _jobs.get(str(_preparing.get(experiment)))
        if running is not None and running.status == "running":
            return jsonify({"error": "the scenes of this experiment are being prepared", "job": running.to_dict()}), 409
        executables = [os.path.join(VALIDATION_ENV, "bin", name) for name in ("make-identogram", "auto-annotate-qc")]
        missing = [e for e in executables if not os.path.exists(e)]
        if missing:
            return jsonify({"error": f"{missing[0]} not found (FLYHOSTEL_VALIDATION_ENV)"}), 500
        folder = _validation_folder(state["basedir"])
        if os.path.exists(os.path.join(folder, "scene_qc.csv")) and not body.get("force"):
            return jsonify({"error": "scene_qc.csv exists already (force to make it again)"}), 409
        flat = to_flat(experiment)
        commands = [
            [executables[0], "--experiment", flat, "--folder", folder, "--time-window-length", str(TIME_WINDOW_LENGTH), "--cache"],
            [executables[1], "--experiment", flat, "--folder", folder, "--n-jobs", str(PREPARE_JOBS)],
        ]

        def work(job):
            # make-identogram writes logs/<scene>_<frame>_status.txt relative to its working directory (main.sh runs
            # in vsc-scripts/manual/validation, which has one): here the validation folder
            os.makedirs(os.path.join(folder, "logs"), exist_ok=True)
            with open(os.path.join(folder, "validation_version.txt"), "w") as handle:
                handle.write(PREPARE_VERSION + "\n")
            env = dict(os.environ, PATH=os.path.join(VALIDATION_ENV, "bin") + os.pathsep + os.environ.get("PATH", ""))
            tails = []
            # make-identogram --cache reads <flat>_tracking_data.feather whenever it exists, whatever the database: an
            # export older than the database (tracked again since) would give the scenes of the old tracking
            stale = []
            for feather in glob.glob(os.path.join(folder, "*_tracking_data.feather")):
                if os.path.getmtime(feather) < os.path.getmtime(state["dbfile"]):
                    os.remove(feather)
                    stale.append(os.path.basename(feather))
            if stale:
                tails.append({"command": "removed tracking exports older than the database: " + ", ".join(stale),
                              "exit_code": 0, "tail": []})
            for i, command in enumerate(commands):
                job.update(i, status="running")
                lines = []
                process = subprocess.Popen(command, cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1)
                for line in process.stdout:
                    lines.append(line.rstrip("\n"))
                    del lines[:-200]
                    match = re.search(r"(\d+)%\|", line)
                    if match:
                        job.update(i, progress=min(0.99, int(match.group(1)) / 100))
                code = process.wait()
                tails.append({"command": " ".join(command), "exit_code": code, "tail": lines[-40:]})
                job.result = {"steps": tails}
                if code != 0:
                    job.update(i, status="failed", error=f"exit code {code}")
                    for j in range(i + 1, len(commands)):
                        job.update(j, status="skipped", error="previous step failed")
                    raise RuntimeError(f"{os.path.basename(command[0])} ended with exit code {code}")
                job.update(i, status="done", progress=1.0)
            scenes, _ = read_scenes(state["basedir"], experiment, state["n_animals"])
            job.result["scenes"] = len(scenes)

        job = start_job(f"Prepare the scenes of {flat}", ["make-identogram", "auto-annotate-qc"], work)
        _preparing[experiment] = job.id
        return jsonify(job.to_dict()), 202

    @app.route("/api/annotate/integrate", methods=["POST"])
    def annotate_integrate():
        state, error = state_or_error()
        if error:
            return error
        if bundle_info(state["basedir"]) is not None:
            return jsonify({"error": BUNDLE_ERROR}), 400
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
        if range_args:
            command += range_args
        elif mode == "save" or body.get("range") == "annotated":
            # save: the range of the integrated feathers
            command += ["--frames-from-annotation"]
        else:
            first, last = tracked_frame_range(state["dbfile"])
            if first is None:
                return jsonify({"error": "nothing tracked in ROI_0"}), 400
            chunksize = state["chunksize"]
            command += ["--fn-interval", str(first // chunksize * chunksize), str((last // chunksize + 1) * chunksize)]
        if mode == "save" and not os.path.exists(os.path.join(folder, "IDENTITY_VAL.feather")):
            return jsonify({"error": "integrate first: no IDENTITY_VAL.feather in the validation folder"}), 400

        # the report prints links to the frames still with problems (links.annotate_url), with the address of this browser
        public_url = request.host_url.rstrip("/")
        who = _author()

        def work(job):
            job.update(0, status="running")
            env = dict(os.environ, PATH=os.path.join(VALIDATION_ENV, "bin") + os.pathsep + os.environ.get("PATH", ""),
                       VALIDATOR_PUBLIC_URL=public_url)
            lines = []
            # own session: Stop kills its worker processes too
            process = subprocess.Popen(command, cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, start_new_session=True)
            _integration_procs[experiment] = process
            for line in process.stdout:
                lines.append(line.rstrip("\n"))
                del lines[:-200]
                match = re.search(r"(\d+)%\|", line)
                if match:
                    job.update(0, progress=min(0.99, int(match.group(1)) / 100))
            code = process.wait()
            _integration_procs.pop(experiment, None)
            stopped = job.id in _stopped
            job.result = {"mode": mode, "command": " ".join(command), "exit_code": code, "tail": lines[-60:],
                          "stopped": stopped}
            if stopped:
                job.update(0, status="failed", error="stopped")
                raise RuntimeError("stopped by " + who)
            if code != 0:
                job.update(0, status="failed", error=f"exit code {code}")
                raise RuntimeError(f"{os.path.basename(executable)} ended with exit code {code}")
            job.update(0, status="done", progress=1.0)

        title = ("Integrate the boxes of " if mode == "integrate" else "Write the validated tables of ") + to_flat(experiment)
        job = start_job(title, [to_flat(experiment)], work)
        _integrating[experiment] = job.id
        return jsonify(job.to_dict()), 202

    @app.route("/api/annotate/integrate/stop", methods=["POST"])
    def annotate_integrate_stop():
        """Kill the running integrate / save of the loaded experiment (its whole process group)."""
        state, error = state_or_error()
        if error:
            return error
        experiment = state["experiment"]
        job = _jobs.get(str(_integrating.get(experiment)))
        process = _integration_procs.get(experiment)
        if job is None or job.status != "running" or process is None or process.poll() is not None:
            return jsonify({"error": "no integration of this experiment is running"}), 409
        _stopped.add(job.id)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        def kill_later():      # workers that ignore SIGTERM
            time.sleep(10)
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        threading.Thread(target=kill_later, daemon=True).start()
        logger.info("Integration job %s of %s stopped by %s", job.id, experiment, _author())
        return jsonify({"stopped": job.id})
