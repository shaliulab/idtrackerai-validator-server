"""Continuity proposal of the Annotate tab: identities followed by position instead of idtrackerai's appearance.

Measured on FlyHostel4_12X_2026-11-15_17-00-00 (3552 frames saved by a person, ZT 11.9–12.4): idtrackerai's local
identities, even mapped per chunk to the person's labels, give 95.4 % of flies / 65.5 % of whole frames right; this
tracker started from the person's labels of a frame gives 98.7 % / 89.5 % (from idtrackerai's first frame: 96.4 % /
74.6 %). The errors left by idtrackerai were neighbouring flies (50–130 px apart) whose identities flip for a few
frames after one of them is not detected for 1–3 frames; two flies present in consecutive frames never swap places.
92 % of the flies a person had to add were within 150 px of a notch (LANDMARKS `notch`: the image is distorted there,
the fly is not segmented or not identified), where it stays: the person's box was a median 11 px from where the fly
was last tracked.

Rules, from an anchor frame to the frame asked, frame by frame in the anchor's chunk:
- anchor: the last frame saved by a person before the frame (its labels), or, if later, the last frame before the
  scenes holding the frame (scene_qc.csv's failures, overlapping / adjacent scenes merged) where idtrackerai
  identified exactly N flies with N distinct local identities, whichever is later; none in the chunk → no proposal.
- each detection (ROI_0 row) gets the fly whose last known position is nearest (one to one, `GATE_PX`).
- a fly not detected keeps its last position. At a notch (≤ `NOTCH_PX` from its centre) it is proposed there, a box
  of the frame's usual size (it sits there, not segmented); a detection near a notch nobody took goes to the fly lost
  at that notch nearest it.
- a detection nobody took elsewhere gets idtrackerai's local identity if that fly is not placed, else no label.
Boxes take `note` / `confidence` (medium = look at it: identity other than idtrackerai's, fly kept at a notch,
ambiguous match; low = no label) as the solver's, so the tab marks them; frame-level problems are `flags`.
"""
import ast
import logging
import os
import sqlite3
import threading
from collections import OrderedDict

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

GATE_PX = 60          # best on the 12X frames (30 px: 71 % of frames right, 60: 89.5 %, 100: 87 %)
NOTCH_PX = 150        # a fly within this of a notch centre is "at the notch"
AMBIGUOUS_PX = 15     # another fly's last position at most this much further than the chosen one: medium
STABLE_FRAMES = 10    # an idtrackerai anchor ends this many consecutive frames with every fly identified, none jumping
SEARCH_STEP = 3000    # frames read at once looking back for an anchor
MAX_RUNS = 8          # runs kept in memory

_lock = threading.Lock()
_runs = OrderedDict()   # (dbfile, anchor key) -> Run
_notches = {}           # dbfile -> [(cx, cy)]
_anchors = {}           # (dbfile, mtime, before, chunk_start) -> (frame, rows)


def notch_centres(dbfile):
    if dbfile not in _notches:
        out = []
        try:
            with sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True) as con:
                for shape, spec in con.execute("SELECT shape, specification FROM LANDMARKS"):
                    if str(shape).lower() != "notch":
                        continue
                    pts = np.asarray(ast.literal_eval(spec), dtype=float)
                    out.append((float(pts[:, 0].mean()), float(pts[:, 1].mean())))
        except (sqlite3.Error, ValueError, SyntaxError) as error:
            logger.warning("No notches in %s: %s", dbfile, error)
        _notches[dbfile] = out
    return _notches[dbfile]


def _near_notch(notches, x, y):
    return any((x - cx) ** 2 + (y - cy) ** 2 <= NOTCH_PX ** 2 for cx, cy in notches)


def read_rows(dbfile, first, last):
    """{frame: [(x, y, local identity or 0)]} of ROI_0 + IDENTITY, in in_frame_index order (as _machine_rows)."""
    out = {}
    with sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True) as con:
        for fn, x, y, lid in con.execute(
                "SELECT r.frame_number, r.x, r.y, i.local_identity FROM ROI_0 r LEFT JOIN IDENTITY i "
                "ON i.frame_number = r.frame_number AND i.in_frame_index = r.in_frame_index "
                "WHERE r.frame_number BETWEEN ? AND ? ORDER BY r.frame_number, r.in_frame_index", (first, last)):
            out.setdefault(fn, []).append((float(x), float(y), int(lid or 0)))
    return out


def stretch_start(scenes, frame_number):
    """First frame of the scenes holding frame_number, overlapping / adjacent scenes merged (frame_number if none)."""
    start = frame_number
    changed = True
    while changed:
        changed = False
        for sc in scenes:
            if sc["start"] < start <= sc["start"] + sc["length"]:
                start = sc["start"]
                changed = True
    return start


def machine_anchor(dbfile, before, chunk_start, n_animals):
    """(frame, rows) of the last frame in [chunk_start, before) ending STABLE_FRAMES consecutive frames whose rows are
    exactly n_animals distinct non-zero local identities, each within GATE_PX of where it was the frame before
    ((None, None) if none; cached per database mtime)."""
    try:
        mtime = os.path.getmtime(dbfile)
    except OSError:
        mtime = None
    key = (dbfile, mtime, before, chunk_start, n_animals)
    if key not in _anchors:
        if len(_anchors) > 1000:
            _anchors.clear()
        _anchors[key] = _machine_anchor(dbfile, before, chunk_start, n_animals)
    return _anchors[key]


def _machine_anchor(dbfile, before, chunk_start, n_animals, chunk_rows=None):
    """chunk_rows: read_rows of the chunk when the caller has them (classify_scenes)."""
    hi = before - 1
    good = 0           # consecutive frames (going back) that qualify, ending at `end`
    end = None
    later = None       # positions by local identity of the frame after the one looked at
    while hi >= chunk_start:
        lo = max(chunk_start, hi - SEARCH_STEP + 1)
        rows = chunk_rows if chunk_rows is not None else read_rows(dbfile, lo, hi)
        for fn in range(hi, lo - 1, -1):
            r = rows.get(fn)
            pos = {lid: (x, y) for x, y, lid in r} if r and len(r) == n_animals else None
            if pos is not None and (0 in pos or len(pos) != n_animals):
                pos = None
            if pos is not None and later is not None and good and all(
                    np.hypot(x - later[lid][0], y - later[lid][1]) < GATE_PX for lid, (x, y) in pos.items()):
                good += 1
            elif pos is not None:
                good, end = 1, (fn, r)
            else:
                good, end = 0, None
            later = pos
            if good >= STABLE_FRAMES:
                return end
        hi = lo - 1
    return None, None


def _box_positions(boxes, rows):
    """{label: (x, y, detected)} of a saved frame: the row inside each labelled box nearest its centre, else the
    box centre (a fly the person drew where nothing was detected)."""
    out, used = {}, set()
    pairs = []
    for i, b in enumerate(boxes):
        if not str(b["label"]).isdigit():
            continue
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
        for j, (x, y, _) in enumerate(rows):
            if b["x0"] <= x <= b["x1"] and b["y0"] <= y <= b["y1"]:
                pairs.append(((x - cx) ** 2 + (y - cy) ** 2, i, j))
    pairs.sort()
    row_of = {}
    for _, i, j in pairs:
        if i not in row_of and j not in used:
            row_of[i] = j
            used.add(j)
    for i, b in enumerate(boxes):
        if not str(b["label"]).isdigit():
            continue
        if i in row_of:
            x, y, _ = rows[row_of[i]]
            out[int(b["label"])] = (x, y, True)
        else:
            out[int(b["label"])] = ((b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2, False)
    return out


class Run:
    """The tracker from one anchor, extended frame by frame; results kept per frame."""

    def __init__(self, dbfile, anchor, kind, start_positions, notches, n_animals):
        self.dbfile, self.anchor, self.kind, self.notches = dbfile, anchor, kind, notches
        self.labels = list(range(1, n_animals + 1))
        # label -> [x, y, last frame detected (None: never in this run), detected in the last frame]
        self.last = {lab: [x, y, anchor, det] for lab, (x, y, det) in start_positions.items()}
        self.upto = anchor
        self.results = {}

    def extend(self, frame_number, rows=None):
        """rows: read_rows holding the frames after self.upto, when the caller has them."""
        if frame_number <= self.upto:
            return
        if rows is None:
            rows = read_rows(self.dbfile, self.upto + 1, frame_number)
        for fn in range(self.upto + 1, frame_number + 1):
            r = rows.get(fn)
            if r is not None:
                self.results[fn] = self.step(fn, r)
        self.upto = frame_number

    def step(self, fn, rows):
        P = np.array([(x, y) for x, y, _ in rows], dtype=float).reshape(-1, 2)
        L = [lab for lab in self.labels if lab in self.last]
        assigned, info = {}, {}
        if len(P) and L:
            Q = np.array([self.last[lab][:2] for lab in L], dtype=float)
            C = np.hypot(P[:, None, 0] - Q[None, :, 0], P[:, None, 1] - Q[None, :, 1])
            ri, ci = linear_sum_assignment(C)
            others = C.copy()
            others[ri, ci] = np.inf
            second = others.min(axis=1)     # the next fly's distance, per detection
            for i, j in zip(ri.tolist(), ci.tolist()):
                if C[i, j] < GATE_PX:
                    assigned[i] = L[j]
                    info[i] = "ambiguous" if second[i] - C[i, j] < AMBIGUOUS_PX else None
        taken = set(assigned.values())
        # a detection at a notch nobody took: the fly lost at that notch nearest it
        for i, (x, y, _) in enumerate(rows):
            if i in assigned or not _near_notch(self.notches, x, y):
                continue
            lost = [(np.hypot(x - self.last[lab][0], y - self.last[lab][1]), lab) for lab in L
                    if lab not in taken and _near_notch(self.notches, *self.last[lab][:2])]
            lost = [t for t in lost if t[0] < NOTCH_PX]
            if lost:
                lab = min(lost)[1]
                assigned[i] = lab
                taken.add(lab)
                info[i] = "notch"
        # elsewhere: idtrackerai's local identity when that fly is not placed
        for i, (_, _, lid) in enumerate(rows):
            if i not in assigned and lid and lid not in taken:
                assigned[i] = lid
                taken.add(lid)
                info[i] = "new"
        out_rows = []
        for i, (x, y, lid) in enumerate(rows):
            lab = assigned.get(i)
            if lab is None:
                out_rows.append((None, "no fly of the anchor nearby: not labelled", "low"))
                continue
            self.last[lab] = [x, y, fn, True]
            kind = info.get(i)
            if kind == "notch":
                out_rows.append((lab, f"at a notch: fly {lab}, lost there", "medium"))
            elif kind == "new":
                out_rows.append((lab, f"no fly of the anchor nearby: idtrackerai's {lid}", "medium"))
            elif lid != lab:
                out_rows.append((lab, f"continuity: idtrackerai says {lid or 'no identity'}", "medium"))
            elif kind == "ambiguous":
                out_rows.append((lab, "another fly was almost as close", "medium"))
            else:
                out_rows.append((lab, "", None))
        ghosts, lost = [], []
        for lab in self.labels:
            if lab in taken:
                continue
            if lab not in self.last:
                lost.append(f"fly {lab} not placed: not in the anchor frame")
                continue
            x, y, seen, _ = self.last[lab]
            self.last[lab][3] = False
            if _near_notch(self.notches, x, y):
                ghosts.append((lab, x, y, seen))
            else:
                lost.append(f"fly {lab} not detected since frame {seen} (last at {x:.0f}, {y:.0f}): merged with a neighbour?")
        return {"rows": out_rows, "ghosts": ghosts, "flags": lost}


def _get_run(key, make):
    with _lock:
        run = _runs.get(key)
        if run is None:
            run = make()
            if run is None:
                return None
            _runs[key] = run
            while len(_runs) > MAX_RUNS:
                _runs.popitem(last=False)
        else:
            _runs.move_to_end(key)
        return run


def track(dbfile, frame_number, chunksize, n_animals, scenes, saved_before, translate=None):
    """{"rows": [(label or None, note, confidence)] in the frame's row order, "ghosts": [(label, x, y, frame last
    seen)], "flags": [text], "anchor": frame, "anchor_kind": "saved" | "tracking"} or None (no anchor in the chunk).

    saved_before: (frame, boxes, updated) of the last frame saved by a person in [chunk start, frame_number), or None.
    translate(frame, rows) -> [label per row]: the person's labels of an idtrackerai anchor's flies (its local
    identities are not the person's when they relabelled the chunk: the run would carry idtrackerai's into the new
    fragments after a crossing, which look like swaps), or None (idtrackerai's local identities).
    """
    frame_number, chunksize = int(frame_number), int(chunksize)
    chunk_start = frame_number // chunksize * chunksize
    notches = notch_centres(dbfile)
    start = stretch_start(scenes, frame_number)
    m_fn, m_rows = machine_anchor(dbfile, min(start, frame_number), chunk_start, n_animals)
    if saved_before is not None and (m_fn is None or saved_before[0] >= m_fn):
        s_fn, boxes, updated = saved_before
        key = (dbfile, "saved", s_fn, updated)

        def make():
            rows = read_rows(dbfile, s_fn, s_fn).get(s_fn, [])
            return Run(dbfile, s_fn, "saved", _box_positions(boxes, rows), notches, n_animals)
    elif m_fn is not None:
        labels = list(translate(m_fn, m_rows)) if translate is not None else [lid for _, _, lid in m_rows]
        key = (dbfile, "tracking", m_fn, tuple(labels))

        def make():
            return Run(dbfile, m_fn, "tracking", {lab: (x, y, True) for (x, y, _), lab in zip(m_rows, labels)},
                       notches, n_animals)
    else:
        return None
    run = _get_run(key, make)
    with _lock:
        run.extend(frame_number)
        result = run.results.get(frame_number)
    if result is None:
        return None
    return {**result, "anchor": run.anchor, "anchor_kind": run.kind}


# ---- easy scenes: what the continuity proposal resolves on its own -------------------------------------------------

CLASSIFY_VERSION = 1


def _stretches(scenes):
    """Overlapping / adjacent scenes merged: [(first frame, last frame, [scenes])], in order."""
    out = []
    for sc in sorted(scenes, key=lambda s: s["start"]):
        end = sc["start"] + sc["length"] - 1
        if out and sc["start"] <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], end)
            out[-1][2].append(sc)
        else:
            out.append([sc["start"], end, [sc]])
    return out


def classify_scenes(dbfile, scenes, chunksize, n_animals, progress=None):
    """{scene start: {"easy": bool, "why": text}} of how the continuity proposal from idtrackerai's anchor
    (machine_anchor before the merged scenes) goes through each scene. Easy: in every frame each detection gets the
    fly last seen nearest it, agreeing with idtrackerai's local identity where it has one, and the only flies missing
    are kept at a notch; accepting the proposal (Enter) should do. Not easy: no anchor in the chunk, the scenes cross
    a chunk boundary, a fly lost away from the notches, a detection nobody claimed, two flies almost as close, or
    continuity giving an identified detection another fly than idtrackerai (on FlyHostel4_12X_2026-11-15, scenes a
    person saved: 96 % of the frames of easy scenes entirely right, ghosts included; of those with such relabels 47 %).
    ~6 min for its 5820 scenes. progress(fraction) is called after each stretch."""
    notches = notch_centres(dbfile)
    stretches = _stretches(scenes)
    out = {}
    chunk, chunk_rows = None, None    # the rows of one chunk at a time: one query instead of several per scene
    for k, (first, last, members) in enumerate(stretches):
        chunk_start = first // chunksize * chunksize
        why = None
        if last // chunksize != first // chunksize:
            why = "crosses a chunk boundary"
        else:
            if chunk != chunk_start:
                chunk, chunk_rows = chunk_start, read_rows(dbfile, chunk_start, chunk_start + chunksize - 1)
            m_fn, m_rows = _machine_anchor(dbfile, first, chunk_start, n_animals, chunk_rows)
            if m_fn is None:
                why = "no frame before it in the chunk with every fly identified"
        if why is None:
            run = Run(dbfile, m_fn, "tracking", {lid: (x, y, True) for x, y, lid in m_rows}, notches, n_animals)
            run.extend(last, chunk_rows)
        for sc in members:
            if why is not None:
                out[sc["start"]] = {"easy": False, "why": why}
                continue
            relabels = notch = 0
            bad = None
            for fn in range(sc["start"], sc["start"] + sc["length"]):
                res = run.results.get(fn)
                if res is None:
                    bad = bad or f"frame {fn} has no detection"
                    continue
                if res["flags"]:
                    bad = bad or f"frame {fn}: {res['flags'][0]}"
                notch += len(res["ghosts"])
                for lab, note, confidence in res["rows"]:
                    if lab is None:
                        bad = bad or f"frame {fn}: a detection no fly claimed"
                    elif note.startswith("at a notch"):
                        notch += 1
                    elif note.startswith("continuity: idtrackerai says no identity"):
                        pass
                    elif note.startswith("continuity:"):
                        relabels += 1
                    elif confidence is not None:
                        bad = bad or f"frame {fn}: {note}"
            if bad is None and relabels:
                bad = f"{relabels} identified detection(s) given another fly than idtrackerai's"
            if bad is not None:
                out[sc["start"]] = {"easy": False, "why": bad}
            else:
                out[sc["start"]] = {"easy": True, "why": f"{notch} fly-frame(s) kept at a notch, the rest as idtrackerai"}
        if progress is not None:
            progress((k + 1) / len(stretches))
    return out
