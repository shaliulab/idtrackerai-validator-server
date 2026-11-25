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
import bisect
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
MAX_RUNS = 8          # runs kept in memory

_lock = threading.Lock()
_runs = OrderedDict()   # (dbfile, anchor key) -> Run
_notches = {}           # dbfile -> [(cx, cy)]
_anchors = OrderedDict()   # (dbfile, mtime, chunk_start, chunksize, n_animals) -> stable_ends
MAX_ANCHOR_CHUNKS = 200
_anchor_lock = threading.Lock()


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


def read_rows(dbfile, first, last, fragments=False):
    """{frame: [(x, y, local identity or 0)]} of ROI_0 + IDENTITY, in in_frame_index order (as _machine_rows);
    fragments: (x, y, local identity or 0, fragment or -1, area) instead."""
    out = {}
    with sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True) as con:
        for fn, x, y, lid, frag, area in con.execute(
                "SELECT r.frame_number, r.x, r.y, i.local_identity, r.fragment, r.area FROM ROI_0 r LEFT JOIN IDENTITY i "
                "ON i.frame_number = r.frame_number AND i.in_frame_index = r.in_frame_index "
                "WHERE r.frame_number BETWEEN ? AND ? ORDER BY r.frame_number, r.in_frame_index", (first, last)):
            if fragments:
                out.setdefault(fn, []).append((float(x), float(y), int(lid or 0), -1 if frag is None else int(frag),
                                               float(area or 0)))
            else:
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


def stable_ends(dbfile, chunk_start, chunksize, n_animals, chunk_rows=None):
    """Sorted frames of the chunk ending STABLE_FRAMES consecutive frames whose rows are exactly n_animals distinct
    non-zero local identities, each within GATE_PX of where it was the frame before: the possible idtrackerai anchors.
    One read of the chunk (~0.3 s), cached per database mtime; chunk_rows: read_rows of the chunk if at hand."""
    try:
        mtime = os.path.getmtime(dbfile)
    except OSError:
        mtime = None
    key = (dbfile, mtime, chunk_start, chunksize, n_animals)
    with _anchor_lock:
        hit = _anchors.get(key)
        if hit is not None:
            _anchors.move_to_end(key)
            return hit
    rows = chunk_rows if chunk_rows is not None else read_rows(dbfile, chunk_start, chunk_start + chunksize - 1)
    ends, run, before = [], 0, None
    for fn in range(chunk_start, chunk_start + chunksize):
        r = rows.get(fn)
        pos = {lid: (x, y) for x, y, lid in r} if r and len(r) == n_animals else None
        if pos is not None and (0 in pos or len(pos) != n_animals):
            pos = None
        if pos is None:
            run = 0
        elif run and all(np.hypot(x - before[lid][0], y - before[lid][1]) < GATE_PX for lid, (x, y) in pos.items()):
            run += 1
        else:
            run = 1
        before = pos
        if run >= STABLE_FRAMES:
            ends.append(fn)
    with _anchor_lock:
        _anchors[key] = ends
        while len(_anchors) > MAX_ANCHOR_CHUNKS:
            _anchors.popitem(last=False)
    return ends


def machine_anchor(dbfile, before, chunk_start, chunksize, n_animals, chunk_rows=None):
    """(frame, rows) of the last frame in [chunk_start, before) ending STABLE_FRAMES consecutive frames with every
    fly identified and none jumping (stable_ends), or (None, None)."""
    ends = stable_ends(dbfile, chunk_start, chunksize, n_animals, chunk_rows)
    i = bisect.bisect_left(ends, before) - 1
    if i < 0:
        return None, None
    fn = ends[i]
    rows = chunk_rows.get(fn) if chunk_rows is not None else read_rows(dbfile, fn, fn).get(fn)
    return fn, rows


def _box_positions(boxes, rows):
    """{label: (x, y, detected)} of a saved frame: the row inside each labelled box nearest its centre, else the
    box centre (a fly the person drew where nothing was detected)."""
    out, used = {}, set()
    pairs = []
    for i, b in enumerate(boxes):
        if not str(b["label"]).isdigit():
            continue
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
        for j, r in enumerate(rows):
            x, y = r[0], r[1]
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
            x, y = rows[row_of[i]][:2]
            out[int(b["label"])] = (x, y, True)
        else:
            out[int(b["label"])] = ((b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2, False)
    return out


def _box_fragments(boxes, rows):
    """{label: fragment} of a saved frame's labelled boxes holding a row (rows with fragments, read_rows)"""
    out, used, pairs = {}, set(), []
    for i, b in enumerate(boxes):
        if not str(b["label"]).isdigit():
            continue
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
        for j, r in enumerate(rows):
            if b["x0"] <= r[0] <= b["x1"] and b["y0"] <= r[1] <= b["y1"]:
                pairs.append(((r[0] - cx) ** 2 + (r[1] - cy) ** 2, i, j))
    pairs.sort()
    done = set()
    for _, i, j in pairs:
        if i not in done and j not in used:
            done.add(i)
            used.add(j)
            out[int(boxes[i]["label"])] = rows[j][3]
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


class RunV2(Run):
    """Version 2 of the tracker (`?proposal=continuity2`), chosen on the frames saved in 11 validated experiments (5
    of 12 flies; idtrackerai_validator_server/scene_scorer.py has the evaluation): in the step a person takes
    (proposal from the frame saved before) it needs 40 % fewer edits on 12X than Run (frames entirely right 92.5 →
    96.2 %), and a scene proposed from the frame before it is entirely right in 76 % of them instead of 61 %.
    Changes, each measured on its own:
    - every fly not detected is kept at its last position, not only at a notch (on 12X a third of the flies people
      drew where nothing was detected were away from the notches, all of them in FlyHostel1_12X_2026-11-11:
      touching flies left unsegmented; 12X edits −31 %);
    - gate `GATE` 100 px instead of 60 (fast flies);
    - after the gated match, the detections and flies left are paired without gate up to `LEFTOVER_PX` +
      `LEFTOVER_PER_FRAME` px per frame the fly was not seen (a fly reappearing away from where it was lost; note
      "reappeared", medium), before idtrackerai's local identity is used;
    - a detection continuing the idtrackerai fragment the fly had when last detected costs `FRAG_BONUS` px less (at
      crossings without identity the nearest fly is wrong; when the person's fragment continues and disagrees with
      the nearest fly, the fragment was right 129 times out of 138).
    Rows need fragments (read_rows(..., fragments=True)); 3-tuples work without the fragment bonus."""
    GATE = 100
    LEFTOVER_PX = 200.0
    LEFTOVER_PER_FRAME = 10.0
    FRAG_BONUS = 30.0

    def __init__(self, dbfile, anchor, kind, start_positions, notches, n_animals, start_frags=None):
        super().__init__(dbfile, anchor, kind, start_positions, notches, n_animals)
        self.frag = dict(start_frags or {})

    def extend(self, frame_number, rows=None):
        if frame_number <= self.upto:
            return
        if rows is None:
            rows = read_rows(self.dbfile, self.upto + 1, frame_number, fragments=True)
        for fn in range(self.upto + 1, frame_number + 1):
            r = rows.get(fn)
            if r is not None:
                self.results[fn] = self.step(fn, r)
        self.upto = frame_number

    def step(self, fn, rows):
        P = np.array([(r[0], r[1]) for r in rows], dtype=float).reshape(-1, 2)
        frags = [r[3] if len(r) > 3 else None for r in rows]
        L = [lab for lab in self.labels if lab in self.last]
        assigned, info = {}, {}
        if len(P) and L:
            Q = np.array([self.last[lab][:2] for lab in L], dtype=float)
            D = np.hypot(P[:, None, 0] - Q[None, :, 0], P[:, None, 1] - Q[None, :, 1])
            cost = D.copy()
            if self.FRAG_BONUS:
                had = [self.frag.get(lab) if self.last[lab][3] else None for lab in L]
                for i, f in enumerate(frags):
                    if f is None or f < 0:
                        continue
                    for j, g in enumerate(had):
                        if g == f:
                            cost[i, j] -= self.FRAG_BONUS
            ri, ci = linear_sum_assignment(cost)
            others = D.copy()
            others[ri, ci] = np.inf
            second = others.min(axis=1)
            for i, j in zip(ri.tolist(), ci.tolist()):
                if D[i, j] < self.GATE:
                    assigned[i] = L[j]
                    info[i] = "ambiguous" if second[i] - D[i, j] < AMBIGUOUS_PX else None
        taken = set(assigned.values())
        for i, r in enumerate(rows):
            if i in assigned or not _near_notch(self.notches, r[0], r[1]):
                continue
            lost = [(np.hypot(r[0] - self.last[lab][0], r[1] - self.last[lab][1]), lab) for lab in L
                    if lab not in taken and _near_notch(self.notches, *self.last[lab][:2])]
            lost = [t for t in lost if t[0] < NOTCH_PX]
            if lost:
                lab = min(lost)[1]
                assigned[i] = lab
                taken.add(lab)
                info[i] = "notch"
        left_rows = [i for i in range(len(rows)) if i not in assigned]
        left_flies = [lab for lab in L if lab not in taken]
        if left_rows and left_flies:
            Q = np.array([self.last[lab][:2] for lab in left_flies], dtype=float)
            Pl = P[left_rows]
            D = np.hypot(Pl[:, None, 0] - Q[None, :, 0], Pl[:, None, 1] - Q[None, :, 1])
            limit = [self.LEFTOVER_PX + self.LEFTOVER_PER_FRAME * (fn - (self.last[lab][2] or fn)) for lab in left_flies]
            ri, ci = linear_sum_assignment(D)
            for a, b in zip(ri.tolist(), ci.tolist()):
                if D[a, b] <= limit[b]:
                    assigned[left_rows[a]] = left_flies[b]
                    taken.add(left_flies[b])
                    info[left_rows[a]] = "leftover"
        for i, r in enumerate(rows):
            lid = r[2]
            if i not in assigned and lid and lid not in taken:
                assigned[i] = lid
                taken.add(lid)
                info[i] = "new"
        out_rows = []
        for i, r in enumerate(rows):
            x, y, lid = r[0], r[1], r[2]
            lab = assigned.get(i)
            if lab is None:
                out_rows.append((None, "no fly of the anchor nearby: not labelled", "low"))
                continue
            self.last[lab] = [x, y, fn, True]
            if frags[i] is not None:
                self.frag[lab] = frags[i]
            kind = info.get(i)
            if kind == "notch":
                out_rows.append((lab, f"at a notch: fly {lab}, lost there", "medium"))
            elif kind == "leftover":
                out_rows.append((lab, f"fly {lab} reappeared away from where it was lost", "medium"))
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
            ghosts.append((lab, x, y, seen))
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


def anchor_run(dbfile, frame_number, chunksize, n_animals, scenes, saved_before, translate=None, version=1):
    """(cache key, make) of the Run the proposal of frame_number starts from (see track), or (None, None)."""
    frame_number, chunksize = int(frame_number), int(chunksize)
    chunk_start = frame_number // chunksize * chunksize
    notches = notch_centres(dbfile)
    start = stretch_start(scenes, frame_number)
    m_fn, m_rows = machine_anchor(dbfile, min(start, frame_number), chunk_start, chunksize, n_animals)
    v2 = version == 2
    if saved_before is not None and (m_fn is None or saved_before[0] >= m_fn):
        s_fn, boxes, updated = saved_before
        key = (dbfile, version, "saved", s_fn, updated)

        def make():
            if v2:
                rows = read_rows(dbfile, s_fn, s_fn, fragments=True).get(s_fn, [])
                return RunV2(dbfile, s_fn, "saved", _box_positions(boxes, rows), notches, n_animals,
                             _box_fragments(boxes, rows))
            rows = read_rows(dbfile, s_fn, s_fn).get(s_fn, [])
            return Run(dbfile, s_fn, "saved", _box_positions(boxes, rows), notches, n_animals)
    elif m_fn is not None:
        labels = list(translate(m_fn, m_rows)) if translate is not None else [lid for _, _, lid in m_rows]
        key = (dbfile, version, "tracking", m_fn, tuple(labels))

        def make():
            start_positions = {lab: (x, y, True) for (x, y, _), lab in zip(m_rows, labels)}
            if v2:
                rows = read_rows(dbfile, m_fn, m_fn, fragments=True).get(m_fn, [])
                frags = {lab: r[3] for r, lab in zip(rows, labels)} if len(rows) == len(labels) else {}
                return RunV2(dbfile, m_fn, "tracking", start_positions, notches, n_animals, frags)
            return Run(dbfile, m_fn, "tracking", start_positions, notches, n_animals)
    else:
        return None, None
    return key, make


def track(dbfile, frame_number, chunksize, n_animals, scenes, saved_before, translate=None, version=1):
    """{"rows": [(label or None, note, confidence)] in the frame's row order, "ghosts": [(label, x, y, frame last
    seen)], "flags": [text], "anchor": frame, "anchor_kind": "saved" | "tracking"} or None (no anchor in the chunk).

    saved_before: (frame, boxes, updated) of the last frame saved by a person in [chunk start, frame_number), or None.
    translate(frame, rows) -> [label per row]: the person's labels of an idtrackerai anchor's flies (its local
    identities are not the person's when they relabelled the chunk: the run would carry idtrackerai's into the new
    fragments after a crossing, which look like swaps), or None (idtrackerai's local identities).
    version: 1 = Run (the default proposal), 2 = RunV2 (`?proposal=continuity2`).
    """
    key, make = anchor_run(dbfile, frame_number, chunksize, n_animals, scenes, saved_before, translate, version)
    if key is None:
        return None
    run = _get_run(key, make)
    with _lock:
        run.extend(int(frame_number))
        result = run.results.get(int(frame_number))
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
            m_fn, m_rows = machine_anchor(dbfile, first, chunk_start, chunksize, n_animals, chunk_rows)
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
