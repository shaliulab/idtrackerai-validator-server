"""Offline replay of the Annotate tab's proposals against the frames a person saved."""
import bisect
import csv
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

WT = "/home/vibflysleep/orca/workspaces/idtrackerai-validator-server/cowfish"
sys.path.insert(0, WT)
from idtrackerai_validator_server import continuity as C  # noqa: E402

S = os.path.dirname(os.path.abspath(__file__)) + "/.."
TAGS = {"FMB", "NEW-FRAGMENT", "CROSSING"}
GHOST_OK_PX = 40


class Exp:
    def __init__(self, flat):
        self.flat = flat
        z = np.load(f"{S}/rows/{flat}.npz", allow_pickle=True)
        self.n, self.cs = int(z["n"]), int(z["cs"])
        self.notches = [tuple(p) for p in z["notches"].tolist()]
        self.dbfile = str(z["dbfile"])
        R = z["rows"]
        self.R = R
        fn = R[:, 0].astype(np.int64)
        self.frames, starts = np.unique(fn, return_index=True)
        ends = np.r_[starts[1:], len(fn)]
        self._idx = {int(f): (int(a), int(b)) for f, a, b in zip(self.frames, starts, ends)}
        self.conc = z["conc"]
        self.ppcm = float(z["ppcm"])
        # saved boxes
        con = sqlite3.connect(f"{S}/side/{flat}.db")
        self.author = dict(con.execute("SELECT frame_number, author FROM frames"))
        self.updated = dict(con.execute("SELECT frame_number, updated FROM frames"))
        self.boxes = defaultdict(list)
        for f, x0, y0, x1, y1, lab in con.execute(
                "SELECT b.frame_number, x0, y0, x1, y1, label FROM boxes b JOIN frames USING (frame_number) ORDER BY b.frame_number, b.rowid"):
            self.boxes[f].append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "label": lab or ""})
        self.saved = sorted(self.author)
        self.saved_set = set(self.saved)
        con.close()
        self.scenes = read_scenes(f"{S}/side/{flat}.scene_qc.csv")
        self.stretches = C._stretches(self.scenes)
        self._st_starts = [s[0] for s in self.stretches]
        self._chunk_cache = {}

    # ---- rows
    def rows(self, f):
        """[(x, y, lid)] in in_frame_index order, as continuity.read_rows"""
        ab = self._idx.get(int(f))
        if ab is None:
            return None
        r = self.R[ab[0]:ab[1]]
        return [(float(x), float(y), int(l)) for x, y, l in r[:, [2, 3, 7]]]

    def full(self, f):
        ab = self._idx.get(int(f))
        return None if ab is None else self.R[ab[0]:ab[1]]

    def chunk_rows(self, chunk):
        if chunk not in self._chunk_cache:
            lo = chunk * self.cs
            out = {}
            for f in range(lo, lo + self.cs):
                r = self.rows(f)
                if r is not None:
                    out[f] = r
            self._chunk_cache = {chunk: out}
            self._chunk_cache[chunk] = out
        return self._chunk_cache[chunk]

    def stretch_of(self, f):
        """(first, last) of the merged stretch holding f, or None"""
        i = bisect.bisect_right(self._st_starts, f) - 1
        if i >= 0 and self.stretches[i][0] <= f <= self.stretches[i][1]:
            return self.stretches[i][0], self.stretches[i][1]
        return None

    def scene_reason(self, f):
        i = bisect.bisect_right(self._st_starts, f) - 1
        if i >= 0 and self.stretches[i][0] <= f <= self.stretches[i][1]:
            return "; ".join(sorted({s["reason"] for s in self.stretches[i][2]}))
        return "outside scenes"

    # ---- ground truth
    def gt(self, f):
        """(labels per row: int or None, ghosts [(label, cx, cy, w, h)], n tag boxes)"""
        rows = self.rows(f) or []
        boxes = self.boxes.get(f, [])
        lab_boxes = [b for b in boxes if b["label"].isdigit()]
        pairs = []
        for i, b in enumerate(lab_boxes):
            cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
            for j, (x, y, _) in enumerate(rows):
                if b["x0"] <= x <= b["x1"] and b["y0"] <= y <= b["y1"]:
                    pairs.append(((x - cx) ** 2 + (y - cy) ** 2, i, j))
        pairs.sort()
        row_lab, used_b = [None] * len(rows), set()
        for _, i, j in pairs:
            if i not in used_b and row_lab[j] is None:
                row_lab[j] = int(lab_boxes[i]["label"])
                used_b.add(i)
        ghosts = [(int(b["label"]), (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2, b["x1"] - b["x0"], b["y1"] - b["y0"])
                  for i, b in enumerate(lab_boxes) if i not in used_b]
        return row_lab, ghosts, sum(b["label"] in TAGS for b in boxes)

    def saved_before(self, f, lo):
        i = bisect.bisect_left(self.saved, f) - 1
        if i >= 0 and self.saved[i] >= lo:
            return self.saved[i]
        return None

    def saved_after(self, f, hi):
        i = bisect.bisect_right(self.saved, f)
        if i < len(self.saved) and self.saved[i] <= hi:
            return self.saved[i]
        return None


def read_scenes(path):
    scenes = []
    with open(path, newline="") as h:
        for row in csv.DictReader(h):
            if row.get("pass") == "True":
                continue
            length = int(float(row["length"]))
            if length > 5000:
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
                reasons.append("failed fragments")
            start = int(float(row["scene_start"]))
            if start < 0:
                length += start
                start = 0
            if length > 0:
                scenes.append({"start": start, "length": length, "reason": ", ".join(reasons)})
    scenes.sort(key=lambda s: s["start"])
    return scenes


# ---- scoring ----------------------------------------------------------------------------------------------------

def score(gt, pred_rows, pred_ghosts, rows=None):
    """errors: list of (kind, detail). gt: Exp.gt(f); pred_rows: [label or None]; pred_ghosts: [(label, x, y)]"""
    g_rows, g_ghosts, _ = gt
    errs = []
    for j, (g, p) in enumerate(zip(g_rows, pred_rows)):
        if g is None:
            if p is not None:
                errs.append(("extra_label", j))
        elif p is None:
            errs.append(("unlabelled", j))
        elif p != g:
            errs.append(("wrong_label", j))
    pg = {lab: (x, y) for lab, x, y in pred_ghosts}
    gl = set()
    for lab, cx, cy, w, h in g_ghosts:
        gl.add(lab)
        if lab in pg and np.hypot(pg[lab][0] - cx, pg[lab][1] - cy) <= max(GHOST_OK_PX, 0.5 * max(w, h)):
            continue
        errs.append(("missing_fly" if lab not in pg else "ghost_misplaced", lab))
    for lab in pg:
        if lab not in gl:
            errs.append(("extra_ghost", lab))
    return errs
