"""Proposal methods replayed offline."""
import numpy as np

from evallib import C


def frag_labels(exp, chunk, cutoff):
    """{fragment: [(frame, label)]} of saved frames of the chunk before cutoff"""
    lo = chunk * exp.cs
    out = {}
    for s in exp.saved:
        if s < lo or s >= cutoff:
            continue
        full = exp.full(s)
        if full is None:
            continue
        labs, _, _ = exp.gt(s)
        for r, lab in zip(full, labs):
            if lab is not None:
                out.setdefault(int(r[4]), []).append((s, lab))
    return out


def anchor_labels(exp, m_fn, labels):
    full = exp.full(m_fn)
    out = [None] * len(full)
    taken = set()
    for i, r in enumerate(full):
        given = labels.get(int(r[4]))
        if not given:
            continue
        before = [lab for fn, lab in given if fn < m_fn]
        after = [lab for fn, lab in given if fn >= m_fn]
        label = before[-1] if before else after[0]
        if label not in taken:
            out[i] = label
            taken.add(label)
    for i, r in enumerate(full):
        lid = int(r[7])
        if out[i] is None and lid and lid not in taken:
            out[i] = lid
            taken.add(lid)
    left = [lab for lab in range(1, exp.n + 1) if lab not in taken]
    for i in range(len(full)):
        if out[i] is None and left:
            out[i] = left.pop(0)
    return out


def make_run(exp, f, saved_cutoff, run_cls=None, use_saved=True):
    """The continuity Run the server would use for frame f when the person had saved the frames < saved_cutoff of
    the chunk (step protocol: saved_cutoff = f; cold: the stretch start). Returns (Run, anchor kind) or (None, None)."""
    run_cls = run_cls or C.Run
    chunk = f // exp.cs
    lo = chunk * exp.cs
    chunk_rows = exp.chunk_rows(chunk)
    st = exp.stretch_of(f)
    start = st[0] if st else f
    m_fn, m_rows = C.machine_anchor(exp.dbfile, min(start, f), lo, exp.cs, exp.n, chunk_rows)
    s = exp.saved_before(saved_cutoff, lo) if use_saved else None
    if s is not None and (m_fn is None or s >= m_fn):
        boxes = exp.boxes[s]
        run = run_cls(exp.dbfile, s, "saved", C._box_positions(boxes, exp.rows(s) or []), exp.notches, exp.n)
        set_frags(run, exp, s, exp.gt(s)[0])
        return run, "saved"
    if m_fn is not None:
        labels = anchor_labels(exp, m_fn, frag_labels(exp, chunk, saved_cutoff))
        run = run_cls(exp.dbfile, m_fn, "tracking", {lab: (x, y, True) for (x, y, _), lab in zip(m_rows, labels)},
                      exp.notches, exp.n)
        set_frags(run, exp, m_fn, labels)
        return run, "tracking"
    return None, None


def set_frags(run, exp, fn, labels):
    if not hasattr(run, "frag"):
        return
    run.exp = exp
    full = exp.full(fn)
    for r, lab in zip(full if full is not None else [], labels):
        if lab is not None:
            run.frag[lab] = int(r[4])


def result_to_pred(res):
    rows = [lab for lab, _, _ in res["rows"]]
    ghosts = [(lab, x, y) for lab, x, y, _ in res["ghosts"]]
    return rows, ghosts


def tracking_pred(exp, f):
    return [lid or None for _, _, lid in exp.rows(f)], []


class GhostRun(C.Run):
    """continuity, but every fly not detected is kept at its last position (not only at a notch)."""
    max_unseen = None

    def step(self, fn, rows):
        res = C.Run.step(self, fn, rows)
        taken = {lab for lab, _, _ in res["rows"] if lab is not None}
        have = {g[0] for g in res["ghosts"]}
        flags = []
        for lab in self.labels:
            if lab in taken or lab in have or lab not in self.last:
                continue
            x, y, seen, _ = self.last[lab]
            if self.max_unseen is not None and seen is not None and fn - seen > self.max_unseen:
                flags.append(f"fly {lab} lost")
                continue
            res["ghosts"].append((lab, x, y, seen))
        res["flags"] = [t for t in res["flags"] if "not in the anchor" in t] + flags
        return res


from scipy.optimize import linear_sum_assignment
import numpy as _np


class Run2(C.Run):
    """continuity with options (class attributes, set by make_variant):
    gate: GATE_PX; ghosts_all: keep every lost fly at its last position; leftover: after the gated match, pair the
    detections and flies left (not at a notch) without gate, up to leftover_px (+ leftover_per_frame px per frame
    unseen); velocity: predict each fly's position from its last step (frames detected consecutively)."""
    gate = C.GATE_PX
    ghosts_all = True
    leftover = False
    leftover_px = 200.0
    leftover_per_frame = 10.0
    velocity = False
    lid_bonus = 0.0     # px subtracted from the cost when the detection's idtrackerai lid equals the fly's label
    frag_bonus = 0.0    # px subtracted when the detection continues the fragment the fly had when last detected
    exp = None

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.vel = {}
        self.frag = {}

    def step(self, fn, rows):
        P = _np.array([(x, y) for x, y, _ in rows], dtype=float).reshape(-1, 2)
        L = [lab for lab in self.labels if lab in self.last]
        assigned, info = {}, {}
        def pred(lab):
            x, y, seen, det = self.last[lab]
            if self.velocity and det and lab in self.vel:
                vx, vy = self.vel[lab]
                return x + vx, y + vy
            return x, y
        if len(P) and L:
            Q = _np.array([pred(lab) for lab in L], dtype=float)
            D = _np.hypot(P[:, None, 0] - Q[None, :, 0], P[:, None, 1] - Q[None, :, 1])
            Cm = D.copy()
            if self.lid_bonus:
                lids = _np.array([lid for _, _, lid in rows])
                Cm = Cm - self.lid_bonus * (lids[:, None] == _np.array(L)[None, :])
            F = None
            if self.frag_bonus and self.exp is not None:
                F = self.exp.full(fn)[:, 4].astype(int)
                fl = _np.array([self.frag.get(lab, -1) if self.last[lab][3] else -1 for lab in L])
                Cm = Cm - self.frag_bonus * (F[:, None] == fl[None, :])
            ri, ci = linear_sum_assignment(Cm)
            others = D.copy(); others[ri, ci] = _np.inf
            second = others.min(axis=1)
            for i, j in zip(ri.tolist(), ci.tolist()):
                if D[i, j] < self.gate:
                    assigned[i] = L[j]
                    info[i] = "ambiguous" if second[i] - D[i, j] < C.AMBIGUOUS_PX else None
        taken = set(assigned.values())
        for i, (x, y, _) in enumerate(rows):
            if i in assigned or not C._near_notch(self.notches, x, y):
                continue
            lost = [(_np.hypot(x - self.last[lab][0], y - self.last[lab][1]), lab) for lab in L
                    if lab not in taken and C._near_notch(self.notches, *self.last[lab][:2])]
            lost = [t for t in lost if t[0] < C.NOTCH_PX]
            if lost:
                lab = min(lost)[1]; assigned[i] = lab; taken.add(lab); info[i] = "notch"
        if self.leftover:
            di = [i for i in range(len(rows)) if i not in assigned]
            fl = [lab for lab in L if lab not in taken]
            if di and fl:
                Q = _np.array([self.last[lab][:2] for lab in fl], float)
                Pd = P[di]
                D = _np.hypot(Pd[:, None, 0] - Q[None, :, 0], Pd[:, None, 1] - Q[None, :, 1])
                lim = _np.array([self.leftover_px + self.leftover_per_frame * (fn - (self.last[lab][2] or fn)) for lab in fl])
                ri, ci = linear_sum_assignment(D)
                for a_, b_ in zip(ri.tolist(), ci.tolist()):
                    if D[a_, b_] <= lim[b_]:
                        assigned[di[a_]] = fl[b_]; taken.add(fl[b_]); info[di[a_]] = "leftover"
        for i, (_, _, lid) in enumerate(rows):
            if i not in assigned and lid and lid not in taken:
                assigned[i] = lid; taken.add(lid); info[i] = "new"
        out_rows = []
        for i, (x, y, lid) in enumerate(rows):
            lab = assigned.get(i)
            if lab is None:
                out_rows.append((None, "no fly of the anchor nearby: not labelled", "low"))
                continue
            ox, oy, oseen, odet = self.last[lab] if lab in self.last else (x, y, None, False)
            if odet and oseen == fn - 1:
                self.vel[lab] = (x - ox, y - oy)
            else:
                self.vel.pop(lab, None)
            self.last[lab] = [x, y, fn, True]
            if self.exp is not None:
                self.frag[lab] = int(self.exp.full(fn)[i, 4])
            kind = info.get(i)
            if kind == "notch":
                out_rows.append((lab, f"at a notch: fly {lab}, lost there", "medium"))
            elif kind == "new":
                out_rows.append((lab, f"no fly of the anchor nearby: idtrackerai's {lid}", "medium"))
            elif kind == "leftover":
                out_rows.append((lab, f"fly {lab} reappeared away from where it was lost", "medium"))
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
            self.vel.pop(lab, None)
            if self.ghosts_all or C._near_notch(self.notches, x, y):
                ghosts.append((lab, x, y, seen))
            else:
                lost.append(f"fly {lab} not detected since frame {seen}")
        return {"rows": out_rows, "ghosts": ghosts, "flags": lost}


def make_variant(name, **kw):
    return type(name, (Run2,), kw)
