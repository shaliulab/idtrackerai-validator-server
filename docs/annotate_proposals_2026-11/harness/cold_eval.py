"""Cold protocol: each stretch (merged scenes) whose frames were all saved, proposed without the person's labels
inside it. Per frame: errors of each method, and whether forward / backward runs agree."""
import bisect
import json
import sys
import time

import numpy as np
import pandas as pd

import methods as M
from evallib import C, Exp, S, score


def run_over(run, frames, rows_of):
    """step the run over frames in the given order (forward or backward); {frame: result}"""
    out = {}
    for fn in frames:
        r = rows_of(fn)
        if r is not None:
            out[fn] = run.step(fn, r)
    return out


def labels_of(res):
    if res is None:
        return None
    return (tuple(lab for lab, _, _ in res["rows"]), tuple(sorted(g[0] for g in res["ghosts"])))


def after_anchor(exp, last, chunk):
    """first frame after `last` starting STABLE_FRAMES stable frames, in the chunk (rows extracted)"""
    lo = chunk * exp.cs
    ends = C.stable_ends(exp.dbfile, lo, exp.cs, exp.n, exp.chunk_rows(chunk))
    i = bisect.bisect_left(ends, last + C.STABLE_FRAMES)
    while i < len(ends):
        start = ends[i] - C.STABLE_FRAMES + 1
        if start > last:
            return start
        i += 1
    return None


def machine_run(exp, fn, cls):
    rows = exp.rows(fn)
    run = cls(exp.dbfile, fn, "tracking", {lid: (x, y, True) for x, y, lid in rows}, exp.notches, exp.n)
    M.set_frags(run, exp, fn, [lid for _, _, lid in rows])
    return run


def gt_run(exp, fn, cls):
    run = cls(exp.dbfile, fn, "saved", C._box_positions(exp.boxes[fn], exp.rows(fn) or []), exp.notches, exp.n)
    M.set_frags(run, exp, fn, exp.gt(fn)[0])
    return run


def main(flat):
    exp = Exp(flat)
    sv = np.array(exp.saved)
    out = []
    t0 = time.time()
    for a, b, members in exp.stretches:
        k = np.searchsorted(sv, b + 1) - np.searchsorted(sv, a)
        if k != b - a + 1 or a // exp.cs != b // exp.cs:
            continue
        chunk = a // exp.cs
        frames = list(range(a, b + 1))
        rows_of = exp.rows
        gts = {f: exp.gt(f) for f in frames}
        res = {}
        # current: human-before anchor (saved frames before the stretch) or machine anchor in the person's labels
        for name, cls in (("fwd_cur", C.Run), ("fwd_gh", M.GhostRun)):
            run, kind = M.make_run(exp, a, a, run_cls=cls)
            if run is None:
                res[name] = {}
                continue
            run.extend(a - 1, exp.chunk_rows(chunk))
            res[name] = run_over(run, frames, rows_of)
            res[name + "_kind"] = kind
            res[name + "_dist"] = a - run.anchor
        # machine-only forward (no person's labels at all)
        m_fn, _ = C.machine_anchor(exp.dbfile, a, chunk * exp.cs, exp.cs, exp.n, exp.chunk_rows(chunk))
        for name, cls in (("fwd_m", M.GhostRun), ("fwd_mcur", C.Run)):
            if m_fn is None:
                res[name] = {}
                continue
            run = machine_run(exp, m_fn, cls)
            run.extend(a - 1, exp.chunk_rows(chunk))
            res[name] = run_over(run, frames, rows_of)
        # backward from the machine anchor after the stretch
        e_fn = after_anchor(exp, b, chunk)
        if e_fn is not None:
            run = machine_run(exp, e_fn, M.GhostRun)
            run_over(run, range(e_fn - 1, b, -1), rows_of)
            res["bwd_m"] = run_over(run, frames[::-1], rows_of)
        else:
            res["bwd_m"] = {}
        # keyframes: the person's first and last frame of the stretch, the middle from both sides
        if len(frames) >= 3:
            run = gt_run(exp, a, M.GhostRun)
            res["key_fwd"] = run_over(run, frames[1:-1], rows_of)
            run = gt_run(exp, b, M.GhostRun)
            res["key_bwd"] = run_over(run, frames[-2:0:-1], rows_of)
        # edits when the person fixes each wrong frame and the proposal restarts from it
        for name, cls in (("sim_cur", C.Run), ("sim_gh", M.GhostRun)):
            run, _ = M.make_run(exp, a, a, run_cls=cls)
            edits = 0
            if run is not None:
                run.extend(a - 1, exp.chunk_rows(chunk))
            for f in frames:
                r = rows_of(f)
                if r is None:
                    continue
                if run is None:
                    pr, pg = M.tracking_pred(exp, f)
                else:
                    pr, pg = M.result_to_pred(run.step(f, r))
                e = score(gts[f], pr, pg)
                if e:
                    edits += len(e)
                    run = gt_run(exp, f, cls)
            res[name] = edits
        reason = "; ".join(sorted({s["reason"] for s in members}))
        for f in frames:
            g = gts[f]
            rec = dict(exp=flat, a=a, b=b, f=f, i=f - a, length=b - a + 1, reason=reason,
                       n_gt_rows=sum(x is not None for x in g[0]), n_gt_ghosts=len(g[1]),
                       anchor_kind=res.get("fwd_cur_kind"), anchor_dist=res.get("fwd_cur_dist"),
                       sim_cur=res["sim_cur"], sim_gh=res["sim_gh"], e_after=e_fn)
            for name in ("fwd_cur", "fwd_gh", "fwd_m", "fwd_mcur", "bwd_m", "key_fwd", "key_bwd"):
                rr = res.get(name, {}).get(f)
                if rr is None:
                    rec[name] = None
                    continue
                pr, pg = M.result_to_pred(rr)
                rec[name] = json.dumps(score(g, pr, pg))
                rec[name + "_lab"] = json.dumps(labels_of(rr))
                rec[name + "_flags"] = len(rr["flags"])
                rec[name + "_unsure"] = sum(1 for lab, note, conf in rr["rows"] if conf is not None)
            out.append(rec)
    pd.DataFrame(out).to_parquet(f"{S}/work/cold_{flat}.parquet")
    print(flat, len(out), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        main(f)
