"""Per fully saved stretch: features of the forward run (anchor as the server: person's frames before, else
idtrackerai's) and of a backward run from the stable frames after it, and the outcome (stretch entirely right).
python stretch_feats.py VARIANT_JSON exp..."""
import json
import sys
import time

import numpy as np
import pandas as pd

import methods as M
from cold_eval import after_anchor, machine_run, run_over, labels_of
from evallib import C, Exp, S, score

REASONS = ["ids missing", "id changes", "broken", "gap", "between chunks", "failed fragments"]


def feats_of_results(exp, frames, res, prefix):
    f = {}
    notes = [n for fn in frames if fn in res for _, n, _ in res[fn]["rows"]]
    f[prefix + "n_unlab"] = sum(1 for fn in frames if fn in res for lab, _, _ in res[fn]["rows"] if lab is None)
    for key, pat in (("relabel", "continuity: idtrackerai says "), ("noid", "continuity: idtrackerai says no identity"),
                     ("ambig", "another fly"), ("new", "no fly of the anchor nearby: idtrackerai"), ("notch", "at a notch"),
                     ("leftover", "reappeared")):
        f[prefix + key] = sum(1 for n in notes if n.startswith(pat))
    f[prefix + "relabel"] -= f[prefix + "noid"]
    f[prefix + "ghost_ff"] = sum(len(res[fn]["ghosts"]) for fn in frames if fn in res)
    f[prefix + "ghost_max"] = max([len(res[fn]["ghosts"]) for fn in frames if fn in res] or [0])
    f[prefix + "flags"] = sum(len(res[fn]["flags"]) for fn in frames if fn in res)
    return f


def geometry(exp, frames, a):
    rows = [exp.full(fn) for fn in frames]
    rows = [r for r in rows if r is not None]
    n = exp.n
    f = {}
    counts = np.array([len(r) for r in rows]) if rows else np.array([n])
    f["rows_min"] = counts.min() - n
    f["rows_max"] = counts.max() - n
    f["frames_short"] = int((counts < n).sum())
    f["frames_long"] = int((counts > n).sum())
    dmins, areas = [], []
    for r in rows:
        if len(r) >= 2:
            P = r[:, 2:4]
            D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
            D[np.diag_indices(len(P))] = np.inf
            dmins.append(D.min())
        areas.extend(r[:, 5].tolist())
    f["dmin"] = min(dmins) if dmins else 999
    f["dmin_med"] = float(np.median(dmins)) if dmins else 999
    f["frames_close40"] = int(sum(d < 40 for d in dmins))
    med_area = np.median(exp.full(a)[:, 5]) if exp.full(a) is not None else np.median(areas or [1])
    f["area_max_ratio"] = (max(areas) / med_area) if areas and med_area else 1
    f["lid0"] = int(sum((r[:, 7] == 0).sum() for r in rows))
    f["frags"] = len({int(x) for r in rows for x in r[:, 4]})
    return f


def main(spec, flat):
    name = spec.pop("name")
    cls = M.make_variant(name, **spec)
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
        run, kind = M.make_run(exp, a, a, run_cls=cls)
        if run is None:
            continue
        run.extend(a - 1, exp.chunk_rows(chunk))
        res = run_over(run, frames, exp.rows)
        errs = {fn: score(exp.gt(fn), *M.result_to_pred(res[fn])) for fn in frames if fn in res}
        id_errs = {fn: [e for e in v if e[0] in ("wrong_label", "unlabelled")] for fn, v in errs.items()}
        rec = dict(exp=flat, a=a, b=b, length=len(frames), n=exp.n, anchor_kind=kind == "saved", anchor_dist=a - run.anchor,
                   ok=all(not v for v in errs.values()), id_ok=all(not v for v in id_errs.values()),
                   n_err=sum(len(v) for v in errs.values()), frames_ok=sum(not v for v in errs.values()),
                   reasons=";".join(sorted({s["reason"] for s in members})), n_scenes=len(members))
        for r in REASONS:
            rec["r_" + r.replace(" ", "_")] = int(any(r in s["reason"] for s in members))
        rec.update(feats_of_results(exp, frames, res, "f_"))
        rec.update(geometry(exp, frames, a))
        # backward from the stable frames after the stretch (idtrackerai's identities there)
        e_fn = after_anchor(exp, b, chunk)
        if e_fn is not None:
            bw = machine_run(exp, e_fn, cls)
            run_over(bw, range(e_fn - 1, b, -1), exp.rows)
            bres = run_over(bw, frames[::-1], exp.rows)
            dis = [fn for fn in frames if fn in res and labels_of(res[fn]) != labels_of(bres.get(fn))]
            rec["b_avail"] = 1
            rec["b_disagree"] = len(dis)
            rec["b_disagree_end"] = int(b in dis)
            rec["b_gap"] = e_fn - b
            rec.update(feats_of_results(exp, frames, bres, "b_"))
        else:
            rec.update(b_avail=0, b_disagree=-1, b_disagree_end=-1, b_gap=-1)
        out.append(rec)
    pd.DataFrame(out).to_parquet(f"{S}/work/feats_{name}_{flat}.parquet")
    print(flat, len(out), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    for f in sys.argv[2:]:
        main(dict(spec), f)
