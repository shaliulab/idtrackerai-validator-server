"""Step protocol: every saved frame f, proposals as the server makes them when the person has saved the frames < f.
Writes one row per frame: method, errors by kind, anchor kind/distance, scene reason."""
import sys, json, time
import pandas as pd
from evallib import Exp, score, S
from methods import make_run, result_to_pred, tracking_pred

def main(flat):
    exp = Exp(flat)
    out = []
    t0 = time.time()
    for f in exp.saved:
        rows = exp.rows(f)
        if rows is None:
            continue
        g = exp.gt(f)
        base = dict(exp=flat, f=f, chunk=f // exp.cs, n_rows=len(rows), n_gt_rows=sum(x is not None for x in g[0]),
                    n_gt_ghosts=len(g[1]), n_tags=g[2], author_frag="fragments" in exp.author[f], reason=exp.scene_reason(f))
        pr, pg = tracking_pred(exp, f)
        e = score(g, pr, pg)
        out.append({**base, "method": "tracking", "errors": json.dumps(e), "n_err": len(e), "anchor": None, "dist": None})
        run, kind = make_run(exp, f, f)
        if run is None:
            out.append({**base, "method": "continuity", "errors": json.dumps(e), "n_err": len(e), "anchor": "none", "dist": None})
            continue
        run.extend(f, exp.chunk_rows(f // exp.cs))
        res = run.results.get(f)
        pr, pg = result_to_pred(res)
        e = score(g, pr, pg)
        out.append({**base, "method": "continuity", "errors": json.dumps(e), "n_err": len(e), "anchor": kind, "dist": f - run.anchor,
                    "notes": json.dumps([n for _, n, _ in res["rows"]]), "flags": len(res["flags"])})
    pd.DataFrame(out).to_parquet(f"{S}/work/step_{flat}.parquet")
    print(flat, len(exp.saved), f"{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    for f in sys.argv[1:]:
        main(f)
