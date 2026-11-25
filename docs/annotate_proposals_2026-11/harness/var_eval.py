"""Step protocol (anchor = person's frames before) + cold simulation (edits when fixing each wrong frame) per variant.
python var_eval.py exp variant_json"""
import sys, json, time
import numpy as np, pandas as pd
from evallib import Exp, score, S, C
import methods as M
from cold_eval import gt_run

def fly_errors(gt, pred_rows, pred_ghosts):
    return score(gt, pred_rows, pred_ghosts)

def evaluate(flat, name, cls):
    exp = Exp(flat)
    sv = np.array(exp.saved)
    step = []
    for f in exp.saved:
        if exp.rows(f) is None: continue
        run, kind = M.make_run(exp, f, f, run_cls=cls)
        if run is None:
            pr, pg = M.tracking_pred(exp, f)
        else:
            run.extend(f, exp.chunk_rows(f // exp.cs)); pr, pg = M.result_to_pred(run.results[f])
        step.append(len(score(exp.gt(f), pr, pg)))
    sims, allok = [], []
    for a, b, members in exp.stretches:
        k = np.searchsorted(sv, b + 1) - np.searchsorted(sv, a)
        if k != b - a + 1 or a // exp.cs != b // exp.cs: continue
        run, _ = M.make_run(exp, a, a, run_cls=cls)
        if run is not None: run.extend(a - 1, exp.chunk_rows(a // exp.cs))
        edits, ok, fixed = 0, True, False
        for f in range(a, b + 1):
            r = exp.rows(f)
            if r is None: continue
            pr, pg = M.tracking_pred(exp, f) if run is None else M.result_to_pred(run.step(f, r))
            e = score(exp.gt(f), pr, pg)
            if e:
                edits += len(e); ok = False; run = gt_run(exp, f, cls)
        sims.append(edits); allok.append(ok)
    step = np.array(step)
    return dict(exp=flat, variant=name, frames=len(step), step_ok=100 * (step == 0).mean(), step_edits=step.sum(),
                stretches=len(sims), cold_allok=100 * np.mean(allok), cold_edits=int(np.sum(sims)))

if __name__ == "__main__":
    flat, spec = sys.argv[1], json.loads(sys.argv[2])
    name = spec.pop("name")
    base = M.Run2 if not spec.pop("orig", False) else C.Run
    cls = M.make_variant(name, **spec) if base is M.Run2 else C.Run
    t = time.time()
    r = evaluate(flat, name, cls)
    with open(f"{S}/work/variants.jsonl", "a") as h:
        h.write(json.dumps(r, default=float) + "\n")
    print(r, f"{time.time()-t:.0f}s", flush=True)
