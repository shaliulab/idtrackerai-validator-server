import json, sys, random
from evallib import Exp, C, score
import methods as M
from repo_v2_eval import RV2
out = []
for flat in ["FlyHostel4_12X_2026-11-15_17-00-00", "FlyHostel1_12X_2026-11-11_17-00-00"]:
    e = Exp(flat)
    fr = e.saved[::3]
    for f in fr:
        if e.rows(f) is None: continue
        g = e.gt(f); res = {}
        for name, cls in (("v1", C.Run), ("v2", RV2)):
            run, kind = M.make_run(e, f, f, run_cls=cls)
            if run is None: break
            run.extend(f, e.chunk_rows(f // e.cs)); res[name] = run.results[f]
        if len(res) < 2: continue
        e1 = score(g, *M.result_to_pred(res["v1"])); e2 = score(g, *M.result_to_pred(res["v2"]))
        if e1 and not e2:
            notes2 = [n for _, n, _ in res["v2"]["rows"]]
            out.append(dict(exp=flat, f=f, e1=e1, anchor=run.anchor, kinds=sorted({k for k, _ in e1}),
                            leftover=any("reappeared" in n for n in notes2)))
print(len(out))
json.dump(out, open("examples.json", "w"))
import collections
print(collections.Counter(tuple(o["kinds"]) for o in out).most_common(8))
print([o for o in out if o["leftover"]][:3])
print([o for o in out if o["kinds"] == ["wrong_label"]][:5])
