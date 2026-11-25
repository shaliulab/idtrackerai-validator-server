"""The server's easy-scene classification (continuity.classify_scenes) on the fully saved stretches, rows from the cache."""
import sys, json, numpy as np, pandas as pd
from evallib import Exp, C, S
out = []
for flat in sys.argv[1:]:
    exp = Exp(flat)
    C.read_rows = lambda dbfile, first, last: {f: exp.rows(f) for f in range(first, last + 1) if exp.rows(f) is not None}
    sv = np.array(exp.saved)
    scenes = []
    for a, b, members in exp.stretches:
        k = np.searchsorted(sv, b + 1) - np.searchsorted(sv, a)
        if k == b - a + 1 and a // exp.cs == b // exp.cs:
            scenes += members
    res = C.classify_scenes(exp.dbfile, scenes, exp.cs, exp.n)
    for a, b, members in exp.stretches:
        if members[0]["start"] in res:
            out.append(dict(exp=flat, a=a, easy=all(res[m["start"]]["easy"] for m in members), easy_any=any(res[m["start"]]["easy"] for m in members),
                            why=res[members[0]["start"]]["why"]))
    print(flat, flush=True)
pd.DataFrame(out).to_parquet(f"{S}/work/easy_now_{sys.argv[1]}.parquet")
