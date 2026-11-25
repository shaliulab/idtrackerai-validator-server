"""Snapshot machine rows of the chunks holding saved frames: per experiment an npz of ROI_0 + IDENTITY rows from
chunk start to the last saved frame of the chunk (+ margin), plus notches, n_animals, chunksize."""
import sys, os, sqlite3, glob, ast, json, time
import numpy as np
ROOT = "/flyhostel_data/videos"
S = os.path.dirname(os.path.abspath(__file__)) + "/.."
MARGIN = 50

def basedir_of(flat):
    if "_arena" in flat:
        flat, arena = flat.split("_arena")
    else:
        arena = None
    p = flat.split("_")
    b = f"{p[0]}/{p[1]}/{p[2]}_{p[3]}"
    return b + (f"/arenas/{arena}" if arena else "")

def main(flat):
    out = f"{S}/rows/{flat}.npz"
    if os.path.exists(out):
        return
    bd = os.path.join(ROOT, basedir_of(flat))
    dbfile = [f for f in glob.glob(bd + "/*.db") if not f.endswith("index.db")][0]
    side = sqlite3.connect(f"{S}/side/{flat}.db")
    con = sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True)
    meta = dict(con.execute("SELECT field, value FROM METADATA"))
    cs = int(float(meta["chunksize"]))
    conf = json.loads(meta["idtrackerai_conf"])
    n = int(conf.get("_number_of_animals", {}).get("value") or conf.get("number_of_animals") or 0)
    notches = []
    for shape, spec in con.execute("SELECT shape, specification FROM LANDMARKS"):
        if str(shape).lower() == "notch":
            pts = np.asarray(ast.literal_eval(spec), float); notches.append(pts.mean(0).tolist())
    saved = np.array([r[0] for r in side.execute("SELECT frame_number FROM frames ORDER BY 1")])
    chunks = np.unique(saved // cs)
    R, I = [], []
    t0 = time.time()
    for c in chunks:
        lo = c * cs; hi = min(saved[saved // cs == c].max() + MARGIN, lo + cs - 1)
        R.append(np.array(con.execute("SELECT frame_number, in_frame_index, x, y, fragment, area, modified FROM ROI_0 "
                                      "WHERE frame_number BETWEEN ? AND ?", (int(lo), int(hi))).fetchall(), float).reshape(-1, 7))
        I.append(np.array(con.execute("SELECT frame_number, in_frame_index, local_identity, identity FROM IDENTITY "
                                      "WHERE frame_number BETWEEN ? AND ?", (int(lo), int(hi))).fetchall(), float).reshape(-1, 4))
    R = np.concatenate(R); I = np.concatenate(I)
    # join
    import pandas as pd
    r = pd.DataFrame(R, columns="fn ifi x y frag area mod".split())
    i = pd.DataFrame(I, columns="fn ifi lid ident".split()).drop_duplicates(["fn", "ifi"])
    m = r.merge(i, on=["fn", "ifi"], how="left").fillna({"lid": 0, "ident": 0}).sort_values(["fn", "ifi"])
    conc = np.array(con.execute("SELECT chunk, local_identity, local_identity_after, identity, is_inferred, is_broken FROM CONCATENATION").fetchall(), float)
    np.savez_compressed(out, rows=m[["fn", "ifi", "x", "y", "frag", "area", "mod", "lid", "ident"]].to_numpy(), notches=np.array(notches),
                        n=n, cs=cs, conc=conc, w=int(meta.get("frame_width", 0)), h=int(meta.get("frame_height", 0)),
                        ppcm=float(meta.get("pixels_per_cm", 0)), dbfile=dbfile)
    print(flat, n, cs, len(chunks), len(m), f"{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    os.makedirs(f"{S}/rows", exist_ok=True)
    for f in sys.argv[1:]:
        main(f)
