"""Learned scorer of the Annotate tab's scenes: how likely the continuity proposal (continuity.RunV2) gets a whole
stretch of scene_qc.csv's scenes (overlapping / adjacent ones merged) right, from what the tracker did in it and the
detections' geometry. Trained on the frames people saved in validated experiments (scripts/train_scene_scorer.py),
kept in $SCENE_SCORER_DIR/model.pkl; with ANNOTATE_EASY_METHOD=learned the tab's easy scenes (◆) are the stretches
scoring at least the model's threshold instead of continuity.classify_scenes's rules.

Measured (2026-11-25) on the 5 validated 12X experiments, 1696 stretches whose frames were all saved, each
experiment scored by a model trained on the 10 others (4 × 12X + 3X / 4X / 6X / 8X arena), proposal from the frame
saved before the stretch (else idtrackerai's anchor) through the whole stretch with no person's input:
- target `id_ok` (every detection gets the person's fly; boxes the person added or removed where nothing / something
  was detected not counted): 1155 stretches (68 %, 12371 frames) accepted at 99.3 % precision, threshold chosen on
  the training experiments; the rules (classify_scenes) accept 778 (46 %) at 96.0 % with this tracker;
- target `ok` (also every box the person drew where nothing was detected): the top half of the stretches by score
  are 98.2 % entirely right, the rules' 46 % 92.0 % (no threshold reaches 99 % on held-out experiments).
Strongest features: closest pair of flies, largest blob (merged flies), flies kept undetected, detections without
a fly of the anchor. A backward run from idtrackerai's stable frames after the stretch added nothing.
"""
import logging
import os
import pickle

import numpy as np

from idtrackerai_validator_server import continuity, lab_paths

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(lab_paths.SCENE_SCORER_DIR, "model.pkl")
REASONS = ["ids missing", "id changes", "broken", "gap", "between chunks", "failed fragments"]
FEATURES = (["length", "n", "anchor_saved", "anchor_dist", "n_scenes"]
            + ["r_" + r.replace(" ", "_") for r in REASONS]
            + ["f_n_unlab", "f_relabel", "f_noid", "f_ambig", "f_new", "f_notch", "f_leftover", "f_ghost_ff",
               "f_ghost_max", "f_flags"]
            + ["rows_min", "rows_max", "frames_short", "frames_long", "dmin", "dmin_med", "frames_close40",
               "area_max_ratio", "lid0", "frags"])
NOTES = (("relabel", "continuity: idtrackerai says "), ("noid", "continuity: idtrackerai says no identity"),
         ("ambig", "another fly"), ("new", "no fly of the anchor nearby: idtrackerai"), ("notch", "at a notch"),
         ("leftover", "fly "))
GHOST_OK_PX = 40      # a fly kept where nothing was detected is right within this of the person's box centre


def features(first, last, n_animals, members, run, results, rows):
    """{feature: value} of one stretch. run: the RunV2 it was proposed with (anchor), results: {frame: step result},
    rows: read_rows(..., fragments=True) holding the stretch's frames."""
    frames = [fn for fn in range(first, last + 1) if fn in results]
    f = {"length": last - first + 1, "n": n_animals, "anchor_saved": int(run.kind == "saved"),
         "anchor_dist": first - run.anchor, "n_scenes": len(members)}
    for r in REASONS:
        f["r_" + r.replace(" ", "_")] = int(any(r in s.get("reason", "") for s in members))
    notes = [note for fn in frames for _, note, _ in results[fn]["rows"]]
    f["f_n_unlab"] = sum(1 for fn in frames for lab, _, _ in results[fn]["rows"] if lab is None)
    for key, prefix in NOTES:
        f["f_" + key] = sum(1 for note in notes if note.startswith(prefix)
                            and (key != "leftover" or "reappeared" in note))
    f["f_relabel"] -= f["f_noid"]
    f["f_ghost_ff"] = sum(len(results[fn]["ghosts"]) for fn in frames)
    f["f_ghost_max"] = max([len(results[fn]["ghosts"]) for fn in frames] or [0])
    f["f_flags"] = sum(len(results[fn]["flags"]) for fn in frames)
    stretch = [np.asarray(rows[fn], dtype=float).reshape(-1, 5) for fn in range(first, last + 1) if fn in rows]
    counts = np.array([len(r) for r in stretch]) if stretch else np.array([n_animals])
    f["rows_min"], f["rows_max"] = int(counts.min() - n_animals), int(counts.max() - n_animals)
    f["frames_short"], f["frames_long"] = int((counts < n_animals).sum()), int((counts > n_animals).sum())
    dmins, areas = [], []
    for r in stretch:
        if len(r) >= 2:
            d = np.hypot(r[:, None, 0] - r[None, :, 0], r[:, None, 1] - r[None, :, 1])
            d[np.diag_indices(len(r))] = np.inf
            dmins.append(float(d.min()))
        areas.extend(r[:, 4].tolist())
    f["dmin"] = min(dmins) if dmins else 999.0
    f["dmin_med"] = float(np.median(dmins)) if dmins else 999.0
    f["frames_close40"] = int(sum(d < 40 for d in dmins))
    first_rows = stretch[0] if stretch else np.zeros((0, 5))
    med_area = float(np.median(first_rows[:, 4])) if len(first_rows) else (float(np.median(areas)) if areas else 0)
    f["area_max_ratio"] = max(areas) / med_area if areas and med_area else 1.0
    f["lid0"] = int(sum(int((r[:, 2] == 0).sum()) for r in stretch))
    f["frags"] = len({int(x) for r in stretch for x in r[:, 3]})
    return f


def stretch_runs(dbfile, scenes, chunksize, n_animals, saved_before, translate, progress=None):
    """Yield (first, last, members, run, results, rows) per stretch of the scenes within one chunk, the RunV2 started
    as the server's continuity2 proposal of its first frame: saved_before(first) -> (frame, boxes, updated) of the
    last frame saved before it in the chunk or None; translate(first) -> translate function for an idtrackerai
    anchor (continuity.track) or None. Stretches across a chunk boundary or without anchor are skipped. Rows are
    read once per chunk."""
    stretches = continuity._stretches(scenes)
    chunk, rows5, rows3 = None, None, None
    notches = continuity.notch_centres(dbfile)
    for k, (first, last, members) in enumerate(stretches):
        if progress is not None:
            progress(k / max(1, len(stretches)))
        if first // chunksize != last // chunksize:
            continue
        lo = first // chunksize * chunksize
        if chunk != lo:
            chunk = lo
            rows5 = continuity.read_rows(dbfile, lo, lo + chunksize - 1, fragments=True)
            rows3 = {fn: [r[:3] for r in rs] for fn, rs in rows5.items()}
        m_fn, m_rows = continuity.machine_anchor(dbfile, first, lo, chunksize, n_animals, rows3)
        saved = saved_before(first)
        if saved is not None and (m_fn is None or saved[0] >= m_fn):
            s_fn, boxes, _ = saved
            r = rows5.get(s_fn, [])
            run = continuity.RunV2(dbfile, s_fn, "saved", continuity._box_positions(boxes, r), notches, n_animals,
                                   continuity._box_fragments(boxes, r))
        elif m_fn is not None:
            tr = translate(first)
            labels = list(tr(m_fn, m_rows)) if tr is not None else [lid for _, _, lid in m_rows]
            r = rows5.get(m_fn, [])
            run = continuity.RunV2(dbfile, m_fn, "tracking",
                                   {lab: (x, y, True) for (x, y, _), lab in zip(m_rows, labels)}, notches, n_animals,
                                   {lab: rr[3] for rr, lab in zip(r, labels)} if len(r) == len(labels) else {})
        else:
            continue
        run.extend(first - 1, rows5)
        results = {}
        for fn in range(first, last + 1):
            if fn in rows5:
                results[fn] = run.step(fn, rows5[fn])
        yield first, last, members, run, results, rows5


# ---- outcome: the proposal against the boxes a person saved ------------------------------------------------------

def frame_errors(boxes, rows, result):
    """[(kind, detail)] of a proposal (continuity step result over rows) against a person's saved boxes: wrong_label,
    unlabelled, extra_label (a detection the person left without fly), missing_fly / ghost_misplaced / extra_ghost
    (flies drawn where nothing was detected). Identity errors are the first two."""
    lab_boxes = [b for b in boxes if str(b["label"]).isdigit()]
    pairs = []
    for i, b in enumerate(lab_boxes):
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
        for j, r in enumerate(rows):
            if b["x0"] <= r[0] <= b["x1"] and b["y0"] <= r[1] <= b["y1"]:
                pairs.append(((r[0] - cx) ** 2 + (r[1] - cy) ** 2, i, j))
    pairs.sort()
    truth, used = [None] * len(rows), set()
    for _, i, j in pairs:
        if i not in used and truth[j] is None:
            truth[j] = int(lab_boxes[i]["label"])
            used.add(i)
    errors = []
    for j, (g, (p, _, _)) in enumerate(zip(truth, result["rows"])):
        if g is None:
            if p is not None:
                errors.append(("extra_label", j))
        elif p is None:
            errors.append(("unlabelled", j))
        elif p != g:
            errors.append(("wrong_label", j))
    kept = {lab: (x, y) for lab, x, y, _ in result["ghosts"]}
    drawn = set()
    for i, b in enumerate(lab_boxes):
        if i in used:
            continue
        lab = int(b["label"])
        drawn.add(lab)
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
        if lab not in kept:
            errors.append(("missing_fly", lab))
        elif np.hypot(kept[lab][0] - cx, kept[lab][1] - cy) > max(GHOST_OK_PX, 0.5 * max(b["x1"] - b["x0"], b["y1"] - b["y0"])):
            errors.append(("ghost_misplaced", lab))
    errors += [("extra_ghost", lab) for lab in kept if lab not in drawn]
    return errors


ID_ERRORS = ("wrong_label", "unlabelled")


# ---- model -----------------------------------------------------------------------------------------------------

THREADS = 1    # OpenMP threads of the model's fits / predictions (64 by default here: they spin on a busy server; one is ~0.1 s)


def limited():
    """Context limiting the OpenMP threads of sklearn's gradient boosting to THREADS"""
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=THREADS, user_api="openmp")


def new_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=0)


def precision_threshold(scores, y, precision, min_accepted=20):
    """Lowest score whose set of stretches scoring at least it is right with >= precision (None if none)."""
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(y)[order]
    cum = np.cumsum(y) / np.arange(1, len(y) + 1)
    ok = np.where(cum >= precision)[0]
    ok = ok[ok >= min_accepted - 1]
    return float(np.asarray(scores)[order][ok.max()]) if len(ok) else None


_model = {}   # path -> (mtime, model dict)


def load_model(path=None):
    """{"model", "features", "threshold", "target", ...} of the trained scorer, or None (not trained yet)."""
    path = path or MODEL_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    hit = _model.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    with open(path, "rb") as handle:
        model = pickle.load(handle)
    _model[path] = (mtime, model)
    return model


def save_model(model, path=None):
    path = path or MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        pickle.dump(model, handle)
    os.replace(tmp, path)


def score_scenes(dbfile, scenes, chunksize, n_animals, saved_before, translate, model=None, progress=None):
    """{scene start: {"easy": bool, "why": text, "p": score or None}} of every scene, its stretch scored by the
    model (load_model()); the easy ones score at least the model's threshold. Raises RuntimeError without model."""
    model = model or load_model()
    if model is None:
        raise RuntimeError(f"no scene scorer trained: run scripts/train_scene_scorer.py (writes {MODEL_PATH})")
    out = {sc["start"]: {"easy": False, "why": "crosses a chunk boundary or no anchor in the chunk", "p": None}
           for sc in scenes}
    names = model["features"]
    batch = []     # one predict_proba at the end: per call it starts its OpenMP threads (30 min of CPU for 1300)
    for first, last, members, run, results, rows in stretch_runs(dbfile, scenes, chunksize, n_animals, saved_before,
                                                                  translate, progress):
        f = features(first, last, n_animals, members, run, results, rows)
        batch.append((members, [float(f[k]) for k in names]))
    if batch:
        with limited():
            ps = model["model"].predict_proba(np.array([x for _, x in batch]))[:, 1]
        for (members, _), p in zip(batch, ps.tolist()):
            easy = p >= model["threshold"]
            why = f"learned score {p:.3f} {'≥' if easy else '<'} {model['threshold']:.3f}"
            for sc in members:
                out[sc["start"]] = {"easy": bool(easy), "why": why, "p": p}
    return out


def main():
    """python -m idtrackerai_validator_server.scene_scorer EXPERIMENT: score its scenes (needs $FLYHOSTEL_VIDEOS)"""
    import argparse
    import json
    from idtrackerai_validator_server import scene_scorer_data
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("experiment")
    args = parser.parse_args()
    exp = scene_scorer_data.open_experiment(args.experiment)
    result = score_scenes(exp["dbfile"], exp["scenes"], exp["chunksize"], exp["n_animals"], exp["saved_before"],
                          exp["translate"])
    easy = sum(1 for r in result.values() if r["easy"])
    print(json.dumps({"scenes": len(result), "easy": easy}, indent=1))


if __name__ == "__main__":
    main()
