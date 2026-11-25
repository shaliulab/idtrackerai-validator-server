#!/usr/bin/env python
"""Train the Annotate tab's scene scorer (idtrackerai_validator_server/scene_scorer.py) on the frames people saved.

    FLYHOSTEL_VIDEOS=/flyhostel_data/videos python scripts/train_scene_scorer.py [EXPERIMENT ...]
        [--target id_ok|ok] [--precision 0.99] [--min-frames 500] [--refresh] [--jobs 4]

Without experiments: every experiment of index.txt with more than one fly, the validated tables
(experiments.VALIDATED_TABLES) and at least --min-frames frames saved in the Annotate tab. Per experiment the
stretches are cached in $SCENE_SCORER_DIR/datasets/<flat>.jsonl (kept unless --refresh or the sidecar / database
changed). Prints a leave-one-experiment-out evaluation (threshold chosen on the other experiments), then writes
$SCENE_SCORER_DIR/model.pkl (all experiments; threshold from their out-of-fold scores) and report.json.
Read-only on the experiments; ~1–5 min per 12X experiment (one read of each chunk with saved scenes)."""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from idtrackerai_validator_server import lab_paths, scene_scorer, scene_scorer_data  # noqa: E402
from idtrackerai_validator_server.experiments import is_validated, to_flat, to_relpath  # noqa: E402

DATA_DIR = os.path.join(lab_paths.SCENE_SCORER_DIR, "datasets")


def candidates(min_frames):
    import sqlite3
    root = os.environ["FLYHOSTEL_VIDEOS"]
    out = []
    with open(os.path.join(root, "index.txt")) as handle:
        for line in handle:
            db = line.strip()
            if not db.endswith(".db") or db.endswith("index.db"):
                continue
            path = os.path.join(root, db) if not os.path.isabs(db) else db
            flat = os.path.basename(path)[:-3]
            side = os.path.join(os.path.dirname(path), "flyhostel", "validation", "validator_annotations.db")
            if not os.path.exists(side) or not is_validated(path):
                continue
            try:
                with sqlite3.connect(f"file:{side}?mode=ro", uri=True) as con:
                    n = con.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
            except sqlite3.Error:
                continue
            if n >= min_frames:
                out.append(flat)
    return sorted(set(out))


def stamp(flat):
    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], to_relpath(flat))
    files = [os.path.join(basedir, flat + ".db"), os.path.join(basedir, "flyhostel", "validation", "validator_annotations.db"),
             os.path.join(basedir, "flyhostel", "validation", "scene_qc.csv")]
    return [[os.path.getmtime(f), os.path.getsize(f)] if os.path.exists(f) else None for f in files]


def build(flat, refresh=False):
    os.nice(10)
    path = os.path.join(DATA_DIR, f"{flat}.jsonl")
    meta = path + ".stamp"
    st = stamp(flat)
    if not refresh and os.path.exists(path) and os.path.exists(meta):
        with open(meta) as handle:
            if json.load(handle) == st:
                return flat, scene_scorer_data.load_dataset(path), "cached"
    t = time.time()
    rows = scene_scorer_data.dataset(flat)
    scene_scorer_data.save_dataset(rows, path)
    with open(meta, "w") as handle:
        json.dump(st, handle)
    return flat, rows, f"{time.time() - t:.0f} s"


def oof_scores(X, y, groups):
    from sklearn.model_selection import GroupKFold
    out = np.zeros(len(y))
    k = min(5, len(set(groups)))
    for a, b in GroupKFold(n_splits=k).split(X, y, groups):
        out[b] = scene_scorer.new_model().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="*")
    parser.add_argument("--target", default="id_ok", choices=["id_ok", "ok"])
    parser.add_argument("--precision", type=float, default=0.99)
    parser.add_argument("--min-frames", type=int, default=500)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", default=scene_scorer.MODEL_PATH)
    args = parser.parse_args()
    flats = [to_flat(e) for e in args.experiments] or candidates(args.min_frames)
    print(f"{len(flats)} experiments: {', '.join(flats)}", flush=True)
    data = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for flat, rows, how in pool.map(build, flats, [args.refresh] * len(flats)):
            data[flat] = rows
            print(f"  {flat}: {len(rows)} stretches ({how})", flush=True)
    rows = [r for f in flats for r in data[f]]
    if len({r["experiment"] for r in rows}) < 2:
        sys.exit("need stretches of at least 2 experiments")
    X = np.array([[float(r[k]) for k in scene_scorer.FEATURES] for r in rows])
    y = np.array([int(r[args.target]) for r in rows])
    groups = np.array([r["experiment"] for r in rows])
    frames = np.array([r["last"] - r["first"] + 1 for r in rows])
    report = {"target": args.target, "precision": args.precision, "experiments": {}, "features": scene_scorer.FEATURES}
    print(f"\nleave one experiment out (target {args.target}, precision {args.precision}):")
    for test in flats:
        te = groups == test
        tr = ~te
        if te.sum() == 0 or len(set(groups[tr])) < 2:
            continue
        thr = scene_scorer.precision_threshold(oof_scores(X[tr], y[tr], groups[tr]), y[tr], args.precision)
        p = scene_scorer.new_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        acc = p >= thr if thr is not None else np.zeros(te.sum(), bool)
        r = {"stretches": int(te.sum()), "frames": int(frames[te].sum()), "right": float(y[te].mean()),
             "threshold": thr, "accepted": int(acc.sum()), "accepted_frames": int(frames[te][acc].sum()),
             "precision": float(y[te][acc].mean()) if acc.any() else None}
        report["experiments"][test] = r
        print(f"  {test}: {r['stretches']} stretches, {100 * r['right']:.1f} % right; accepted {r['accepted']} "
              f"({r['accepted_frames']} frames) at {100 * (r['precision'] or 0):.1f} % precision", flush=True)
    thr = scene_scorer.precision_threshold(oof_scores(X, y, groups), y, args.precision)
    if thr is None:
        sys.exit(f"no threshold reaches {args.precision} precision")
    model = scene_scorer.new_model().fit(X, y)
    acc = sum(r["accepted"] for r in report["experiments"].values())
    right = sum(r["accepted"] * (r["precision"] or 0) for r in report["experiments"].values())
    report.update(threshold=thr, trained_on=flats, stretches=len(rows), trained_at=time.time(),
                  loeo_accepted=acc, loeo_precision=right / acc if acc else None)
    scene_scorer.save_model({"model": model, "features": scene_scorer.FEATURES, "threshold": thr,
                             "target": args.target, "precision": args.precision, "report": report}, args.output)
    with open(os.path.join(os.path.dirname(args.output), "report.json"), "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"\nthreshold {thr:.4f}; model written to {args.output}")


if __name__ == "__main__":
    main()
