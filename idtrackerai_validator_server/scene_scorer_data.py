"""Training data of the scene scorer (scene_scorer.py): the stretches of scene_qc.csv's scenes a person saved
entirely in an experiment, proposed as the server would just before the person reached them (anchor: the last frame
they saved before it in the chunk, else idtrackerai's in their labels of the frames before), with the outcome
against what they saved. Read-only on the experiment (database and sidecar opened mode=ro)."""
import bisect
import json
import os
import sqlite3

import numpy as np

from idtrackerai_validator_server import continuity, scene_scorer
from idtrackerai_validator_server.compat import get_number_of_animals
from idtrackerai_validator_server.experiments import to_flat, to_relpath


def _basedir(experiment):
    return os.path.join(os.environ["FLYHOSTEL_VIDEOS"], to_relpath(experiment))


def _read_scenes(basedir, experiment, n_animals):
    from idtrackerai_validator_server.box_annotations import read_scenes
    return read_scenes(basedir, experiment, n_animals)[0]


def open_experiment(experiment, sidecar=None):
    """dict: experiment, dbfile, chunksize, n_animals, scenes, saved {frame: boxes}, saved_before(first),
    translate(first) (idtrackerai anchors in the person's labels of the frames before `first`). sidecar: a copy of
    validator_annotations.db to read instead of the experiment's."""
    flat = to_flat(experiment)
    basedir = _basedir(experiment)
    dbfile = os.path.join(basedir, flat + ".db")
    with sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True) as con:
        chunksize = int(float(con.execute("SELECT value FROM METADATA WHERE field = 'chunksize'").fetchone()[0]))
    n_animals = int(get_number_of_animals(flat))
    scenes = _read_scenes(basedir, experiment, n_animals)
    sidecar = sidecar or os.path.join(basedir, "flyhostel", "validation", "validator_annotations.db")
    saved, updated = {}, {}
    if os.path.exists(sidecar):
        with sqlite3.connect(f"file:{sidecar}?mode=ro", uri=True) as con:
            updated = dict(con.execute("SELECT frame_number, updated FROM frames"))
            for fn, x0, y0, x1, y1, lab in con.execute(
                    "SELECT b.frame_number, x0, y0, x1, y1, label FROM boxes b JOIN frames f USING (frame_number) "
                    "ORDER BY b.frame_number, b.rowid"):
                saved.setdefault(fn, []).append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "label": lab or ""})
    for fn in updated:
        saved.setdefault(fn, [])
    order = sorted(saved)

    def saved_before(first):
        i = bisect.bisect_left(order, first) - 1
        lo = first // chunksize * chunksize
        if i >= 0 and order[i] >= lo:
            fn = order[i]
            return fn, saved[fn], updated.get(fn)
        return None

    def translate(first):
        """labels of the fragments in the frames saved before `first` in its chunk, as anchor_labels reads them"""
        from idtrackerai_validator_server.box_annotations import anchor_labels, box_fragments, _machine_rows
        lo = first // chunksize * chunksize
        labels = {}
        for fn in order[bisect.bisect_left(order, lo):bisect.bisect_left(order, first)]:
            rows = _machine_rows(dbfile, fn)
            for b, frag in zip(saved[fn], box_fragments(saved[fn], rows)):
                if frag is not None and str(b["label"]).isdigit():
                    labels.setdefault(str(frag), []).append((fn, b["label"]))
        if not labels:
            return None
        return lambda fn, _rows: anchor_labels(dbfile, fn, n_animals, labels, {})

    return {"experiment": flat, "dbfile": dbfile, "chunksize": chunksize, "n_animals": n_animals, "scenes": scenes,
            "saved": saved, "saved_before": saved_before, "translate": translate}


def dataset(experiment, sidecar=None, progress=None):
    """[{experiment, first, last, features..., ok, id_ok, n_errors}] of the stretches whose frames were all saved."""
    exp = open_experiment(experiment, sidecar)
    saved = exp["saved"]
    full = [sc for sc in exp["scenes"] if all(fn in saved for fn in range(sc["start"], sc["start"] + sc["length"]))]
    # a stretch is kept only when every one of its scenes is fully saved
    out = []
    for first, last, members, run, results, rows in scene_scorer.stretch_runs(
            exp["dbfile"], full, exp["chunksize"], exp["n_animals"], exp["saved_before"], exp["translate"], progress):
        if any(fn not in saved for fn in range(first, last + 1)):
            continue
        errors = [e for fn in range(first, last + 1) if fn in results
                  for e in scene_scorer.frame_errors(saved[fn], rows[fn], results[fn])]
        rec = {"experiment": exp["experiment"], "first": first, "last": last,
               **scene_scorer.features(first, last, exp["n_animals"], members, run, results, rows),
               "ok": int(not errors), "id_ok": int(not [e for e in errors if e[0] in scene_scorer.ID_ERRORS]),
               "n_errors": len(errors), "errors": json.dumps(sorted({e[0] for e in errors}))}
        out.append(rec)
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def save_dataset(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as handle:
        for r in rows:
            handle.write(json.dumps(r, default=_json_default) + "\n")
    os.replace(path + ".tmp", path)


def load_dataset(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]
