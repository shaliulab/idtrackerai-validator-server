"""
pe_validation.py  —  PE bout validation endpoints for the FlyHostel viewer.

Designed to drop into the existing flat-route app.py (NOT a blueprint, to match the
app's style). It reuses the global SELECTED_EXPERIMENT rather than taking experiment
on every request, and keeps annotations in their OWN sqlite file so switching
experiments (which rebinds SQLAlchemy to the tracking .db) never touches them.

INTEGRATION (see the chat message):
  1. put this file next to app.py
  2. in app.py, after `app = Flask(__name__)` and the imports, add:
         from pe_validation import register_pe_validation
         register_pe_validation(app, get_selected_experiment=lambda: SELECTED_EXPERIMENT)
  3. set PE_BOUTS_DIR / PE_MEDIA_DIR / PE_DB below (or via env)
"""
import os
import sqlite3
import datetime
import pandas as pd
from flask import jsonify, request, send_from_directory

from flyhostel.utils import (
    get_basedir,
    get_chunksize,
    get_framerate
)

# --- where your pipeline wrote things (edit or set via env) ---------------------
# PE media lives UNDER EACH EXPERIMENT'S TREE, so the media dir is DERIVED from the
# experiment per request (see _media_dir), not a fixed constant.
PE_DB          = os.environ.get("PE_DB", "pe_annotations.db")      # separate from tracking DB


def _media_dir(experiment):
    # parent of both plots/ and videos/, so a request for "videos/xxx.mp4" or
    # "plots/xxx.png" resolves as a subpath. send_from_directory blocks ../ escapes.
    return os.path.join(get_basedir(experiment.replace("/", "_")),
                        "flyhostel", "proboscis_extensions")


_VERDICTS = ("pe", "feed", "groom", "walk", "other", "merge", "unsure")


def _init_db():
    verdict_list = ",".join(f"'{v}'" for v in _VERDICTS)
    with sqlite3.connect(PE_DB) as c:
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS pe_annotations (
                experiment  TEXT    NOT NULL,
                identity    INTEGER NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame   INTEGER NOT NULL,
                burst_id    INTEGER,
                bout_uid    REAL,
                verdict     TEXT    NOT NULL CHECK (verdict IN ({verdict_list})),
                pe_score    REAL,
                reviewer    TEXT,
                reviewed_at TEXT    NOT NULL,
                PRIMARY KEY (experiment, identity, start_frame, end_frame)
            )""")


def _fly_id(experiment, identity):
    # experiment global is stored as "FlyHostel4/2X/2025-02-04"; the feather/media use
    # the flat "FlyHostel4_2X_2025-02-04__01" form.
    flat = experiment.replace("/", "_")
    return f"{flat}__{str(int(identity)).zfill(2)}"


def register_pe_validation(app, get_selected_experiment):
    """Attach the PE-validation routes to an existing Flask `app`.
    `get_selected_experiment` is a 0-arg callable returning the current experiment
    string (pass `lambda: SELECTED_EXPERIMENT` from app.py so it stays live)."""
    _init_db()

    def _experiment_or_400():
        exp = get_selected_experiment()
        if not exp:
            return None, (jsonify({"error": "no experiment loaded"}), 400)
        return exp, None

    @app.route("/api/pe/bouts", methods=["GET"])
    def pe_bouts():
        """PE-labelled bouts for a fly, joined with any existing verdict.
        Query: ?identity=1  (experiment comes from the loaded session)."""
        exp, err = _experiment_or_400()
        if err:
            return err

        fly = request.args["fly"]

        experiment, identity = fly.split("__")
        identity = int(identity)

        basedir=get_basedir(experiment)
        pe_bouts_dir=f"{basedir}/flyhostel/proboscis_extensions/pe_bouts"

        chunksize=get_chunksize(exp.replace("/", "_"))
        feather = os.path.join(pe_bouts_dir, f"{fly}_pe_bouts.feather")
        if not os.path.exists(feather):
            return jsonify({"error": f"no bouts feather for {fly}"}), 404

        df = pd.read_feather(feather)
        # bursts that contain at least one PE bout
        # pe_bursts = set(df.loc[df["label"] == "pe", "burst_id"].unique())
        # df = df[df["burst_id"].isin(pe_bursts)].copy()

        # low score first among the PE ones; keep burst grouping intact
        df = df.sort_values(["burst_id", "start_fn"])

        df["start_fidx"] = df["start_fn"] % chunksize
        df["end_fidx"]   = df["end_fn"]   % chunksize
        df["is_pe"]      = (df["label"] == "pe")          # annotatable vs display-only

        cols = [c for c in ("burst_id", "bout_uid", "start_fn", "end_fn",
                            "start_fidx", "end_fidx", "n_in_burst", "is_solitary",
                            "pe_score", "dur_s", "label", "label_reason", "is_pe")
                if c in df.columns]
        bouts = df[cols].to_dict("records")

        # attach existing verdicts
        with sqlite3.connect(PE_DB) as c:
            c.row_factory = sqlite3.Row
            seen = {(r["start_frame"], r["end_frame"]): r["verdict"]
                    for r in c.execute(
                        "SELECT start_frame,end_frame,verdict FROM pe_annotations "
                        "WHERE experiment=? AND identity=?", (exp, identity))}

        fly = _fly_id(exp, identity)

        for b in bouts:
            b["verdict"] = seen.get((int(b["start_fn"]), int(b["end_fn"])))
            b["trace_stem"] = f"{fly}_burst_{int(b['burst_id'])}"                       # -> plots/{...}.png
            b["media_stem"] = f"{fly}_burst_{int(b['burst_id'])}_bout_{int(b['bout_uid'])}"
        return jsonify(bouts)

    @app.route("/api/pe/annotate", methods=["POST"])
    def pe_annotate():
        exp, err = _experiment_or_400()
        if err:
            return err
        d = request.get_json(force=True)
        need = ("fly", "start_frame", "end_frame", "verdict")
        if not all(k in d for k in need):
            return jsonify({"error": f"need {need}"}), 400
        if d["verdict"] not in _VERDICTS:
            return jsonify({"error": f"verdict must be one of {_VERDICTS}"}), 400

        identity = int(d["fly"].rsplit("__", 1)[1])   # only the identity from fly

        with sqlite3.connect(PE_DB) as c:
            c.execute("""
                INSERT INTO pe_annotations
                (experiment, identity, start_frame, end_frame, burst_id, bout_uid,
                verdict, pe_score, reviewer, reviewed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(experiment, identity, start_frame, end_frame)
                DO UPDATE SET verdict=excluded.verdict, pe_score=excluded.pe_score,
                            reviewer=excluded.reviewer, reviewed_at=excluded.reviewed_at
            """, (exp, identity, int(d["start_frame"]), int(d["end_frame"]),   # ← exp, not flat
                d.get("burst_id"), d.get("bout_uid"), d["verdict"], d.get("pe_score"),
                d.get("reviewer", "anon"), datetime.datetime.utcnow().isoformat()))
        return jsonify({"ok": True})

    @app.route("/api/pe/media/<path:filename>", methods=["GET"])
    def pe_media(filename):
        exp, err = _experiment_or_400()
        if err:
            return err
        resp = send_from_directory(os.path.realpath(_media_dir(exp)), filename)
        # explicit CORS so a cross-origin <video crossorigin> can taint-free-draw to canvas
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Range"
        resp.headers["Access-Control-Expose-Headers"] = "Content-Range, Accept-Ranges, Content-Length"
        return resp

    @app.route("/api/pe/export", methods=["GET"])
    def pe_export():
        """All verdicts for the loaded experiment as JSON (for training / audit)."""
        exp, err = _experiment_or_400()
        if err:
            return err
        with sqlite3.connect(PE_DB) as c:
            c.row_factory = sqlite3.Row
            rows = [dict(r) for r in c.execute(
                "SELECT * FROM pe_annotations WHERE experiment=?", (exp,))]
        return jsonify(rows)


    @app.route("/api/pe/trace", methods=["GET"])
    def pe_trace():
        """Per-frame trace + bout spans + inter-bout gaps for ONE burst.
        Query: ?identity=1&burst_id=2639"""
        exp, err = _experiment_or_400()
        if err:
            return err
        fly = request.args["fly"]
        burst_id = int(request.args["burst_id"])

        experiment, identity = fly.split("__")
        identity = int(identity)

        fps = get_framerate(exp.replace("/", "_"))

        traces_file = os.path.join(_media_dir(exp), f"{fly}_traces.feather")   # the extract_burst_traces output
        if not os.path.exists(traces_file):
            return jsonify({"error": f"no trace feather for {fly}"}), 404
        d = pd.read_feather(traces_file)
        d = d[d["burst_id"] == burst_id].sort_values("frame_number").copy()
        if d.empty:
            return jsonify({"error": f"burst {burst_id} not in trace"}), 404

        # normalize like the R script: t from 0, dist from its min
        f0 = int(d["frame_number"].min())
        d["t_s"] = (d["frame_number"] - f0) / fps
        dmin = d["dist_mm"].min(skipna=True)
        d["dist_rel"] = d["dist_mm"] - dmin

        # per-bout spans (duration) and inter-bout gaps, computed once here
        spans = (d.dropna(subset=["bout_in_burst"])
                .groupby("bout_in_burst")["t_s"].agg(["min", "max"])
                .reset_index().sort_values("min"))
        spans_out = [{"bout_in_burst": int(r["bout_in_burst"]),
                    "t0": float(r["min"]), "t1": float(r["max"]),
                    "dur": float(r["max"] - r["min"])} for _, r in spans.iterrows()]
        gaps_out = []
        for a, b in zip(spans_out[:-1], spans_out[1:]):
            gaps_out.append({"g0": a["t1"], "g1": b["t0"], "gap": b["t0"] - a["t1"]})

        return jsonify({
            "fly": fly, "burst_id": burst_id, "fps": fps, "start_frame": f0,
            "points": [{"t_s": round(float(t), 4),
                    "dist": None if pd.isna(v) else round(float(v), 4),
                    "conf": None if pd.isna(cf) else round(float(cf), 4),   # NEW
                    "bout_uid": None if pd.isna(u) else int(u),
                    "is_peak": bool(p)}
                   for t, v, cf, u, p in zip(d["t_s"], d["dist_rel"],
                                             d["prob_conf"], d["bout_uid"],  # NEW: prob_conf
                                             d["is_peak"])],
            "spans": spans_out, "gaps": gaps_out,
        })