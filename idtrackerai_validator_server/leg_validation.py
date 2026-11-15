"""
leg_validation.py  —  validation of the middle-leg movement detector (Leg movements tab).

The detector (scripts/analyses/arousability/leg_movements.py of the FlyHostel analysis repository) finds repeated movements of a
middle leg while the thorax stays in place, from the raw pose only, and writes per fly:

    $LEG_MOVEMENTS_DIR/events/<fly>.csv       one row per event: event, start, end (s since ZT0), leg (mL, mR, both), strokes,
                                              strokes_per_s, amplitude, front_strokes, hind_strokes, thorax_displacement, alone,
                                              immobile_s, tip_score, jump_fraction, clean (pose quality: clean = the tip of the
                                              middle leg is scored well and moves smoothly, not in single-frame jumps)
    $LEG_MOVEMENTS_DIR/traces/<fly>.feather   what the detector saw, from 3 s before to 3 s after every event (30 Hz): event, t,
                                              speed (thorax), and per leg (mL, mR, fL, fR, rL, rR) the distance of the tip from
                                              its resting position (body lengths) and stroke_<leg>

The counting step (leg_movements_per_fly.py) writes $LEG_MOVEMENTS_DIR/model/movements.csv: every event of its population (middle
legs alone, while immobile) with the classifier's p_middle, movement (counted as a real movement), near (during a contact with
another fly or just after it) and partner (that fly). Every event shown gets those columns when it is in the file, and the
selections movements, movements_near and movements_away sample the counted movements (all, near another fly, away from others).
Besides the verdict, a review says whether the moving leg touches the other fly (touches: yes, no, no_partner = no other fly in the
clip).

Events are not tied to the loaded experiment: the tab plays each fly's HLS movie (motionmapper/<NN>/movie, as the Ethograms tab)
at the time of the event, so a random sample can mix flies of every experiment. Verdicts go to their own sqlite file,
$LEG_ANNOTATIONS_DB (default: leg_annotations.db next to pe_annotations.db), keyed on the fly and the start and end of the event.
"""
import datetime
import io
import os
import os.path
import sqlite3

import numpy as np
import pandas as pd
from flask import Response, jsonify, request, session

from idtrackerai_validator_server.ethogram import FLY_PATTERN, _fly_dir
from idtrackerai_validator_server.pe_validation import PE_DB

LEG_MOVEMENTS_DIR = os.environ.get("LEG_MOVEMENTS_DIR", "/flyhostel_data/sync_paper_data/modelling/leg_movements")
LEG_DB = os.environ.get("LEG_ANNOTATIONS_DB") or os.path.join(os.path.dirname(os.path.abspath(PE_DB)), "leg_annotations.db")

# what the reviewer sees in the clip (keys 1-7 in the tab)
VERDICTS = ("middle", "groom", "walk", "posture", "pose_error", "other", "unsure")
WHICH_LEG = ("left", "right", "both")
TOUCHES = ("yes", "no", "no_partner")
MOVEMENT_COLUMNS = ["p_middle", "movement", "near", "partner"]
MOVEMENT_SELECTIONS = {"movements": None, "movements_near": True, "movements_away": False}
IMMOBILE_S = 10   # "while immobile": immobile in the sleep trace for at least this long when the event starts
EVENT_COLUMNS = ["event", "start", "end", "leg", "strokes", "strokes_per_s", "amplitude", "front_strokes", "hind_strokes",
                 "thorax_displacement", "alone", "immobile_s", "tip_score", "jump_fraction", "clean"]
LEGS = ("mL", "mR", "fL", "fR", "rL", "rR")

_CACHE = {}   # path -> (mtime, DataFrame)


def _read_cached(path, reader):
    mtime = os.path.getmtime(path)
    hit = _CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    df = reader(path)
    _CACHE[path] = (mtime, df)
    return df


def _events_path(fly):
    return os.path.join(LEG_MOVEMENTS_DIR, "events", f"{fly}.csv")


def _traces_path(fly):
    return os.path.join(LEG_MOVEMENTS_DIR, "traces", f"{fly}.feather")


def _read_events(path):
    if os.path.getsize(path) <= 1:     # a fly without events: empty csv
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.read_csv(path)
    if "event" not in df.columns:      # tables written before the detector numbered its events
        df.insert(0, "event", np.arange(len(df)))
    return df


def fly_events(fly):
    return _read_cached(_events_path(fly), _read_events)


def all_flies():
    folder = os.path.join(LEG_MOVEMENTS_DIR, "events")
    if not os.path.isdir(folder):
        return []
    return sorted(name[:-4] for name in os.listdir(folder) if name.endswith(".csv") and FLY_PATTERN.match(name[:-4]))


def has_movie(fly):
    return os.path.exists(os.path.join(_fly_dir(fly), "movie", "movie.m3u8"))


def _key(start, end):
    # events are identified by their fly, start and end (ms precision, as the csv)
    return (round(float(start), 3), round(float(end), 3))


def _init_db():
    verdicts = ",".join(f"'{v}'" for v in VERDICTS)
    legs = ",".join(f"'{v}'" for v in WHICH_LEG)
    with sqlite3.connect(LEG_DB) as c:
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS leg_annotations (
                fly         TEXT    NOT NULL,
                start       REAL    NOT NULL,
                end         REAL    NOT NULL,
                event       INTEGER,
                leg         TEXT,
                verdict     TEXT    CHECK (verdict IS NULL OR verdict IN ({verdicts})),
                which_leg   TEXT    CHECK (which_leg IS NULL OR which_leg IN ({legs})),
                notes       TEXT,
                reviewer    TEXT,
                reviewed_at TEXT    NOT NULL,
                PRIMARY KEY (fly, start, end)
            )""")
        columns = [r[1] for r in c.execute("PRAGMA table_info(leg_annotations)")]
        if "touches" not in columns:
            touches = ",".join(f"'{v}'" for v in TOUCHES)
            c.execute(f"ALTER TABLE leg_annotations ADD COLUMN touches TEXT CHECK (touches IS NULL OR touches IN ({touches}))")


def _annotations(flies=None):
    with sqlite3.connect(LEG_DB) as c:
        c.row_factory = sqlite3.Row
        if flies is None:
            rows = c.execute("SELECT * FROM leg_annotations").fetchall()
        else:
            flies = list(flies)
            rows = []
            for i in range(0, len(flies), 500):
                part = flies[i:i + 500]
                rows += c.execute(f"SELECT * FROM leg_annotations WHERE fly IN ({','.join('?' * len(part))})", part).fetchall()
    return {(r["fly"], *_key(r["start"], r["end"])): dict(r) for r in rows}


def _flag(df, column):
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].astype(str).str.lower() == "true"


def _select(df, which):
    """which: all, or any of clean, alone, immobile joined by _ (alone_immobile, clean_alone_immobile...)"""
    keep = pd.Series(True, index=df.index)
    tokens = set(which.split("_"))
    if "immobile" in tokens:
        keep &= df["immobile_s"].fillna(0) >= IMMOBILE_S
    if "alone" in tokens:
        keep &= _flag(df, "alone")
    if "clean" in tokens:
        keep &= _flag(df, "clean")
    return df.loc[keep]


def _model_queue():
    """The events to review next chosen by leg_movements_model.py (model/queue.csv: fly, event, ...), in the order of the file"""
    path = os.path.join(LEG_MOVEMENTS_DIR, "model", "queue.csv")
    if not os.path.exists(path):
        return None
    q = pd.read_csv(path).reset_index().rename(columns={"index": "_order"})
    parts = [fly_events(f).assign(fly=f).merge(rows[["event", "_order"]], on="event") for f, rows in q.groupby("fly")
             if os.path.exists(_events_path(f))]
    if not parts:
        return pd.DataFrame(columns=["fly"] + EVENT_COLUMNS)
    return pd.concat(parts, ignore_index=True).sort_values("_order").drop(columns="_order")


def _movements():
    """model/movements.csv of leg_movements_per_fly.py (fly, event, p_middle, movement, near, partner), or None"""
    path = os.path.join(LEG_MOVEMENTS_DIR, "model", "movements.csv")
    if not os.path.exists(path):
        return None
    return _read_cached(path, lambda p: pd.read_csv(p, usecols=["fly", "event"] + MOVEMENT_COLUMNS))


def _movement_events(near):
    """The events counted as movements (near: True near another fly, False away, None all), with the detector's columns"""
    m = _movements()
    if m is None:
        return None
    m = m.loc[m["movement"].astype(str).str.lower() == "true"]
    if near is not None:
        m = m.loc[(m["near"].astype(str).str.lower() == "true") == near]
    parts = [fly_events(f).assign(fly=f).merge(rows[["event"]], on="event") for f, rows in m.groupby("fly")
             if os.path.exists(_events_path(f))]
    if not parts:
        return pd.DataFrame(columns=["fly"] + EVENT_COLUMNS)
    return pd.concat(parts, ignore_index=True).sort_values(["fly", "event"]).reset_index(drop=True)


def _with_movements(df):
    """The columns of movements.csv for the events in it (None for the others)"""
    m = _movements()
    if m is None or df.empty:
        return df.assign(**{c: None for c in MOVEMENT_COLUMNS})
    return df.drop(columns=MOVEMENT_COLUMNS, errors="ignore").merge(m, on=["fly", "event"], how="left")


def _records(df):
    df = df.astype(object).where(pd.notna(df), None)
    return df.to_dict("records")


def _reviewer():
    return session.get("user") or os.environ.get("USER") or "anonymous"


def register_leg_validation(app):
    """Attach the /api/legs/* routes to the Flask `app`."""
    _init_db()

    def _check(fly):
        if not fly or not FLY_PATTERN.match(fly):
            return jsonify({"error": f"invalid fly id {fly}"}), 400
        if not os.path.exists(_events_path(fly)):
            return jsonify({"error": f"no leg movement events for {fly} in {LEG_MOVEMENTS_DIR}/events"}), 404
        return None

    @app.route("/api/legs/flies", methods=["GET"])
    def legs_flies():
        """Flies with detector output: events, events while immobile, reviewed events, whether the fly has a movie."""
        flies = all_flies()
        reviewed = {}
        for (fly, _, _), row in _annotations().items():
            if row["verdict"]:
                reviewed[fly] = reviewed.get(fly, 0) + 1
        out = []
        for fly in flies:
            df = fly_events(fly)
            out.append({"fly": fly, "events": int(len(df)), "immobile": int(len(_select(df, "immobile"))),
                        "reviewed": reviewed.get(fly, 0), "has_movie": has_movie(fly),
                        "has_traces": os.path.exists(_traces_path(fly))})
        return jsonify({"directory": LEG_MOVEMENTS_DIR, "database": LEG_DB, "verdicts": VERDICTS,
                        "immobile_s": IMMOBILE_S, "flies": out})

    @app.route("/api/legs/events", methods=["GET"])
    def legs_events():
        """The events to review, with their verdicts.
        ?fly=<fly>: every event of the fly, in time order; or ?sample=N[&seed=S]: the events already reviewed (in the order they
        were reviewed), then N events not reviewed yet, in a random order of the events of every fly (with a movie or not:
        has_movie says) that depends only on the seed and the selection, so a review can be continued later. &which=all, or
        clean, alone, immobile joined by _ (_select)."""
        which = request.args.get("which", "all")
        fly = request.args.get("fly")
        if fly:
            err = _check(fly)
            if err:
                return err
            df = _select(fly_events(fly), which).assign(fly=fly).sort_values("start")
        else:
            n = request.args.get("sample", type=int)
            if not n:
                return jsonify({"error": "pass ?fly= or ?sample="}), 400
            seed = request.args.get("seed", default=0, type=int)
        if not fly and which == "model":
            # the queue of leg_movements_model.py: events where its classifier is least sure, in the order of the file
            df = _model_queue()
            if df is None:
                return jsonify({"error": "no model/queue.csv in LEG_MOVEMENTS_DIR: run leg_movements_model.py"}), 404
        elif not fly:
            if which in MOVEMENT_SELECTIONS:
                # the movements counted by leg_movements_per_fly.py
                df = _movement_events(MOVEMENT_SELECTIONS[which])
                if df is None:
                    return jsonify({"error": "no model/movements.csv in LEG_MOVEMENTS_DIR: run leg_movements_per_fly.py"}), 404
                if df.empty:
                    return jsonify([])
            else:
                # every fly with events, movie or not: the queue must not change when movies are deleted or remade
                flies = all_flies()
                parts = [_select(fly_events(f), which).assign(fly=f) for f in flies]
                parts = [p for p in parts if len(p)]
                if not parts:
                    return jsonify([])
                df = pd.concat(parts, ignore_index=True)
            df = df.sample(frac=1, random_state=seed)
            done = _annotations(df["fly"].unique())
            reviewed_at = pd.Series([done.get((f, *_key(s, e)), {}).get("reviewed_at") for f, s, e in
                                     zip(df["fly"], df["start"], df["end"])], index=df.index)
            df = pd.concat([df.loc[reviewed_at.notna()].assign(_at=reviewed_at).sort_values("_at").drop(columns="_at"),
                            df.loc[reviewed_at.isna()].head(n)])
        notes = _annotations(df["fly"].unique())
        df = _with_movements(df[["fly"] + [c for c in EVENT_COLUMNS if c in df.columns]])
        rows = _records(df)
        movies = {f: has_movie(f) for f in df["fly"].unique()}
        for r in rows:
            a = notes.get((r["fly"], *_key(r["start"], r["end"])), {})
            r.update(verdict=a.get("verdict"), which_leg=a.get("which_leg"), touches=a.get("touches"), notes=a.get("notes") or "",
                     reviewer=a.get("reviewer"), has_movie=movies[r["fly"]])
        return jsonify(rows)

    @app.route("/api/legs/trace", methods=["GET"])
    def legs_trace():
        """Trace of one event (?fly=&event=): t, speed, and per leg its distance from rest and stroke flags, as lists."""
        fly = request.args.get("fly")
        err = _check(fly)
        if err:
            return err
        event = request.args.get("event", type=int)
        path = _traces_path(fly)
        if not os.path.exists(path):
            return jsonify({"error": f"no traces for {fly}: rerun the detector (leg_movements.py) to write {path}"}), 404
        df = _read_cached(path, pd.read_feather)
        df = df.loc[df["event"] == event].sort_values("t")
        if df.empty:
            return jsonify({"error": f"no trace for event {event} of {fly}"}), 404
        out = {"t": df["t"].round(4).tolist(), "speed": df["speed"].astype(float).round(4).tolist()}
        for leg in LEGS:
            out[leg] = [None if not np.isfinite(v) else round(float(v), 4) for v in df[leg]]
            out[f"stroke_{leg}"] = df[f"stroke_{leg}"].astype(int).tolist()
        return jsonify(out)

    @app.route("/api/legs/annotate", methods=["POST"])
    def legs_annotate():
        """{fly, event, start, end, leg, verdict, which_leg, touches, notes}: saves the review of an event; all of verdict,
        which_leg, touches and notes empty deletes it."""
        d = request.get_json(force=True)
        need = ("fly", "start", "end")
        if not all(k in d for k in need):
            return jsonify({"error": f"need {need}"}), 400
        err = _check(d["fly"])
        if err:
            return err
        verdict, which_leg, notes = d.get("verdict") or None, d.get("which_leg") or None, (d.get("notes") or "").strip() or None
        touches = d.get("touches") or None
        if touches is not None and touches not in TOUCHES:
            return jsonify({"error": f"touches must be one of {TOUCHES}"}), 400
        if verdict is not None and verdict not in VERDICTS:
            return jsonify({"error": f"verdict must be one of {VERDICTS}"}), 400
        if which_leg is not None and which_leg not in WHICH_LEG:
            return jsonify({"error": f"which_leg must be one of {WHICH_LEG}"}), 400
        start, end = _key(d["start"], d["end"])
        with sqlite3.connect(LEG_DB) as c:
            if verdict is None and which_leg is None and touches is None and notes is None:
                c.execute("DELETE FROM leg_annotations WHERE fly=? AND start=? AND end=?", (d["fly"], start, end))
                return jsonify({"ok": True, "deleted": True})
            c.execute("""
                INSERT INTO leg_annotations (fly, start, end, event, leg, verdict, which_leg, touches, notes, reviewer, reviewed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(fly, start, end) DO UPDATE SET event=excluded.event, leg=excluded.leg, verdict=excluded.verdict,
                    which_leg=excluded.which_leg, touches=excluded.touches, notes=excluded.notes, reviewer=excluded.reviewer,
                    reviewed_at=excluded.reviewed_at
            """, (d["fly"], start, end, d.get("event"), d.get("leg"), verdict, which_leg, touches, notes, _reviewer(),
                  datetime.datetime.utcnow().isoformat()))
        return jsonify({"ok": True, "reviewer": _reviewer()})

    @app.route("/api/legs/export", methods=["GET"])
    def legs_export():
        """Every review as CSV, with the detector's columns of the event (empty if the event is no longer in its output)."""
        notes = pd.DataFrame(list(_annotations().values()))
        if notes.empty:
            notes = pd.DataFrame(columns=["fly", "start", "end", "event", "leg", "verdict", "which_leg", "notes", "reviewer",
                                          "reviewed_at", "touches"])
        detector = []
        for fly in notes["fly"].unique():
            if os.path.exists(_events_path(fly)):
                ev = fly_events(fly).drop(columns=["event", "leg"], errors="ignore").assign(fly=fly)
                ev["start"], ev["end"] = ev["start"].round(3), ev["end"].round(3)
                detector.append(ev)
        if detector:
            notes = notes.merge(pd.concat(detector, ignore_index=True), on=["fly", "start", "end"], how="left")
        buffer = io.StringIO()
        notes.sort_values(["fly", "start"]).to_csv(buffer, index=False)
        return Response(buffer.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=leg_annotations.csv"})
