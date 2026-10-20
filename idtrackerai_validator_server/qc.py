"""QC tab: frames of the loaded experiment that do not track the expected number of flies.

    GET  /api/qc    last QC of the loaded experiment and tracking table (null if none
                    yet, or the database changed since), and the job computing it
    POST /api/qc    start that job (one per database and table at a time)

A frame is ok when its IDENTITY (or IDENTITY_VAL) rows are exactly the N flies of the
experiment, each with its own non-zero identity. With a single fly, identity 0 is the
fly (flyhostel's convention, `get_identities` -> [0]): one row of any identity is ok,
and bad frames are only `not_detected` or `other`. Every frame of STORE_INDEX between
the first and last chunk with tracking is checked, so frames without any row count
as bad (`not_detected`). Bad frames are split into
    not_detected  fewer rows than flies, none of them with identity 0
    unidentified  some fly missing and a row with identity 0 (detected, identity lost)
    other         extra rows or an identity seen twice

The concatenation links the local identities of the last frame of a chunk with those
of the first frame of the next one: when a fly is not tracked there, the following
chunks can lose its identity (identity 0 in every frame) or their rows. `first` / `last` of each
chunk (flies identified on its first / last frame) and its CONCATENATION rows
(`broken`, `inferred`) show where that happened.

Reading takes ~0.1 s per chunk (one grouped query on the frame_number index), so the
QC runs as a job, and is kept in memory and in $QC_CACHE_DIR (one JSON per database
and table, valid while the database's mtime is unchanged).
"""
import json
import logging
import os
import sqlite3
import time
from threading import Lock

import numpy as np
from flask import jsonify

from idtrackerai_validator_server.experiments import to_flat
from idtrackerai_validator_server.jobs import start_job, _jobs

logger = logging.getLogger(__name__)

CACHE_DIR = os.environ.get("QC_CACHE_DIR", os.path.expanduser("~/.cache/idtrackerai-validator/qc"))
BINS_PER_CHUNK = 30         # timeline resolution (10 s bins for 45000-frame chunks at 150 fps)
TOP_RUNS = 200              # longest runs of bad frames listed
# Run lengths (frames) are counted in these bins: [lo, hi)
RUN_BINS_SECONDS = [(0, 1), (1, 10), (10, 60), (60, 600), (600, None)]
CACHE_VERSION = 2          # bump when the rules change: older caches are recomputed

_cache = {}                 # (dbfile, table, n) -> result
_running = {}               # (dbfile, table, n) -> job id
_lock = Lock()


def _connect(dbfile):
    return sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True, check_same_thread=False)


def _cache_path(experiment, table, n_animals):
    return os.path.join(CACHE_DIR, f"{to_flat(experiment)}__{table}__{n_animals}.json")


def _tracked_chunks(con, table, chunksize):
    first = con.execute(f'SELECT frame_number FROM "{table}" ORDER BY frame_number LIMIT 1').fetchone()
    last = con.execute(f'SELECT frame_number FROM "{table}" ORDER BY frame_number DESC LIMIT 1').fetchone()
    if first is None:
        return None
    return int(first[0]) // chunksize, int(last[0]) // chunksize


def _store_frames(con, chunk, chunksize):
    """Frame numbers recorded in `chunk` (STORE_INDEX), else the whole chunk."""
    try:
        rows = con.execute("SELECT frame_number FROM STORE_INDEX WHERE chunk = ? ORDER BY frame_number", (chunk,)).fetchall()
    except sqlite3.Error:
        rows = []
    if rows:
        return np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    return np.arange(chunk * chunksize, (chunk + 1) * chunksize, dtype=np.int64)


def _concatenation(con, table):
    """chunk -> {rows, broken, inferred} of the CONCATENATION table matching `table`
    (CONCATENATION_VAL has no is_broken / is_inferred)."""
    name = table.replace("IDENTITY", "CONCATENATION")
    try:
        columns = {row[1] for row in con.execute(f'PRAGMA table_info("{name}")')}
        if not columns:
            return name, {}
        broken = "SUM(is_broken)" if "is_broken" in columns else "NULL"
        inferred = "SUM(is_inferred)" if "is_inferred" in columns else "NULL"
        rows = con.execute(f'SELECT chunk, COUNT(*), {broken}, {inferred} FROM "{name}" GROUP BY chunk').fetchall()
    except sqlite3.Error as error:
        logger.warning("Could not read %s: %s", name, error)
        return name, {}
    return name, {int(c): {"rows": n, "broken": b, "inferred": i} for c, n, b, i in rows}


def _runs(bad):
    """(start, end) indices (end exclusive) of the runs of True in `bad`."""
    if not bad.any():
        return np.empty((0, 2), dtype=np.int64)
    edges = np.diff(np.concatenate([[0], bad.view(np.int8), [0]]))
    return np.stack([np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)], axis=1)


def compute_qc(dbfile, table, n_animals, chunksize, framerate, progress=None):
    con = _connect(dbfile)
    try:
        chunks = _tracked_chunks(con, table, chunksize)
        if chunks is None:
            raise RuntimeError(f"{table} has no rows")
        concatenation_table, concatenation = _concatenation(con, table)
        first_chunk, last_chunk = chunks
        bin_size = max(1, chunksize // BINS_PER_CHUNK)
        query = (f'SELECT frame_number, COUNT(*), COUNT(DISTINCT NULLIF(identity, 0)), '
                 f'SUM(identity = 0 OR identity IS NULL) FROM "{table}" '
                 f'WHERE frame_number BETWEEN ? AND ? GROUP BY frame_number')

        totals = {"frames": 0, "ok": 0, "not_detected": 0, "unidentified": 0, "other": 0}
        hist = np.zeros(n_animals + 1, dtype=np.int64)
        per_chunk = []
        runs = []               # [start frame, end frame (inclusive), length, min flies identified]
        open_run = None         # run reaching the end of the previous chunk

        all_chunks = range(first_chunk, last_chunk + 1)
        for k, chunk in enumerate(all_chunks):
            frames = _store_frames(con, chunk, chunksize)
            rows = con.execute(query, (int(frames[0]), int(frames[-1]))).fetchall()
            n_rows = np.zeros(len(frames), dtype=np.int64)
            n_ids = np.zeros(len(frames), dtype=np.int64)
            n_zero = np.zeros(len(frames), dtype=np.int64)
            if rows:
                data = np.array(rows, dtype=np.int64)
                pos = np.searchsorted(frames, data[:, 0])
                keep = (pos < len(frames)) & (frames[np.minimum(pos, len(frames) - 1)] == data[:, 0])
                n_rows[pos[keep]], n_ids[pos[keep]], n_zero[pos[keep]] = data[keep, 1], data[keep, 2], data[keep, 3]
            if n_animals == 1:
                # identity 0 is the fly: every row identifies it
                n_ids, n_zero = np.minimum(n_rows, 1), np.zeros_like(n_rows)

            ok = (n_ids == n_animals) & (n_rows == n_animals)
            missing = n_ids < n_animals
            unidentified = missing & (n_zero > 0)
            not_detected = missing & (n_zero == 0)
            other = ~ok & ~missing
            bad = ~ok

            chunk_hist = np.bincount(np.minimum(n_ids, n_animals), minlength=n_animals + 1)
            hist += chunk_hist
            bins = np.minimum((frames - chunk * chunksize) // bin_size, BINS_PER_CHUNK - 1)
            bad_bins = np.bincount(bins, weights=bad, minlength=BINS_PER_CHUNK)
            all_bins = np.bincount(bins, minlength=BINS_PER_CHUNK)

            # Runs of bad frames, joined across chunks when the frames are consecutive
            chunk_runs = _runs(bad)
            continued = None
            if open_run is not None:
                if len(chunk_runs) and chunk_runs[0][0] == 0 and open_run[1] + 1 == frames[0]:
                    continued = open_run
                else:
                    runs.append(open_run)
                open_run = None
            for i, (start, end) in enumerate(chunk_runs):
                run = [int(frames[start]), int(frames[end - 1]), int(end - start), int(n_ids[start:end].min())]
                if i == 0 and continued is not None:
                    continued[1], continued[2], continued[3] = run[1], continued[2] + run[2], min(continued[3], run[3])
                    run = continued
                if end == len(frames):
                    open_run = run
                else:
                    runs.append(run)
            longest = int((chunk_runs[:, 1] - chunk_runs[:, 0]).max()) if len(chunk_runs) else 0

            counts = {"frames": int(len(frames)), "ok": int(ok.sum()), "not_detected": int(not_detected.sum()),
                      "unidentified": int(unidentified.sum()), "other": int(other.sum())}
            for key, value in counts.items():
                totals[key] += value
            per_chunk.append({
                "chunk": chunk, **counts,
                "hist": chunk_hist.tolist(),
                "first": int(n_ids[0]), "last": int(n_ids[-1]),
                "longest_run": longest,
                "first_bad": int(frames[np.argmax(bad)]) if bad.any() else None,
                "concatenation": concatenation.get(chunk),
                # % of bad frames per bin of the chunk (None: no frame in the bin)
                "bins": [None if a == 0 else int(round(100 * b / a)) for b, a in zip(bad_bins, all_bins)],
            })
            if progress:
                progress((k + 1) / len(all_chunks))
        if open_run is not None:
            runs.append(open_run)
    finally:
        con.close()

    # A chunk boundary is broken when a fly is not identified on either side of it
    boundaries = [{"chunk": a["chunk"], "next": b["chunk"], "last": a["last"], "first": b["first"]}
                  for a, b in zip(per_chunk, per_chunk[1:]) if a["last"] < n_animals or b["first"] < n_animals]

    lengths = np.array([r[2] for r in runs], dtype=np.int64)
    run_bins = []
    for lo, hi in RUN_BINS_SECONDS:
        sel = (lengths >= lo * framerate) & ((lengths < hi * framerate) if hi is not None else True)
        run_bins.append({"lo": lo, "hi": hi, "runs": int(sel.sum()), "frames": int(lengths[sel].sum())})
    order = np.argsort(-lengths, kind="stable")[:TOP_RUNS]

    return {
        "version": CACHE_VERSION,
        "table": table,
        "concatenation_table": concatenation_table,
        "n_animals": n_animals,
        "chunksize": chunksize,
        "framerate": framerate,
        "bin_size": bin_size,
        "bins_per_chunk": BINS_PER_CHUNK,
        "first_chunk": first_chunk,
        "last_chunk": last_chunk,
        "totals": totals,
        "hist": hist.tolist(),
        "chunks": per_chunk,
        "boundaries": boundaries,
        "n_runs": len(runs),
        "run_bins": run_bins,
        "runs": [runs[i] for i in order],
    }


def _read_cache(path, mtime):
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if data.get("db_mtime") != mtime or data.get("version") != CACHE_VERSION:
        return None
    return data


def _write_cache(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(data, handle)
        os.replace(tmp, path)
    except OSError as error:
        logger.warning("Could not keep the QC in %s: %s", path, error)


def register_qc(app, get_state):
    """get_state() -> None (no database loaded) or a dict with experiment, dbfile,
    table (IDENTITY or IDENTITY_VAL), n_animals, chunksize, framerate."""

    def current():
        state = get_state()
        if state is None:
            return None, None, None
        key = (state["dbfile"], state["table"], state["n_animals"])
        mtime = os.path.getmtime(state["dbfile"])
        return state, key, mtime

    def cached(state, key, mtime):
        with _lock:
            result = _cache.get(key)
        if result is not None and result["db_mtime"] == mtime:
            return result
        result = _read_cache(_cache_path(state["experiment"], state["table"], state["n_animals"]), mtime)
        if result is not None:
            with _lock:
                _cache[key] = result
        return result

    def running_job(key):
        with _lock:
            job_id = _running.get(key)
        job = _jobs.get(job_id) if job_id else None
        return job if job is not None and job.status == "running" else None

    @app.route("/api/qc", methods=["GET"])
    def get_qc():
        state, key, mtime = current()
        if state is None:
            return jsonify({"error": "Load an experiment with a database first"}), 400
        job = running_job(key)
        return jsonify({
            "experiment": state["experiment"], "table": state["table"], "n_animals": state["n_animals"],
            "result": cached(state, key, mtime),
            "job": job.to_dict() if job else None,
        })

    @app.route("/api/qc", methods=["POST"])
    def start_qc():
        state, key, mtime = current()
        if state is None:
            return jsonify({"error": "Load an experiment with a database first"}), 400
        job = running_job(key)
        if job is not None:
            return jsonify(job.to_dict())

        def work(job):
            def progress(p):
                job.update(0, progress=p)
            job.update(0, status="running")
            result = compute_qc(state["dbfile"], state["table"], state["n_animals"],
                                state["chunksize"], state["framerate"], progress)
            result.update(experiment=state["experiment"], db_mtime=mtime, computed_at=time.time())
            with _lock:
                _cache[key] = result
            _write_cache(_cache_path(state["experiment"], state["table"], state["n_animals"]), result)
            job.update(0, status="done", progress=1.0)
            totals = result["totals"]
            job.result = {"frames": totals["frames"], "ok": totals["ok"]}

        job = start_job(f"QC {state['experiment']} ({state['table']})", [state["experiment"]], work)
        with _lock:
            _running[key] = job.id
        return jsonify(job.to_dict())
