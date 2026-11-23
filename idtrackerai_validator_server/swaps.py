"""Identity swap check of the QC tab: places where a fly's identity may have jumped onto another fly's track.

    GET  /api/qc/swaps?source=machine|validated      last result (null if none yet / the database changed) + job
    POST /api/qc/swaps  {"source", "params"}         start the job (returns the running one if any)
    GET  /api/qc/swaps.csv?source=                   the events as CSV

Nothing is corrected: a swap-like move can be real (a fly running or jumping), the events are a list to look at.

Tracking read: `machine` = ROI_0 + IDENTITY, flies keyed by their local identity in the chunk (the machine tables'
`identity` column is mostly 0 before the integration), carried across a chunk boundary by CONCATENATION
(`local_identity` of chunk c -> `local_identity_after` in c + 1; a fly without link starts afresh); `validated` =
ROI_0_VAL + IDENTITY_VAL, keyed by `identity`. Rows without identity (0) and identities seen twice in a frame are
left out. ROI_0 and IDENTITY are read with one query each per chunk and joined in numpy (~0.3 s per chunk cold; the
SQL join walked the frame_number index at ~3 s per chunk).

For every fly A detected at frame t, with its last position pA at frame fA (gap = t - fA <= `max_gap_s`, so a fly
missing for a few frames, e.g. during a crossing, is compared with where it was before), and every other fly B with
its last known position pB as of fA (where B was when A was last seen):
    d_own   = |A(t) - pA|      how far A's identity jumped
    d_other = |A(t) - pB|      how far A(t) is from where B was
A transition is a candidate when d_own >= jump_px, d_other <= near_px and d_other <= ratio * d_own (closer to B's
place than to its own, clearly). jump_px = max(jump_bl body lengths, step_factor x the 99th percentile of the
chunk's one-frame displacements shorter than jump_bl): a fly sitting still next to another (centroid jitter) never gets there, walking
flies need a jump several times their usual step. Body length in px = body_mm x METADATA pixels_per_cm / 10.
Types:
    exchange   A took B's place and B took A's (B's own candidate with partner A within pair_s): two identities
               swapped, the case of (a)
    takeover   A took B's place, B is not there any more (missing, or moved away without taking A's): (b)
    onto       A landed on B's place while B is still detected there (a fly split in two blobs, a missed fly...)
Candidates of the same pair of flies less than merge_s apart are one event; `n_transitions` counts them and
`net_swap` is true when an odd number of exchanges happened (an even number = identities flickered between two flies
touching and came back). `encounter` = the two flies were within encounter_bl body lengths before (the classic
crossing). Score (0-1) = closeness (1 - d_other / near_px) x jump size (d_own / 2 jump_px, at most 1) x gap factor
(1 up to 2 frames, then exp(-gap / gap_tau_s)), times TYPE_WEIGHT; flickers (no net swap) x FLICKER_WEIGHT.

Kept in memory and in $QC_CACHE_DIR/swaps/ (per experiment, source and parameters) while the database's mtime and
CACHE_VERSION are unchanged; the job reports per chunk.
"""
import csv
import io
import json
import logging
import os
import sqlite3
import time
from threading import Lock

import numpy as np
from flask import Response, jsonify, request

from idtrackerai_validator_server.experiments import to_flat
from idtrackerai_validator_server.jobs import start_job, _jobs

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.environ.get("QC_CACHE_DIR", os.path.expanduser("~/.cache/idtrackerai-validator/qc")), "swaps")
CACHE_VERSION = 1
DEFAULT_PIXELS_PER_CM = 333

DEFAULTS = {
    "body_mm": 2.5,         # fly body length used as unit
    "jump_bl": 0.35,        # an identity moving less than this (body lengths) is never a candidate
    "step_factor": 3.0,     # ... nor less than this x the chunk's 99th percentile one-frame step
    "near_bl": 0.5,         # A(t) at most this far from B's last place
    "near_frac": 0.15,      # ... or this fraction of A's jump, when larger (a jump across the arena)
    "unidentified": 1.0,    # 1: detections without identity count as other flies (B = 0)
    "ratio": 0.7,           # ... and at most this fraction of A's own jump
    "max_gap_s": 2.0,       # a fly unseen for longer is not compared with its old place
    "pair_s": 0.5,          # B's own candidate with partner A within this: exchange
    "merge_s": 1.0,         # candidates of a pair closer than this are one event
    "encounter_bl": 1.5,    # flies this close before the jump: close encounter
    "gap_tau_s": 1.0,       # score decay with the time the fly was not seen
}
TYPE_WEIGHT = {"exchange": 1.0, "takeover": 0.8, "onto": 0.5}
FLICKER_WEIGHT = 0.5
SOURCES = {"machine": ("ROI_0", "IDENTITY", "local_identity"),
           "validated": ("ROI_0_VAL", "IDENTITY_VAL", "identity")}

_cache = {}          # (dbfile, source, params key) -> result
_running = {}        # (dbfile, source) -> job id
_lock = Lock()


def params_with(overrides=None):
    params = dict(DEFAULTS)
    for key, value in (overrides or {}).items():
        if key in DEFAULTS and value is not None:
            params[key] = float(value)
    return params


def _params_key(params):
    return json.dumps({k: params[k] for k in sorted(DEFAULTS)}, sort_keys=True)


# --- the detector, on the rows of one chunk -------------------------------------------------------------------
def detect_block(frames, keys, xs, ys, f0, f1, framerate, body_px, params, carry=None):
    """Candidates of rows (frame, key > 0, x, y) of frames [f0, f1].

    carry: {key: (x, y, frame)} last positions before f0 (keys already those of this block).
    Returns (candidates, carry_out, jump_px); candidates are dicts sorted by frame."""
    carry = carry or {}
    frames = np.asarray(frames, dtype=np.int64)
    keys = np.asarray(keys, dtype=np.int64)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    inside = (frames >= f0) & (frames <= f1)
    unid = inside & (keys == 0)
    u_order = np.argsort(frames[unid], kind="stable")
    uf, ux, uy = frames[unid][u_order], xs[unid][u_order], ys[unid][u_order]
    sel = inside & (keys > 0)
    frames, keys, xs, ys = frames[sel], keys[sel], xs[sel], ys[sel]
    all_keys = np.union1d(np.unique(keys), np.array(sorted(carry), dtype=np.int64))
    n = len(all_keys)
    T = int(f1 - f0 + 1)
    if n < 1 or T < 1 or (n < 2 and not len(uf)):
        return [], _carry_out_simple(frames, keys, xs, ys, carry), None
    col = np.searchsorted(all_keys, keys)
    row = (frames - f0 + 1).astype(np.int64)

    # an identity on two rows of a frame: neither is used
    flat = row * n + col
    uniq, counts = np.unique(flat, return_counts=True)
    dup = np.isin(flat, uniq[counts > 1])
    row, col, xs_, ys_, fr = row[~dup], col[~dup], xs[~dup], ys[~dup], frames[~dup]

    P = np.full((T + 1, n, 2), np.nan)
    S = np.full((T + 1, n), np.iinfo(np.int64).min // 2, dtype=np.int64)
    P[row, col, 0], P[row, col, 1], S[row, col] = xs_, ys_, fr
    for key, (x, y, f) in carry.items():
        c = int(np.searchsorted(all_keys, key))
        P[0, c] = (x, y)
        S[0, c] = f
    present = ~np.isnan(P[:, :, 0])
    idx = np.where(present, np.arange(T + 1)[:, None], 0)
    idx = np.maximum.accumulate(idx, axis=0)
    cols = np.arange(n)

    max_gap = params["max_gap_s"] * framerate
    near = params["near_bl"] * body_px

    # one-frame steps -> jump threshold
    both = present[2:] & present[1:-1]
    steps = np.hypot(*(P[2:] - P[1:-1])[both].T) if both.any() else np.empty(0)
    steps = steps[steps < params["jump_bl"] * body_px]       # identity jumps left out
    p99 = float(np.percentile(steps, 99)) if len(steps) >= 100 else 0.0
    jump = max(params["jump_bl"] * body_px, params["step_factor"] * p99)

    tt = np.arange(1, T + 1)
    r = idx[:-1]                                      # (T, n): row of each fly's last position before t
    pA = P[r, cols]                                   # (T, n, 2)
    fA = S[r, cols]
    seen_before = present[r, cols]
    new = P[1:]
    gap = S[1:] - fA
    d_own = np.hypot(new[..., 0] - pA[..., 0], new[..., 1] - pA[..., 1])
    with np.errstate(invalid="ignore"):
        mask = present[1:] & seen_before & (gap <= max_gap) & (d_own >= jump)
    ti, ai = np.nonzero(mask)
    candidates = []
    if len(ti):
        t = tt[ti]                                    # rows of the candidates
        rr = r[ti, ai]                                # row of A's last position
        ridx = idx[rr]                                # (M, n) rows of every fly's last position as of then
        pB = P[ridx, cols]                            # (M, n, 2)
        fB = S[ridx, cols]
        validB = present[ridx, cols] & ((S[rr, ai][:, None] - fB) <= max_gap)
        validB[np.arange(len(ti)), ai] = False
        newA = new[ti, ai]                            # (M, 2)
        d_other = np.hypot(newA[:, None, 0] - pB[..., 0], newA[:, None, 1] - pB[..., 1])
        d_other[~validB] = np.inf
        bi = np.argmin(d_other, axis=1)
        dmin = d_other[np.arange(len(ti)), bi]
        dA = d_own[ti, ai]
        near_m = np.maximum(near, params["near_frac"] * dA)     # a long jump may land a bit further off
        for m in range(len(ti)):
            a, row_t = int(ai[m]), int(t[m])
            pa = pA[ti[m], a]
            frame = int(f0 + row_t - 1)
            b, pb, b_key = int(bi[m]), pB[m, bi[m]], int(all_keys[bi[m]])
            d_b = float(dmin[m])
            # a detection without identity where A was last seen, not A itself: "another fly" too
            fa = int(S[rr[m], a])
            lo, hi = np.searchsorted(uf, fa), np.searchsorted(uf, fa, side="right")
            if params["unidentified"] and hi > lo:
                du = np.hypot(ux[lo:hi] - newA[m, 0], uy[lo:hi] - newA[m, 1])
                j = int(np.argmin(du))
                pu = np.array([ux[lo + j], uy[lo + j]])
                if du[j] < d_b and np.hypot(*(pu - pa)) >= jump:
                    b, pb, b_key, d_b = None, pu, 0, float(du[j])
            if not (d_b <= near_m[m] and d_b <= params["ratio"] * dA[m]):
                continue
            if b is None:
                lo, hi = np.searchsorted(uf, frame), np.searchsorted(uf, frame, side="right")
                b_stays = bool(hi > lo and np.hypot(ux[lo:hi] - pb[0], uy[lo:hi] - pb[1]).min() <= near)
            else:
                b_now = P[row_t, b]
                b_stays = bool(present[row_t, b] and np.hypot(*(b_now - pb)) <= near)
            g = int(gap[ti[m], a])
            candidates.append({
                "frame": frame, "a": int(all_keys[a]), "b": b_key,
                "d_own": float(dA[m]), "d_other": d_b, "gap": g,
                "d_before": float(np.hypot(*(pa - pb))), "b_stays": b_stays,
                "a_from": [round(float(pa[0]), 1), round(float(pa[1]), 1)],
                "a_to": [round(float(newA[m, 0]), 1), round(float(newA[m, 1]), 1)],
                "b_from": [round(float(pb[0]), 1), round(float(pb[1]), 1)],
            })
    last = idx[-1]
    carry_out = {int(all_keys[c]): (float(P[last[c], c, 0]), float(P[last[c], c, 1]), int(S[last[c], c]))
                 for c in range(n) if present[last[c], c]}
    return candidates, carry_out, jump


def _carry_out_simple(frames, keys, xs, ys, carry):
    out = dict(carry)
    for f, k, x, y in zip(frames, keys, xs, ys):
        out[int(k)] = (float(x), float(y), int(f))
    return out


def _score(c, jump, near, framerate, params):
    closeness = max(0.0, 1 - c["d_other"] / near) if near > 0 else 0.0
    size = min(1.0, c["d_own"] / (2 * jump)) if jump else 1.0
    gap_s = c["gap"] / framerate
    gap_factor = 1.0 if c["gap"] <= 2 else float(np.exp(-gap_s / params["gap_tau_s"]))
    return closeness * size * gap_factor


def events_from_candidates(candidates, framerate, body_px, params):
    """Pair exchanges, merge candidates of a pair into events. Candidates carry `jump` and `chunk`."""
    near = params["near_bl"] * body_px
    pair_frames = params["pair_s"] * framerate
    merge_frames = params["merge_s"] * framerate
    candidates = sorted(candidates, key=lambda c: (c["chunk"], c["frame"]))
    by_pair = {}
    for c in candidates:
        by_pair.setdefault((c["chunk"], c["a"], c["b"]), []).append(c)
    for c in candidates:
        c["score"] = _score(c, c["jump"], near, framerate, params)
        partners = by_pair.get((c["chunk"], c["b"], c["a"]), [])
        c["exchange"] = any(abs(p["frame"] - c["frame"]) <= pair_frames for p in partners)
        c["type"] = "exchange" if c["exchange"] else ("onto" if c["b_stays"] else "takeover")

    groups = {}
    for c in candidates:
        groups.setdefault((c["chunk"],) + tuple(sorted((c["a"], c["b"]))), []).append(c)
    events = []
    for (chunk, f1, f2), cs in groups.items():
        current = [cs[0]]
        for c in cs[1:]:
            if c["frame"] - current[-1]["frame"] <= merge_frames:
                current.append(c)
            else:
                events.append(_event(chunk, (f1, f2), current, body_px, params))
                current = [c]
        events.append(_event(chunk, (f1, f2), current, body_px, params))
    events.sort(key=lambda e: -e["score"])
    return events


def _event(chunk, flies, cs, body_px, params):
    types = {c["type"] for c in cs}
    kind = "exchange" if "exchange" in types else ("takeover" if "takeover" in types else "onto")
    # an exchange is two candidates (A onto B's place, B onto A's, maybe frames apart): count one side's
    exchange_frames = sorted({c["frame"] for c in cs if c["type"] == "exchange" and c["a"] == flies[0]})
    n_swaps = 0
    last = None
    for f in exchange_frames:
        if last is None or f - last > 1:
            n_swaps += 1
        last = f
    net_swap = n_swaps % 2 == 1 if kind == "exchange" else True
    best = max(cs, key=lambda c: c["score"])
    score = max(c["score"] for c in cs) * TYPE_WEIGHT[kind]
    if not net_swap:
        score *= FLICKER_WEIGHT
    encounter = best["d_before"] <= params["encounter_bl"] * body_px
    reasons = {
        "exchange": "flies {a} and {b} exchanged places",
        "takeover": "fly {a} jumped to where {b} was",
        "onto": "fly {a} jumped onto {b}, still there",
    }
    other = f"fly {best['b']}" if best["b"] else "a fly without identity"
    reason = reasons[kind].format(a=best["a"], b=other if kind != "exchange" else best["b"])
    if kind == "exchange" and not net_swap:
        reason += f" and back ({n_swaps} swaps)"
    if best["gap"] > 1:
        reason += f" after {best['gap']} frames unseen"
    if encounter:
        reason += ", during a close encounter"
    return {
        "chunk": chunk, "start": min(c["frame"] for c in cs), "end": max(c["frame"] for c in cs),
        "frame": best["frame"], "flies": list(flies), "a": best["a"], "b": best["b"],
        "type": kind, "score": round(float(score), 3), "n_transitions": len(cs), "n_swaps": n_swaps,
        "net_swap": net_swap, "encounter": bool(encounter), "reason": reason,
        "d_own": round(best["d_own"], 1), "d_other": round(best["d_other"], 1),
        "d_before": round(best["d_before"], 1), "gap": best["gap"], "jump_px": round(best["jump"], 1),
        "a_from": best["a_from"], "a_to": best["a_to"], "b_from": best["b_from"],
    }


# --- reading the database ---------------------------------------------------------------------------------------
def _connect(dbfile):
    return sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True, check_same_thread=False)


def has_tables(con, source):
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    roi, ident, _ = SOURCES[source]
    return roi in names and ident in names


def available_sources(dbfile):
    con = _connect(dbfile)
    try:
        return [s for s in SOURCES if has_tables(con, s)]
    finally:
        con.close()


def _pixels_per_cm(con):
    try:
        row = con.execute("SELECT value FROM METADATA WHERE field = 'pixels_per_cm'").fetchone()
        return float(row[0]) if row and row[0] else DEFAULT_PIXELS_PER_CM
    except (sqlite3.Error, ValueError):
        return DEFAULT_PIXELS_PER_CM


def read_chunk(con, source, first, last):
    """(frames, keys, x, y) of [first, last]: ROI_0 and IDENTITY read separately, joined on (frame, in_frame_index)."""
    roi, ident, key_column = SOURCES[source]
    r = np.array(con.execute(f'SELECT frame_number, in_frame_index, x, y FROM "{roi}" '
                             'WHERE frame_number BETWEEN ? AND ?', (first, last)).fetchall(), dtype=float)
    i = np.array(con.execute(f'SELECT frame_number, in_frame_index, {key_column} FROM "{ident}" '
                             'WHERE frame_number BETWEEN ? AND ?', (first, last)).fetchall(), dtype=float)
    if not len(r) or not len(i):
        return (np.empty(0, np.int64),) * 2 + (np.empty(0),) * 2
    i = i[~np.isnan(i).any(axis=1)]
    rk = r[:, 0].astype(np.int64) * 1024 + r[:, 1].astype(np.int64)
    ik = i[:, 0].astype(np.int64) * 1024 + i[:, 1].astype(np.int64)
    order = np.argsort(ik, kind="stable")
    ik, ivals = ik[order], i[order, 2]
    pos = np.searchsorted(ik, rk)
    ok = pos < len(ik)
    ok[ok] = ik[pos[ok]] == rk[ok]
    keys = np.zeros(len(r), dtype=np.int64)
    keys[ok] = ivals[pos[ok]].astype(np.int64)
    return r[:, 0].astype(np.int64), keys, r[:, 2], r[:, 3]


def _links(con):
    """chunk -> {local_identity: local_identity_after} of the machine CONCATENATION."""
    out = {}
    try:
        for chunk, lid, after in con.execute(
                "SELECT chunk, local_identity, local_identity_after FROM CONCATENATION"):
            if lid and after:
                out.setdefault(int(chunk), {})[int(lid)] = int(after)
    except sqlite3.Error as error:
        logger.warning("No CONCATENATION: %s", error)
    return out


def _tracked_range(con, roi):
    first = con.execute(f'SELECT MIN(frame_number) FROM "{roi}"').fetchone()[0]
    last = con.execute(f'SELECT MAX(frame_number) FROM "{roi}"').fetchone()[0]
    return (None, None) if first is None else (int(first), int(last))


def compute_swaps(dbfile, source, n_animals, chunksize, framerate, params=None, progress=None, chunks=None):
    params = params_with(params)
    started = time.time()
    con = _connect(dbfile)
    try:
        if not has_tables(con, source):
            raise RuntimeError(f"No {source} tracking tables in {dbfile}")
        body_px = params["body_mm"] * _pixels_per_cm(con) / 10
        first, last = _tracked_range(con, SOURCES[source][0])
        if first is None:
            raise RuntimeError(f"{SOURCES[source][0]} has no rows")
        all_chunks = list(range(first // chunksize, last // chunksize + 1))
        if chunks is not None:
            all_chunks = [c for c in all_chunks if c in set(chunks)]
        links = _links(con) if source == "machine" else {}
        candidates, per_chunk, carry, prev_chunk = [], [], {}, None
        for k, chunk in enumerate(all_chunks):
            f0, f1 = max(first, chunk * chunksize), min(last, (chunk + 1) * chunksize - 1)
            frames, keys, xs, ys = read_chunk(con, source, f0, f1)
            if prev_chunk != chunk - 1:
                carry = {}
            elif source == "machine":
                link = links.get(chunk - 1, {})
                carry = {link[k_]: v for k_, v in carry.items() if k_ in link}
            found, carry, jump = detect_block(frames, keys, xs, ys, f0, f1, framerate, body_px, params, carry)
            for c in found:
                c["chunk"], c["jump"] = chunk, jump
            candidates += found
            per_chunk.append({"chunk": chunk, "candidates": len(found),
                              "jump_px": None if jump is None else round(jump, 1)})
            prev_chunk = chunk
            if progress:
                progress((k + 1) / len(all_chunks))
    finally:
        con.close()
    events = events_from_candidates(candidates, framerate, body_px, params)
    return {
        "version": CACHE_VERSION, "source": source, "table": SOURCES[source][1],
        "key": SOURCES[source][2], "n_animals": n_animals, "chunksize": chunksize, "framerate": framerate,
        "params": params, "body_px": round(body_px, 1), "near_px": round(params["near_bl"] * body_px, 1),
        "first_chunk": all_chunks[0] if all_chunks else None, "last_chunk": all_chunks[-1] if all_chunks else None,
        "chunks": per_chunk, "n_candidates": len(candidates),
        "counts": {t: sum(e["type"] == t for e in events) for t in TYPE_WEIGHT},
        "events": events, "seconds": round(time.time() - started, 1),
    }


CSV_COLUMNS = ["start", "end", "frame", "clock", "chunk", "a", "b", "type", "score", "net_swap", "n_swaps",
               "n_transitions", "encounter", "d_own", "d_other", "d_before", "gap", "jump_px", "reason"]


def to_csv(result):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(CSV_COLUMNS[:5] + ["fly_a_" + result["key"], "fly_b_" + result["key"]] + CSV_COLUMNS[7:])
    for e in sorted(result["events"], key=lambda e: e["start"]):
        writer.writerow([e.get(c) for c in CSV_COLUMNS])
    return out.getvalue()


# --- cache and routes ---------------------------------------------------------------------------------------------
def _cache_path(experiment, source, params):
    import hashlib
    digest = hashlib.sha1(_params_key(params).encode()).hexdigest()[:10]
    return os.path.join(CACHE_DIR, f"{to_flat(experiment)}__{source}__{digest}.json")


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
        logger.warning("Could not keep the swap check in %s: %s", path, error)


def register_swaps(app, get_state):
    """get_state() -> None or a dict with experiment, dbfile, n_animals, chunksize, framerate, use_val."""
    _last_params = {}       # (dbfile, source) -> params of the last run started

    def current(source=None):
        state = get_state()
        if state is None:
            return None, None, "Load an experiment with a database first"
        sources = available_sources(state["dbfile"])
        if source is None:
            source = "validated" if state.get("use_val") and "validated" in sources else "machine"
        if source not in SOURCES:
            return None, None, f"Unknown source {source}"
        state = dict(state, source=source, sources=sources, mtime=os.path.getmtime(state["dbfile"]))
        return state, (state["dbfile"], source), None

    def cached(state, params):
        key = (state["dbfile"], state["source"], _params_key(params))
        with _lock:
            result = _cache.get(key)
        if result is not None and result["db_mtime"] == state["mtime"]:
            return result
        result = _read_cache(_cache_path(state["experiment"], state["source"], params), state["mtime"])
        if result is not None:
            with _lock:
                _cache[key] = result
        return result

    def running_job(key):
        with _lock:
            job_id = _running.get(key)
        job = _jobs.get(job_id) if job_id else None
        return job if job is not None and job.status == "running" else None

    def params_of(key):
        return _last_params.get(key) or params_with()

    @app.route("/api/qc/swaps", methods=["GET"])
    def get_swaps():
        state, key, error = current(request.args.get("source"))
        if state is None:
            return jsonify({"error": error}), 400
        job = running_job(key)
        return jsonify({
            "experiment": state["experiment"], "source": state["source"], "sources": state["sources"],
            "n_animals": state["n_animals"], "defaults": DEFAULTS, "params": params_of(key),
            "result": cached(state, params_of(key)) if state["source"] in state["sources"] else None,
            "job": job.to_dict() if job else None,
        })

    @app.route("/api/qc/swaps.csv", methods=["GET"])
    def swaps_csv():
        state, key, error = current(request.args.get("source"))
        if state is None:
            return jsonify({"error": error}), 400
        result = cached(state, params_of(key))
        if result is None:
            return jsonify({"error": "No swap check yet"}), 404
        name = f"{to_flat(state['experiment'])}_swaps_{state['source']}.csv"
        return Response(to_csv(result), mimetype="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.route("/api/qc/swaps", methods=["POST"])
    def start_swaps():
        body = request.get_json(silent=True) or {}
        state, key, error = current(body.get("source"))
        if state is None:
            return jsonify({"error": error}), 400
        if state["source"] not in state["sources"]:
            return jsonify({"error": f"This experiment has no {state['source']} tables"}), 400
        if state["n_animals"] < 2:
            return jsonify({"error": "One fly: nothing to swap with"}), 400
        job = running_job(key)
        if job is not None:
            return jsonify(job.to_dict())
        try:
            params = params_with(body.get("params"))
        except (TypeError, ValueError):
            return jsonify({"error": "Bad parameters"}), 400
        _last_params[key] = params

        def work(job):
            job.update(0, status="running")
            result = compute_swaps(state["dbfile"], state["source"], state["n_animals"], state["chunksize"],
                                   state["framerate"], params, lambda p: job.update(0, progress=p))
            try:
                from idtrackerai_validator_server.box_annotations import _clocks
                clocks = _clocks(state["dbfile"], [e["frame"] for e in result["events"]])
            except Exception:
                logger.exception("No clock for the swap events")
                clocks = {}
            for e in result["events"]:
                e["clock"] = clocks.get(e["frame"])
            result.update(experiment=state["experiment"], db_mtime=state["mtime"], computed_at=time.time())
            with _lock:
                _cache[(state["dbfile"], state["source"], _params_key(params))] = result
            _write_cache(_cache_path(state["experiment"], state["source"], params), result)
            job.update(0, status="done", progress=1.0)
            job.result = {"events": len(result["events"]), "counts": result["counts"]}

        job = start_job(f"Swap check {state['experiment']} ({state['source']})", [state["experiment"]], work)
        with _lock:
            _running[key] = job.id
        return jsonify(job.to_dict())
