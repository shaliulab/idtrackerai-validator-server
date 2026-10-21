"""Pose of the loaded experiment: quality control (QC tab) and one fly at a time (Pose tab).

The pose of a fly is the SLEAP analysis file flyhostel compiles from its chunks
(<basedir>/motionmapper/NN/pose_raw/<experiment>__NN/<experiment>__NN.h5, made by
`recreate_pose_file` when missing): `tracks` (1, 2, nodes, frames), relative to the
top-left corner of the fly's square; `point_scores` (1, nodes, frames), the confidence
of each body part; `instance_scores` (1, frames); `files`, its chunks (consecutive),
so frame index i is frame first_chunk * chunksize + i. A body part is detected when
SLEAP kept its peak (confidence >= 0.2): undetected parts have NaN coordinates and
confidence 0; frames of chunks without pose are NaN everywhere (`missing`).

Pose score of a frame: SLEAP's instance score of centered-instance models, the sum of
the confidences of the detected body parts, divided by the number of body parts
scored (SCORED_NODES, the parts the viewer draws: leg joints are left out), so 1 =
every part found with full confidence (SLEAP's peaks can exceed 1 a little).

    GET  /api/pose/qc                       last pose QC of every fly (null per fly if
                                            none or its pose file changed) and its job
    POST /api/pose/qc                       start that job (one item per fly)
    GET  /api/pose/flies                    flies of the loaded experiment, `available`
                                            = pose file exists
    GET  /api/pose/<fly>/frame/<n>          body parts (absolute x, y, confidence) and
                                            scores of the fly at frame n
    GET  /api/pose/<fly>/trace?start=&end=  per-frame parts detected, mean confidence
                                            and pose score of frames [start, end)

The QC reads only `point_scores` (~12 s for 17.5 M frames) and is kept in memory and
in $QC_CACHE_DIR/pose (one JSON per fly, valid while the pose file's mtime and
CACHE_VERSION are unchanged).
"""
import json
import logging
import os
import time
from pathlib import Path
from threading import Lock

import h5py
import numpy as np
from flask import jsonify, request

from flyhostel.utils import get_identities, get_square_width, get_square_height
from flyhostel.utils.pose_export import recreate_pose_file, get_pose_file_path

from idtrackerai_validator_server.experiments import to_flat
from idtrackerai_validator_server.jobs import start_job, _jobs

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.environ.get("QC_CACHE_DIR", os.path.expanduser("~/.cache/idtrackerai-validator/qc")), "pose")
CACHE_VERSION = 1
SCORED_NODES = ["proboscis", "thorax", "abdomen", "fLL", "mLL", "rLL", "fRL", "mRL", "rRL", "head", "lW", "rW"]
BINS_PER_CHUNK = 30         # as the QC of the tracking: 10 s bins for 45000-frame chunks at 150 fps
LOW_SCORE = 0.5             # frames below this pose score count as poor
SCORE_BINS = 20             # histogram of the pose score over [0, 1] (above 1 in the last bin)
WORST_BINS = 50             # bins with the lowest mean score listed
MAX_TRACE = 60000           # frames per /trace request
# Each chunk of `tracks` is 45000 frames (6.5 MB): without a bigger cache h5py
# decompresses it again for every frame read
H5_CACHE_BYTES = 64 * 1024 ** 2

_cache = {}                 # pose file -> result
_running = {}               # experiment -> job id
_lock = Lock()

# Open pose files of the loaded experiment: <experiment>__NN -> h5py.File (None: no pose)
_h5_file_cache = {}
_h5_cache_lock = Lock()


def pose_path(experiment, fly_id_str):
    return get_pose_file_path(to_flat(experiment), int(fly_id_str), "raw")


def get_h5_file(fly_id_str, experiment):
    """Get or open H5 file with caching. The file is in the experiment's basedir
    (<basedir>/motionmapper/NN/pose_raw/<experiment>__NN/), recreated when missing
    or unreadable; None (cached, not retried) if that fails too."""
    experiment = to_flat(experiment)
    cache_key = f"{experiment}__{fly_id_str}"

    with _h5_cache_lock:
        if cache_key in _h5_file_cache:
            return _h5_file_cache[cache_key]

        pose_file = pose_path(experiment, fly_id_str)
        # recreate_pose_file writes <output>/<experiment>__NN/<experiment>__NN.h5
        folder = os.path.dirname(os.path.dirname(pose_file))
        f = None
        try:
            if not Path(pose_file).exists():
                logger.info("%s not found, recreating it", pose_file)
                recreate_pose_file(experiment, int(fly_id_str), output=folder)
            try:
                f = h5py.File(pose_file, 'r', rdcc_nbytes=H5_CACHE_BYTES)
            except OSError as e:
                logger.warning("Failed to open pose file %s: %s. Recreating it", pose_file, e)
                os.remove(pose_file)
                recreate_pose_file(experiment, int(fly_id_str), output=folder)
                f = h5py.File(pose_file, 'r', rdcc_nbytes=H5_CACHE_BYTES)
            logger.debug("Opened pose file: %s", pose_file)
        except Exception as e:
            logger.error("No pose for %s: %s", cache_key, e)
        _h5_file_cache[cache_key] = f
        return f


def close_h5_files():
    """Close all cached H5 file handles"""
    with _h5_cache_lock:
        for f in _h5_file_cache.values():
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        _h5_file_cache.clear()


def first_frame(h5_file, chunksize):
    """Frame number of index 0 of the pose file (its first chunk)."""
    return int(os.path.basename(h5_file["files"][0].decode()).split(".")[0]) * chunksize


def node_names(h5_file):
    return [name.decode() for name in h5_file["node_names"][:]]


def scored_indices(names):
    scored = [i for i, name in enumerate(names) if name in SCORED_NODES]
    return scored or list(range(len(names)))


def frame_metrics(scores, scored):
    """scores: (nodes, frames) confidences. Per frame: body parts detected, mean
    confidence of those (NaN if none), pose score, and whether the frame has no pose."""
    missing = np.isnan(scores).all(axis=0)
    sub = np.nan_to_num(scores[scored], nan=0.0)
    detected = sub > 0
    n_detected = detected.sum(axis=0)
    total = sub.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        confidence = np.where(n_detected > 0, total / n_detected, np.nan)
    score = total / len(scored)
    confidence[missing] = np.nan
    score[missing] = np.nan
    return n_detected, confidence, score, missing


def _round(value, digits=3):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def _bin_means(values, valid, bins, n_bins):
    counts = np.bincount(bins[valid], minlength=n_bins)
    sums = np.bincount(bins[valid], weights=values[valid], minlength=n_bins)
    return [None if c == 0 else _round(s / c) for s, c in zip(sums, counts)]


def compute_pose_qc(path, chunksize, framerate, progress=None):
    with h5py.File(path, "r") as h5_file:
        names = node_names(h5_file)
        scored = scored_indices(names)
        n_scored = len(scored)
        start = first_frame(h5_file, chunksize)
        dataset = h5_file["point_scores"]
        n_frames = dataset.shape[2]
        n_chunks = int(np.ceil(n_frames / chunksize))
        bin_size = max(1, chunksize // BINS_PER_CHUNK)

        hist = np.zeros(n_scored + 1, dtype=np.int64)
        score_hist = np.zeros(SCORE_BINS, dtype=np.int64)
        node_detected = np.zeros(len(names), dtype=np.int64)
        node_conf = np.zeros(len(names))
        totals = {"frames": 0, "missing": 0, "empty": 0, "complete": 0, "low": 0}
        sums = {"n_detected": 0.0, "confidence": 0.0, "confidence_frames": 0, "score": 0.0}
        per_chunk = []
        worst = []          # (mean score, first frame of the bin)

        for k in range(n_chunks):
            a, b = k * chunksize, min(n_frames, (k + 1) * chunksize)
            scores = dataset[0, :, a:b]
            n_detected, confidence, score, missing = frame_metrics(scores, scored)
            valid = ~missing
            has_conf = valid & (n_detected > 0)
            n_valid = int(valid.sum())

            detected_nodes = (np.nan_to_num(scores, nan=0.0) > 0)
            chunk_node_detected = detected_nodes.sum(axis=1)
            chunk_node_conf = np.where(detected_nodes, scores, 0).sum(axis=1)
            node_detected += chunk_node_detected
            node_conf += chunk_node_conf

            hist += np.bincount(n_detected[valid], minlength=n_scored + 1)
            score_hist += np.bincount(np.clip((score[valid] * SCORE_BINS).astype(np.int64), 0, SCORE_BINS - 1),
                                      minlength=SCORE_BINS)
            counts = {
                "frames": b - a, "missing": int(missing.sum()),
                "empty": int((valid & (n_detected == 0)).sum()),
                "complete": int((valid & (n_detected == n_scored)).sum()),
                "low": int((valid & (score < LOW_SCORE)).sum()),
            }
            for key, value in counts.items():
                totals[key] += value
            sums["n_detected"] += float(n_detected[valid].sum())
            sums["confidence"] += float(confidence[has_conf].sum())
            sums["confidence_frames"] += int(has_conf.sum())
            sums["score"] += float(score[valid].sum())

            bins = np.minimum(np.arange(b - a) // bin_size, BINS_PER_CHUNK - 1)
            score_bins = _bin_means(score, valid, bins, BINS_PER_CHUNK)
            chunk_first = start + a
            for j, value in enumerate(score_bins):
                if value is not None:
                    worst.append((value, chunk_first + j * bin_size))

            low_frames = np.flatnonzero(valid & (score < LOW_SCORE))
            per_chunk.append({
                "chunk": chunk_first // chunksize, **counts,
                "n_detected": _round(n_detected[valid].mean()) if n_valid else None,
                "confidence": _round(confidence[has_conf].mean()) if has_conf.any() else None,
                "score": _round(score[valid].mean()) if n_valid else None,
                "score_p10": _round(np.percentile(score[valid], 10)) if n_valid else None,
                "first_low": int(chunk_first + low_frames[0]) if len(low_frames) else None,
                "bins": {
                    "n_detected": _bin_means(n_detected.astype(float), valid, bins, BINS_PER_CHUNK),
                    "confidence": _bin_means(confidence, has_conf, bins, BINS_PER_CHUNK),
                    "score": score_bins,
                },
                # % of valid frames where each body part is detected, and its mean confidence
                "nodes": {
                    "detected": [_round(100 * d / n_valid, 1) if n_valid else None for d in chunk_node_detected],
                    "confidence": [_round(c / d) if d else None for c, d in zip(chunk_node_conf, chunk_node_detected)],
                },
            })
            if progress:
                progress((k + 1) / n_chunks)

    n_valid = totals["frames"] - totals["missing"]
    worst.sort()
    return {
        "version": CACHE_VERSION,
        "nodes": names,
        "scored": [names[i] for i in scored],
        "chunksize": chunksize,
        "framerate": framerate,
        "bin_size": bin_size,
        "bins_per_chunk": BINS_PER_CHUNK,
        "low_score": LOW_SCORE,
        "first_frame": start,
        "first_chunk": per_chunk[0]["chunk"] if per_chunk else None,
        "last_chunk": per_chunk[-1]["chunk"] if per_chunk else None,
        "totals": totals,
        "mean": {
            "n_detected": _round(sums["n_detected"] / n_valid) if n_valid else None,
            "confidence": _round(sums["confidence"] / sums["confidence_frames"]) if sums["confidence_frames"] else None,
            "score": _round(sums["score"] / n_valid) if n_valid else None,
        },
        "hist": hist.tolist(),
        "score_hist": score_hist.tolist(),
        "node_totals": {
            "detected": [_round(100 * d / n_valid, 1) if n_valid else None for d in node_detected],
            "confidence": [_round(c / d) if d else None for c, d in zip(node_conf, node_detected)],
        },
        "chunks": per_chunk,
        "worst_bins": [[frame, value] for value, frame in worst[:WORST_BINS]],
    }


def _cache_path(path):
    return os.path.join(CACHE_DIR, os.path.basename(path).replace(".h5", ".json"))


def _read_cache(path, mtime):
    try:
        with open(_cache_path(path)) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if data.get("pose_mtime") != mtime or data.get("version") != CACHE_VERSION:
        return None
    return data


def _write_cache(path, data):
    target = _cache_path(path)
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(data, handle)
        os.replace(tmp, target)
    except OSError as error:
        logger.warning("Could not keep the pose QC in %s: %s", target, error)


def _flies(experiment):
    """[(fly, NN)] of the experiment, e.g. (FlyHostel1_2X_<date>__01, "01")."""
    flat = to_flat(experiment)
    return [(f"{flat}__{str(identity).zfill(2)}", str(identity).zfill(2)) for identity in get_identities(flat)]


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def register_pose(app, get_state):
    """get_state() -> None (no database loaded) or a dict with experiment, chunksize,
    framerate and centroid(frame_number, identity) -> (x, y) or None."""

    def cached(path):
        mtime = _mtime(path)
        if mtime is None:
            return None
        with _lock:
            result = _cache.get(path)
        if result is not None and result["pose_mtime"] == mtime:
            return result
        result = _read_cache(path, mtime)
        if result is not None:
            with _lock:
                _cache[path] = result
        return result

    def running_job(experiment):
        with _lock:
            job_id = _running.get(experiment)
        job = _jobs.get(job_id) if job_id else None
        return job if job is not None and job.status == "running" else None

    def fly_of(fly):
        state = get_state()
        if state is None:
            return None, None
        flies = dict(_flies(state["experiment"]))
        return state, flies.get(fly)

    @app.route("/api/pose/flies", methods=["GET"])
    def pose_flies():
        state = get_state()
        if state is None:
            return jsonify([])
        return jsonify([{"fly": fly, "identity": int(nn), "available": os.path.exists(pose_path(state["experiment"], nn))}
                        for fly, nn in _flies(state["experiment"])])

    @app.route("/api/pose/qc", methods=["GET"])
    def get_pose_qc():
        state = get_state()
        if state is None:
            return jsonify({"error": "Load an experiment with a database first"}), 400
        flies = []
        for fly, nn in _flies(state["experiment"]):
            path = pose_path(state["experiment"], nn)
            flies.append({"fly": fly, "identity": int(nn), "path": path,
                          "available": os.path.exists(path), "result": cached(path)})
        job = running_job(state["experiment"])
        return jsonify({"experiment": state["experiment"], "flies": flies, "job": job.to_dict() if job else None})

    @app.route("/api/pose/qc", methods=["POST"])
    def start_pose_qc():
        state = get_state()
        if state is None:
            return jsonify({"error": "Load an experiment with a database first"}), 400
        job = running_job(state["experiment"])
        if job is not None:
            return jsonify(job.to_dict())
        flies = _flies(state["experiment"])

        def work(job):
            failed = []
            for i, (fly, nn) in enumerate(flies):
                job.update(i, status="running")
                try:
                    path = pose_path(state["experiment"], nn)
                    if not os.path.exists(path):
                        # compiles it from the chunks' .predictions.h5 (minutes)
                        recreate_pose_file(to_flat(state["experiment"]), int(nn), output=os.path.dirname(os.path.dirname(path)))
                    mtime = _mtime(path)
                    result = compute_pose_qc(path, state["chunksize"], state["framerate"],
                                             lambda p, i=i: job.update(i, progress=p))
                    result.update(fly=fly, pose_mtime=mtime, computed_at=time.time())
                    with _lock:
                        _cache[path] = result
                    _write_cache(path, result)
                    job.update(i, status="done", progress=1.0)
                except Exception as error:
                    logger.exception("Pose QC of %s failed", fly)
                    job.update(i, status="failed", error=str(error))
                    failed.append(fly)
            if failed and len(failed) == len(flies):
                raise RuntimeError(f"No pose for {', '.join(failed)}")

        job = start_job(f"Pose QC {state['experiment']}", [fly for fly, _ in flies], work)
        with _lock:
            _running[state["experiment"]] = job.id
        return jsonify(job.to_dict())

    @app.route("/api/pose/<fly>/frame/<int:frame_number>", methods=["GET"])
    def pose_frame(fly, frame_number):
        state, nn = fly_of(fly)
        if nn is None:
            return jsonify({"error": f"Not a fly of the loaded experiment: {fly}"}), 404
        h5_file = get_h5_file(nn, state["experiment"])
        if h5_file is None:
            return jsonify({"error": f"No pose for {fly}"}), 404
        experiment = to_flat(state["experiment"])
        start = first_frame(h5_file, state["chunksize"])
        n_frames = h5_file["point_scores"].shape[2]
        out = {"fly": fly, "frame_number": frame_number, "chunksize": state["chunksize"], "first_frame": start, "last_frame": start + n_frames - 1,
               "square": [get_square_width(experiment), get_square_height(experiment)]}
        i = frame_number - start
        if not 0 <= i < n_frames:
            return jsonify({**out, "error": "Frame outside the pose file", "nodes": []})

        names = node_names(h5_file)
        scored = scored_indices(names)
        scores = h5_file["point_scores"][0, :, i]
        xy = h5_file["tracks"][0, :, :, i]
        n_detected, confidence, score, missing = frame_metrics(scores[:, None], scored)
        centroid = state["centroid"](frame_number, int(nn))
        # coordinates are relative to the top-left corner of the square centred on the fly
        top_left = None if centroid is None else (centroid[0] - out["square"][0] // 2, centroid[1] - out["square"][1] // 2)
        nodes = []
        for k, name in enumerate(names):
            rx, ry = float(xy[0, k]), float(xy[1, k])
            detected = bool(np.isfinite(rx) and np.isfinite(ry))
            nodes.append({
                "name": name, "scored": k in scored, "detected": detected,
                "confidence": _round(scores[k]),
                "rel": [round(rx, 2), round(ry, 2)] if detected else None,
                "xy": [round(top_left[0] + rx, 2), round(top_left[1] + ry, 2)] if detected and top_left else None,
            })
        return jsonify({
            **out, "nodes": nodes, "centroid": centroid, "missing": bool(missing[0]),
            "n_detected": int(n_detected[0]), "n_scored": len(scored),
            "confidence": _round(confidence[0]), "score": _round(score[0]),
            "instance_score": _round(h5_file["instance_scores"][0, i]) if "instance_scores" in h5_file else None,
        })

    @app.route("/api/pose/<fly>/trace", methods=["GET"])
    def pose_trace(fly):
        state, nn = fly_of(fly)
        if nn is None:
            return jsonify({"error": f"Not a fly of the loaded experiment: {fly}"}), 404
        h5_file = get_h5_file(nn, state["experiment"])
        if h5_file is None:
            return jsonify({"error": f"No pose for {fly}"}), 404
        try:
            start_frame, end_frame = int(request.args["start"]), int(request.args["end"])
        except (KeyError, ValueError):
            return jsonify({"error": "start and end (frame numbers) required"}), 400
        end_frame = min(end_frame, start_frame + MAX_TRACE)
        first = first_frame(h5_file, state["chunksize"])
        n_frames = h5_file["point_scores"].shape[2]
        a, b = max(0, start_frame - first), min(n_frames, end_frame - first)
        if b <= a:
            return jsonify({"start": start_frame, "n_detected": [], "confidence": [], "score": []})
        scores = h5_file["point_scores"][0, :, a:b]
        n_detected, confidence, score, missing = frame_metrics(scores, scored_indices(node_names(h5_file)))
        clean = lambda values: [None if not np.isfinite(v) else round(float(v), 3) for v in values]
        return jsonify({
            "start": first + a, "n_scored": len(scored_indices(node_names(h5_file))),
            "n_detected": [None if m else int(n) for n, m in zip(n_detected, missing)],
            "confidence": clean(confidence), "score": clean(score),
        })
