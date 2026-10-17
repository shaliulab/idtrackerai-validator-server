"""Tracking rows read from idtrackerai's blobs_collection*.npy instead of the
database, to check an ongoing run.

idtrackerai writes one session per chunk, <basedir>/idtrackerai/session_<chunk:06d>/,
and saves its ListOfBlobs after each step; the most advanced file is used.
A ListOfBlobs takes seconds to unpickle, so the last few are kept in memory and
reloaded when idtrackerai rewrites them (mtime changes).
"""
import os
import glob
import logging
from collections import OrderedDict
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)

# Most advanced step first
CANDIDATES = [
    ("tracking", "blobs_collection_no_gaps.npy"),
    ("tracking", "blobs_collection.npy"),
    ("preprocessing", "blobs_collection_interpolated.npy"),
    ("preprocessing", "blobs_collection.npy"),
    ("preprocessing", "blobs_collection_segmented.npy"),
]
CACHE_SIZE = 3

_cache = OrderedDict()  # path -> (mtime, ListOfBlobs)
_cache_lock = Lock()
_path_locks = {}


def find_blobs_collection(basedir, chunk):
    session = os.path.join(basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}")
    for step, filename in CANDIDATES:
        path = os.path.join(session, step, filename)
        if os.path.exists(path):
            return path
    return None


def has_blobs_collections(basedir):
    pattern = os.path.join(basedir, "idtrackerai", "session_*", "*", "blobs_collection*.npy")
    return next(glob.iglob(pattern), None) is not None


def session_chunks(basedir):
    """Chunks with at least one blobs_collection file, sorted."""
    sessions = glob.glob(os.path.join(basedir, "idtrackerai", "session_*"))
    chunks = []
    for session in sessions:
        try:
            chunk = int(os.path.basename(session).split("_")[1])
        except (IndexError, ValueError):
            continue
        if find_blobs_collection(basedir, chunk) is not None:
            chunks.append(chunk)
    return sorted(chunks)


def _load(path):
    mtime = os.path.getmtime(path)
    with _cache_lock:
        hit = _cache.get(path)
        if hit is not None and hit[0] == mtime:
            _cache.move_to_end(path)
            return hit[1], mtime
        path_lock = _path_locks.setdefault(path, Lock())

    # One load per file: the client prefetches frames of the same chunk in parallel
    with path_lock:
        with _cache_lock:
            hit = _cache.get(path)
            if hit is not None and hit[0] == mtime:
                return hit[1], mtime
        logger.info("Loading %s", path)
        list_of_blobs = np.load(path, allow_pickle=True).item()
        with _cache_lock:
            _cache[path] = (mtime, list_of_blobs)
            _cache.move_to_end(path)
            while len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
    return list_of_blobs, mtime


def clear_cache():
    with _cache_lock:
        _cache.clear()


def _blobs_in_frame(list_of_blobs, frame_number, chunk, chunksize):
    """blobs_in_video is usually extended to the whole video (index = frame number),
    else it starts at the chunk's first frame."""
    blobs_in_video = list_of_blobs.blobs_in_video
    for index in (frame_number, frame_number - chunk * chunksize):
        if 0 <= index < len(blobs_in_video):
            blobs = blobs_in_video[index]
            if not blobs or blobs[0].frame_number == frame_number:
                return blobs
    return []


def _centroids_and_identities(blob):
    try:
        centroids = blob.final_centroids_full_resolution
    except Exception:
        centroids = []
    if not centroids:
        centroids = [blob.centroid]
    try:
        identities = list(blob.final_identities or [])
    except Exception:
        identities = []
    if len(identities) != len(centroids):
        # e.g. an unsolved crossing: one centroid, no single identity
        identities = [identities[0] if len(identities) == 1 else None]
        centroids = [blob.centroid]
    return centroids, identities


def read_blobs(basedir, frame_number, chunksize):
    """(rows, info): rows like the ROI_0/IDENTITY rows of /api/tracking, one per
    centroid of each blob; identity 0 = no identity (yet)."""
    chunk = frame_number // chunksize
    path = find_blobs_collection(basedir, chunk)
    info = {"chunk": chunk, "path": None}
    if path is None:
        info["error"] = f"No blobs_collection for chunk {chunk}"
        return [], info

    info["path"] = os.path.relpath(path, basedir)
    try:
        list_of_blobs, mtime = _load(path)
    except Exception as error:
        # e.g. idtrackerai is writing it right now
        logger.error("Cannot load %s: %s", path, error)
        info["error"] = f"Cannot load {info['path']}: {error}"
        return [], info
    info["mtime"] = mtime

    rows = []
    for blob in _blobs_in_frame(list_of_blobs, frame_number, chunk, chunksize):
        centroids, identities = _centroids_and_identities(blob)
        for (x, y), identity in zip(centroids, identities):
            rows.append({
                "x": float(x),
                "y": float(y),
                "in_frame_index": int(blob.in_frame_index),
                "fragment": -1 if blob.fragment_identifier is None else int(blob.fragment_identifier),
                "area": float(blob.area),
                "identity": int(identity) if identity is not None else 0,
                "local_identity": None,
                "modified": int(bool(getattr(blob, "_modified", False))),
                "crossing": bool(blob.is_a_crossing),
            })
    return rows, info
