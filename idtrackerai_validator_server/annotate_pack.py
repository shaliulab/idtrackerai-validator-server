"""
annotate_pack.py  —  the Annotate tab's pre-download: every frame of the scenes to check, made once on the
server and fetched by the browser a chunk at a time, so stepping through the scenes reads nothing from the server.

Per experiment, in $ANNOTATE_PACK_CACHE/<flat>/ (lab_paths, default ~/.cache/idtrackerai-validator/annotate_pack):

    <chunk:06d>.h264         the chunk's scene frames in frame order as one H.264 Annex B stream (x264 PRESET,
                             crf CRF, yuv420p, no B-frames: decode order = frame order; an IDR frame at the start
                             of every run of consecutive frames and every GOP frames, SPS / PPS repeated before each,
                             an access unit delimiter before every frame). ~19 kB per 2100 px frame and ~52 dB PSNR
                             against the recording (the tab's JPEGs at quality 50: ~113 kB, ~47 dB). Decoded in the
                             browser with WebCodecs.
    <chunk:06d>.contours.npz idtrackerai's blobs of each frame (backend.process_frame with the database's config:
                             exactly app._default_contours'), which the proposals are made from: /api/annotate/frame
                             reads them here instead of reading and segmenting the frame (app._annotate_state).
    <chunk:06d>.json         written last (the chunk is built): stamp, frames, byte offset / size / key of each frame
                             in the stream, codec string, width, height.
    <chunk:06d>.data.gz      gzip of {sig, frames: {frame: /api/annotate/frame's answer}}: the saved boxes and the
                             proposals of every frame of the chunk, made again when `sig` (chunk_sigs: database,
                             scene_qc.csv, the last solver run, the frames of the chunk saved and when) changes.
    <chunk:06d>.data.c2.gz   the same with continuity proposals of tracker v2 (continuity.RunV2): the data is made
                             for the continuity version the browser asks for (?continuity= / "continuity", the
                             Annotate tab's proposal choice; default the server's), each with its own sig, so a
                             person on v1 and one on v2 share the streams and contours but not the frame data.

Built by a job (POST /api/annotate/pack): chunks in the order asked, WORKERS spawned processes each reading its
chunk's frames from the recording (forward, exchange_bundle._SourceReader), encoding them with ffmpeg, segmenting
them, then making the chunk's frame data with a state of their own (pack_chunk); built chunks whose data sig
changed get data-only tasks. Server memory stays bounded: the server holds one chunk's data (a request making it
again) and CONTOURS_CHUNKS chunks of contours; a worker one frame and its chunk's contours.

The browser (AnnotateTab.js, "Pre-download") keeps what it fetched in IndexedDB; GET /api/annotate/pack tells it
which chunks are built and each chunk's data sig, so it fetches again only the data of chunks saved elsewhere.
"""
import gzip
import hashlib
import io
import json
import logging
import multiprocessing
import os
import shutil
import sqlite3
import subprocess
import threading
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import yaml
from flask import Response, jsonify, request, send_file

from idtrackerai_validator_server import lab_paths
from idtrackerai_validator_server.experiments import to_flat
from idtrackerai_validator_server.jobs import _jobs, start_job

logger = logging.getLogger(__name__)

PACK_VERSION = 1
CRF = int(os.environ.get("ANNOTATE_PACK_CRF", "30"))
GOP = 12
PRESET = "veryfast"
FFMPEG_THREADS = 2
WORKERS = int(os.environ.get("ANNOTATE_PACK_WORKERS", str(max(1, min(4, (os.cpu_count() or 2) // 16)))))
CONTOURS_CHUNKS = 4        # chunks of contours kept in memory
KEEP_DAYS = 30             # packs of other experiments untouched for longer are deleted when a job starts

_lock = threading.Lock()
_running = {}              # flat -> job id
_stop = {}                 # flat -> threading.Event
_chunk_locks = {}          # (flat, chunk) -> Lock (data made by one request at a time)
_contours = OrderedDict()  # (flat, chunk, mtime) -> {frame: [array (N, 2)]}
_frames_cache = {}         # dbfile -> (stamp, {chunk: [frames]})


def pack_dir(experiment):
    return os.path.join(lab_paths.ANNOTATE_PACK_CACHE, to_flat(experiment))


def _mtime_size(path):
    try:
        st = os.stat(path)
        return [st.st_mtime, st.st_size]
    except OSError:
        return [None, None]


def _store(basedir):
    with open(os.path.join(basedir, "metadata.yaml")) as handle:
        store = yaml.safe_load(handle)["__store"]
    return {"width": int(store["imgshape"][1]), "height": int(store["imgshape"][0]),
            "framerate": float(store["framerate"]), "extension": store.get("extension", ".mp4")}


def _config_hash(config):
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


def video_stamp(state):
    """What a chunk's stream and contours depend on, besides its frames"""
    return [PACK_VERSION, CRF, GOP, PRESET, *_mtime_size(os.path.join(state["basedir"], "index.db")),
            _config_hash(state.get("config"))]


def pack_frames(state):
    """{chunk: [frames]} of the scenes the Annotate tab lists (scene_qc.csv + the unsure chunk boundaries), inside
    the recording. Cached per database / scene_qc.csv."""
    from idtrackerai_validator_server import box_annotations as ba
    basedir, dbfile = state["basedir"], state["dbfile"]
    stamp = [*_mtime_size(dbfile), *_mtime_size(os.path.join(ba._validation_folder(basedir), "scene_qc.csv")),
             *_mtime_size(os.path.join(basedir, "index.db"))]
    hit = _frames_cache.get(dbfile)
    if hit and hit[0] == stamp:
        return hit[1]
    scenes, source = ba.read_scenes(basedir, state["experiment"], state["n_animals"])
    if source["kind"] != "bundle":
        scenes = ba.add_boundary_scenes(scenes, ba.unsure_boundaries(dbfile, state["chunksize"], state["n_animals"]),
                                        state["chunksize"])
    with sqlite3.connect(f"file:{os.path.join(basedir, 'index.db')}?mode=ro", uri=True) as con:
        lo, hi = con.execute("SELECT MIN(frame_number), MAX(frame_number) FROM frames").fetchone()
    frames = set()
    for sc in scenes:
        frames.update(range(max(sc["start"], lo), min(sc["start"] + sc["length"], hi + 1)))
    out = {}
    for fn in sorted(frames):
        out.setdefault(fn // state["chunksize"], []).append(fn)
    _frames_cache[dbfile] = (stamp, out)
    return out


def _read_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def vkey(stamp, frames):
    """Key of a chunk's stream: its frames and how they were encoded (the browser keeps it with the stream)"""
    return hashlib.sha1(json.dumps([stamp, frames]).encode()).hexdigest()[:16]


def chunk_index(state, chunk, frames=None, stamp=None):
    """The chunk's index (<chunk>.json) when it is built for the frames and stamp of now, else None"""
    index = _read_json(os.path.join(pack_dir(state["experiment"]), f"{chunk:06d}.json"))
    if index is None:
        return None
    frames = frames if frames is not None else pack_frames(state).get(chunk)
    if frames is None or index.get("vkey") != vkey(stamp or video_stamp(state), frames):
        return None
    return index


def chunk_sigs(state, chunks, version=1):
    """{chunk: sig} of the frame data (continuity proposals of tracker `version`): changes when the database,
    scene_qc.csv, the solver's proposals or a frame of the chunk saved (any person, any computer: frames.updated)
    changes."""
    from idtrackerai_validator_server import box_annotations as ba
    base = [PACK_VERSION, *_mtime_size(state["dbfile"]),
            *_mtime_size(os.path.join(ba._validation_folder(state["basedir"]), "scene_qc.csv"))]
    if version == 2:
        base.append("continuity2")
    saved, run = {}, None
    con = ba._connect(state["basedir"])
    if con is not None:
        try:
            with con:
                last = con.execute("SELECT MAX(t) FROM proposal_runs").fetchone()
                run = last[0] if last else None
                for fn, updated in con.execute("SELECT frame_number, updated FROM frames ORDER BY frame_number"):
                    saved.setdefault(fn // state["chunksize"], []).append((fn, updated))
        finally:
            con.close()
    return {c: hashlib.sha1(json.dumps([*base, run, saved.get(c, [])]).encode()).hexdigest()[:16] for c in chunks}


# --- building a chunk (worker process) ----------------------------------------------------------------------

def split_access_units(data):
    """[(offset, size, key)] of the access units of an Annex B stream written with an AUD before every frame"""
    marker = b"\x00\x00\x00\x01\x09"
    starts = []
    at = data.find(marker)
    while at >= 0:
        starts.append(at)
        at = data.find(marker, at + 1)
    units = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(data)
        unit = data[start:end]
        key, pos = False, unit.find(b"\x00\x00\x01", 4)
        while pos >= 0:
            if unit[pos + 3] & 0x1F == 5:
                key = True
                break
            pos = unit.find(b"\x00\x00\x01", pos + 3)
        units.append((start, end - start, key))
    return units


def codec_string(data):
    """avc1.PPCCLL of the stream's first SPS"""
    pos = data.find(b"\x00\x00\x01\x67")
    if pos < 0:
        raise ValueError("no SPS in the stream")
    profile, constraints, level = data[pos + 4], data[pos + 5], data[pos + 6]
    return f"avc1.{profile:02x}{constraints:02x}{level:02x}"


def encode_contours(by_frame):
    """{frame: [contour]} -> arrays for np.savez_compressed (points delta coded: idtrackerai's contours step 1 px)"""
    frames, counts, lens, parts = [], [], [], []
    for fn in sorted(by_frame):
        cs = [np.asarray(c, dtype=np.int32).reshape(-1, 2) for c in by_frame[fn]]
        frames.append(fn)
        counts.append(len(cs))
        for c in cs:
            lens.append(len(c))
            if len(c):
                parts.append(np.diff(c, axis=0, prepend=np.zeros((1, 2), np.int32)))
    points = np.concatenate(parts) if parts else np.zeros((0, 2), np.int32)
    return {"frames": np.asarray(frames, np.int64), "counts": np.asarray(counts, np.int32),
            "lens": np.asarray(lens, np.int32), "deltas": points.astype(np.int32)}


def decode_contours(arrays):
    """inverse of encode_contours: {frame: [array (N, 2) int32]}"""
    lens = arrays["lens"]
    deltas = arrays["deltas"]
    out, k, at = {}, 0, 0
    for fn, n in zip(arrays["frames"].tolist(), arrays["counts"].tolist()):
        cs = []
        for _ in range(n):
            m = int(lens[k])
            cs.append(np.cumsum(deltas[at:at + m], axis=0, dtype=np.int64).astype(np.int32))
            at += m
            k += 1
        out[fn] = cs
    return out


def _init_worker():
    """A worker uses ~1 core besides its ffmpeg (OpenCV's decoding and segmentation would take every core)"""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;2"
    import cv2
    cv2.setNumThreads(1)


def build_chunk(task):
    """Worker: the stream, contours and index of one chunk (task: see _task). Returns (summary, contours)."""
    from idtrackerai_validator_server.exchange_bundle import _SourceReader
    t0 = time.time()
    out, chunk, frames = task["out"], task["chunk"], task["frames"]
    width, height = task["width"], task["height"]
    stream = os.path.join(out, f"{chunk:06d}.h264")
    part = stream + ".part"
    # an IDR frame at the start of each run of consecutive frames (a jump to a scene decodes from there)
    starts = [i for i, fn in enumerate(frames) if i == 0 or frames[i - 1] != fn - 1]
    cmd = [task["ffmpeg"], "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
           "-s", f"{width}x{height}", "-r", "1", "-i", "-",
           "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF), "-g", str(GOP), "-bf", "0",
           "-forced-idr", "1", "-force_key_frames", ",".join(str(i) for i in starts),
           "-x264-params", "aud=1:repeat-headers=1", "-threads", str(FFMPEG_THREADS),
           "-pix_fmt", "yuv420p", "-f", "h264", part]
    config = task["config"]
    segment = None
    if config is not None:
        try:
            from idtrackerai_validator_server.backend import process_frame
            segment = process_frame
        except Exception as error:     # no idtrackerai: no contours (the frames are read and segmented on demand)
            logger.warning("No contours in the pre-download: %s", error)
    contours = {}
    reader = _SourceReader(task["basedir"], task["extension"])
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for k, (fn, idx) in enumerate(zip(frames, task["frame_idx"])):
            if k % 100 == 0 and task.get("parent") and os.getppid() != task["parent"]:
                raise RuntimeError("the server stopped")       # orphaned by a restart: do not finish the chunk
            image = reader.read(chunk, idx)
            if image.shape != (height, width):
                raise ValueError(f"frame {fn} is {image.shape}, metadata.yaml says {(height, width)}")
            encoder.stdin.write(np.ascontiguousarray(image).tobytes())
            if segment is not None:
                try:
                    contours[fn] = segment(image, json.loads(json.dumps(config)))
                except Exception as error:
                    logger.warning("No contours for frame %s: %s", fn, error)
                    contours[fn] = []
    finally:
        reader.close()
        encoder.stdin.close()
        error = encoder.stderr.read().decode(errors="replace")
        code = encoder.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed on chunk {chunk}: {error.strip()[-300:]}")
    with open(part, "rb") as handle:
        data = handle.read()
    units = split_access_units(data)
    if len(units) != len(frames):
        raise RuntimeError(f"chunk {chunk}: {len(units)} frames encoded, {len(frames)} given")
    if segment is not None:
        np.savez_compressed(os.path.join(out, f"{chunk:06d}.contours.part.npz"), **encode_contours(contours))
        os.replace(os.path.join(out, f"{chunk:06d}.contours.part.npz"), os.path.join(out, f"{chunk:06d}.contours.npz"))
    os.replace(part, stream)
    index = {"vkey": vkey(task["stamp"], frames), "stamp": task["stamp"], "chunk": chunk, "frames": frames,
             "offsets": [u[0] for u in units], "sizes": [u[1] for u in units], "keys": [u[2] for u in units],
             "codec": codec_string(data), "width": width, "height": height, "bytes": len(data),
             "contours": segment is not None, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out, f"{chunk:06d}.json.part"), "w") as handle:
        json.dump(index, handle)
    os.replace(os.path.join(out, f"{chunk:06d}.json.part"), os.path.join(out, f"{chunk:06d}.json"))
    return {"chunk": chunk, "frames": len(frames), "bytes": len(data), "seconds": index["seconds"]}, \
        (contours if segment is not None else None)


def pack_chunk(task):
    """Worker: one chunk's stream (task["build"]) and frame data (task["data"]: the state box_annotations needs,
    contours from the stream's build or its file). Returns a summary."""
    summary, contours = build_chunk(task) if task["build"] else ({"chunk": task["chunk"], "frames": 0, "bytes": 0}, None)
    if task["data"] is not None:
        t0 = time.time()
        if contours is None:
            path = os.path.join(task["out"], f"{task['chunk']:06d}.contours.npz")
            if os.path.exists(path):
                with np.load(path) as arrays:
                    contours = decode_contours({k: arrays[k] for k in arrays.files})
        state = dict(task["data"])
        size = tuple(state.pop("frame_size"))
        state["size"] = lambda: size
        if state.pop("bundle"):     # a minimal bundle of the Drive exchange: the publisher's contours (app._default_contours)
            from idtrackerai_validator_server.exchange_bundle import bundle_contours
            state["contours"] = lambda fn: bundle_contours(task["basedir"], fn)
        else:
            state["contours"] = (lambda fn: contours.get(fn, [])) if contours is not None else (lambda fn: [])
        version = task.get("continuity", 1)
        sig = chunk_sigs(state, [task["chunk"]], version)[task["chunk"]]
        summary["data_bytes"] = len(write_data(state, task["chunk"], task["frames"], sig, version))
        summary["data_seconds"] = round(time.time() - t0, 1)
    return summary


def _task(state, chunk, frames, store, stamp, build, data, size, version=1):
    idx = []
    if build:
        with sqlite3.connect(f"file:{os.path.join(state['basedir'], 'index.db')}?mode=ro", uri=True) as con:
            found = dict(con.execute("SELECT frame_number, frame_idx FROM frames WHERE frame_number BETWEEN ? AND ?",
                                     (frames[0], frames[-1])).fetchall())
        idx = [int(found[fn]) for fn in frames]
    from idtrackerai_validator_server.exchange_bundle import bundle_info
    bundle = bundle_info(state["basedir"]) is not None
    worker_state = {k: state[k] for k in ("experiment", "basedir", "dbfile", "chunksize", "n_animals")}
    worker_state["bundle"] = bundle
    return {"basedir": state["basedir"], "out": pack_dir(state["experiment"]), "chunk": chunk, "frames": frames,
            "frame_idx": idx, "stamp": stamp, "config": None if bundle else state.get("config"), "ffmpeg": lab_paths.FFMPEG,
            "build": build, "parent": os.getpid(), "data": {**worker_state, "frame_size": list(size)} if data else None,
            "continuity": version, **store}


# --- server side ----------------------------------------------------------------------------------------

def cached_contours(experiment, frame_number, chunksize):
    """The contours of a frame from its built chunk ({frame: arrays} of <chunk>.contours.npz), else None"""
    chunk = frame_number // chunksize
    path = os.path.join(pack_dir(experiment), f"{chunk:06d}.contours.npz")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    key = (to_flat(experiment), chunk, mtime)
    with _lock:
        hit = _contours.get(key)
        if hit is not None:
            _contours.move_to_end(key)
    if hit is None:
        try:
            with np.load(path) as arrays:
                hit = decode_contours({k: arrays[k] for k in arrays.files})
        except (OSError, ValueError, KeyError) as error:
            logger.warning("Cannot read %s: %s", path, error)
            return None
        with _lock:
            _contours[key] = hit
            while len(_contours) > CONTOURS_CHUNKS:
                _contours.popitem(last=False)
    return hit.get(frame_number)


def _data_path(experiment, chunk, version=1):
    return os.path.join(pack_dir(experiment), f"{chunk:06d}.data.c2.gz" if version == 2 else f"{chunk:06d}.data.gz")


def kept_sig(experiment, chunk, version=1):
    try:
        with open(_data_path(experiment, chunk, version) + ".sig") as handle:
            return handle.read().strip()
    except OSError:
        return None


def write_data(state, chunk, frames, sig, version=1):
    """Make the chunk's frame data (gzip of {sig, chunk, continuity, frames: {frame: /api/annotate/frame's answer}}
    with continuity proposals of tracker `version`), keep it in <chunk>.data[.c2].gz (+ .sig) and return it. sig:
    chunk_sigs, read before (a save meanwhile changes it)."""
    from idtrackerai_validator_server import box_annotations as ba
    payload = {"sig": sig, "chunk": chunk, "continuity": version, "frames": {}}
    for fn in frames:
        payload["frames"][str(fn)] = ba.frame_payload(state, fn, None, version)
    raw = gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), 6)
    path = _data_path(state["experiment"], chunk, version)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tag = f".{os.getpid()}.{threading.get_ident()}.part"
    with open(path + tag, "wb") as handle:
        handle.write(raw)
    os.replace(path + tag, path)
    with open(path + ".sig" + tag, "w") as handle:
        handle.write(sig)
    os.replace(path + ".sig" + tag, path + ".sig")
    return raw


def chunk_data(state, chunk, frames, sig, version=1):
    """gzip bytes of the chunk's frame data: the kept ones when made with this sig, else made now"""
    key = (to_flat(state["experiment"]), chunk)
    with _lock:
        lock = _chunk_locks.setdefault(key, threading.Lock())
    with lock:
        if kept_sig(state["experiment"], chunk, version) == sig:
            try:
                with open(_data_path(state["experiment"], chunk, version), "rb") as handle:
                    return handle.read()
            except OSError:
                pass
        return write_data(state, chunk, frames, sig, version)


def status(state, version=1):
    flat = to_flat(state["experiment"])
    folder = pack_dir(state["experiment"])
    by_chunk = pack_frames(state)
    stamp = video_stamp(state)
    sigs = chunk_sigs(state, by_chunk, version)
    chunks = []
    for chunk, frames in by_chunk.items():
        index = chunk_index(state, chunk, frames, stamp)
        data_sig = kept_sig(state["experiment"], chunk, version)
        chunks.append({"chunk": chunk, "frames": len(frames), "first": frames[0], "last": frames[-1],
                       "built": index is not None, "vkey": vkey(stamp, frames), "bytes": index["bytes"] if index else None,
                       "sig": sigs[chunk], "data_ready": data_sig == sigs[chunk] and index is not None,
                       "data_bytes": os.path.getsize(_data_path(state["experiment"], chunk, version))
                       if data_sig == sigs[chunk] else None})
    built = [c for c in chunks if c["built"]]
    job = _jobs.get(_running.get(flat))
    return {"experiment": flat, "folder": folder, "version": PACK_VERSION, "continuity": version, "chunks": chunks,
            "frames": sum(c["frames"] for c in chunks), "built_frames": sum(c["frames"] for c in built),
            "bytes": sum(c["bytes"] for c in built),
            "data_bytes": sum(c["data_bytes"] or 0 for c in chunks),
            "job": job.to_dict() if job is not None and job.status == "running" else None,
            "workers": WORKERS}


def prune(keep):
    """Delete the packs of other experiments untouched for KEEP_DAYS"""
    root = lab_paths.ANNOTATE_PACK_CACHE
    try:
        names = os.listdir(root)
    except OSError:
        return
    limit = time.time() - KEEP_DAYS * 86400
    for name in names:
        path = os.path.join(root, name)
        if name == keep or not os.path.isdir(path):
            continue
        newest = max([os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path)] or [os.path.getmtime(path)])
        if newest < limit:
            logger.info("Deleting the pre-download of %s (untouched for %d days)", name, KEEP_DAYS)
            shutil.rmtree(path, ignore_errors=True)


def start(state, order=None, data=True, version=1):
    """The job building the chunks not built yet (in `order` first) and their frame data. Returns (job, started)."""
    flat = to_flat(state["experiment"])
    with _lock:
        job = _jobs.get(_running.get(flat))
        if job is not None and job.status == "running":
            return job, False
    by_chunk = pack_frames(state)
    if not by_chunk:
        raise LookupError("no scenes to pre-download (Prepare scenes first)")
    order = [c for c in (order or []) if c in by_chunk]
    order += [c for c in sorted(by_chunk) if c not in set(order)]
    store = _store(state["basedir"])
    stamp = video_stamp(state)
    folder = pack_dir(state["experiment"])
    os.makedirs(folder, exist_ok=True)
    prune(flat)
    stop = threading.Event()
    total_frames = sum(len(by_chunk[c]) for c in order)

    def work(job):
        size = state["size"]()
        built = {c for c in order if chunk_index(state, c, by_chunk[c], stamp) is not None}
        sigs = chunk_sigs(state, order, version) if data else {}
        tasks = [(c, c not in built, data and (c not in built or kept_sig(state["experiment"], c, version) != sigs[c]))
                 for c in order]
        tasks = [t for t in tasks if t[1] or t[2]]
        done_frames = sum(len(by_chunk[c]) for c in built)
        data_total = sum(1 for t in tasks if t[2])
        data_done, summaries, failed = 0, [], []
        job.update(0, status="running", progress=done_frames / total_frames)
        job.update(1, status="running" if data_total else "done" if data else "skipped", progress=0 if data_total else 1)
        context = multiprocessing.get_context("spawn")   # the server has threads: no fork
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context, initializer=_init_worker) as pool:
            pending = {}
            queue = list(tasks)
            while queue or pending:
                while queue and len(pending) < WORKERS * 2 and not stop.is_set():
                    c, build, make = queue.pop(0)
                    task = _task(state, c, by_chunk[c], store, stamp, build, make, size, version)
                    pending[pool.submit(pack_chunk, task)] = (c, build, make)
                if stop.is_set():
                    for future in pending:
                        future.cancel()
                    break
                finished, _ = wait(list(pending), timeout=5, return_when=FIRST_COMPLETED)
                for future in finished:
                    c, build, make = pending.pop(future)
                    try:
                        summaries.append(future.result())
                    except Exception as error:
                        logger.exception("Pre-download: chunk %s failed", c)
                        failed.append({"chunk": c, "error": str(error)})
                    if build:
                        done_frames += len(by_chunk[c])
                        job.update(0, progress=done_frames / total_frames)
                    if make:
                        data_done += 1
                        job.update(1, progress=data_done / data_total)
        ended = "cancelled" if stop.is_set() else "failed" if failed else "done"
        job.update(0, status=ended)
        job.update(1, status=ended if data else "skipped")
        job.result = {"chunks": len(order), "encoded": sum(1 for t in tasks if t[1]), "data": data_total,
                      "failed": failed, "stopped": stop.is_set(),
                      "bytes": sum(e["bytes"] for e in summaries), "frames": sum(e["frames"] for e in summaries),
                      "data_bytes": sum(e.get("data_bytes", 0) for e in summaries)}
        if failed and not stop.is_set():
            raise RuntimeError(f"{len(failed)} chunk(s) failed, e.g. {failed[0]['chunk']}: {failed[0]['error']}")

    with _lock:
        job = start_job(f"Pre-download of {flat}: {total_frames} frames in {len(order)} chunks",
                        ["frames (H.264) + contours", "frame data"], work)
        _running[flat] = job.id
        _stop[flat] = stop
    return job, True


def _version(asked):
    """continuity version of a request (box_annotations.continuity_version: 1 / 2, else the server's default)"""
    from idtrackerai_validator_server import box_annotations as ba
    return ba.continuity_version(asked)


def register_annotate_pack(app, get_state):
    """get_state(): box_annotations' state (+ config: the idtrackerai config), or None"""

    def state_or_error():
        state = get_state()
        if state is None:
            return None, (jsonify({"error": "load an experiment with a tracking database"}), 400)
        return state, None

    @app.route("/api/annotate/pack", methods=["GET"])
    def pack_status():
        state, error = state_or_error()
        if error:
            return error
        return jsonify(status(state, _version(request.args.get("continuity"))))

    @app.route("/api/annotate/pack", methods=["POST"])
    def pack_start():
        """{"chunks": [order], "data": bool}: the job building what is missing (202), or the running one (200)"""
        state, error = state_or_error()
        if error:
            return error
        body = request.get_json(silent=True) or {}
        try:
            job, started = start(state, [int(c) for c in body.get("chunks") or []], bool(body.get("data", True)),
                                 _version(body.get("continuity")))
        except LookupError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"job": job.to_dict(), "started": started}), 202 if started else 200

    @app.route("/api/annotate/pack/stop", methods=["POST"])
    def pack_stop():
        state, error = state_or_error()
        if error:
            return error
        flat = to_flat(state["experiment"])
        job = _jobs.get(_running.get(flat))
        if job is None or job.status != "running":
            return jsonify({"error": "no pre-download being made"}), 409
        _stop[flat].set()
        return jsonify({"job": job.to_dict()})

    @app.route("/api/annotate/pack", methods=["DELETE"])
    def pack_delete():
        state, error = state_or_error()
        if error:
            return error
        flat = to_flat(state["experiment"])
        job = _jobs.get(_running.get(flat))
        if job is not None and job.status == "running":
            return jsonify({"error": "being made: stop it first"}), 409
        folder = pack_dir(state["experiment"])
        freed = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder)) if os.path.isdir(folder) else 0
        shutil.rmtree(folder, ignore_errors=True)
        with _lock:
            for key in [k for k in _contours if k[0] == flat]:
                del _contours[key]
        return jsonify({"freed": freed})

    def built(chunk):
        state, error = state_or_error()
        if error:
            return None, None, error
        index = chunk_index(state, chunk)
        if index is None:
            return None, None, (jsonify({"error": f"chunk {chunk} is not built (POST /api/annotate/pack)"}), 404)
        return state, index, None

    @app.route("/api/annotate/pack/<int:chunk>/index", methods=["GET"])
    def pack_index(chunk):
        _, index, error = built(chunk)
        return error or jsonify(index)

    @app.route("/api/annotate/pack/<int:chunk>/video", methods=["GET"])
    def pack_video(chunk):
        state, _, error = built(chunk)
        if error:
            return error
        path = os.path.join(pack_dir(state["experiment"]), f"{chunk:06d}.h264")
        return send_file(path, mimetype="application/octet-stream", conditional=True, max_age=0)

    @app.route("/api/annotate/pack/<int:chunk>/data", methods=["GET"])
    def pack_data(chunk):
        """gzip of {sig, chunk, frames: {frame: /api/annotate/frame's answer}} (sent as bytes: the browser keeps
        them compressed), made now unless kept with the sig of now"""
        state, _, error = built(chunk)
        if error:
            return error
        frames = pack_frames(state)[chunk]
        version = _version(request.args.get("continuity"))
        sig = chunk_sigs(state, [chunk], version)[chunk]
        raw = chunk_data(state, chunk, frames, sig, version)
        response = Response(raw, mimetype="application/gzip")
        response.headers["X-Data-Sig"] = sig
        response.headers["Cache-Control"] = "no-store"
        return response
