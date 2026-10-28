"""SLEAP ground truth labels of the loaded experiment, made in the "SLEAP labels" tab.

The pose model is trained on the .slp files of $SLEAP_LABELS_DIR (labels_train*.slp,
labels_test*.slp; made with sleap-label, SLEAP 1.3.3). Each of their videos is one chunk
of one fly, a copy of <basedir>/flyhostel/single_animal/<lid:03d>/<chunk:06d>.mp4 kept as

    SLEAP_DATA/<experiment>_<chunk:06d>_<local identity:03d>/<same>.mp4

(older entries: .../<chunk:06d>.mp4, or <experiment>_<chunk:06d>/<lid:03d>/<chunk:06d>.mp4;
paths relative to the .slp's folder or absolute), so frame_idx i of a video is frame
chunk * chunksize + i of the experiment, and its coordinates are those of the pose file
(relative to the fly's square). The local identity is the fly's idtrackerai identity in
that chunk (deg_labels._local_identities).

The files are read and written with sleap-io (0.9.5: its files load in SLEAP 1.3.3,
checked frame by frame), videos left closed (open_videos=False: saving then takes ~4 s
instead of ~20 s for the 530 videos of labels_train_v6). A file stays in memory; each
change marks it dirty and a saver thread writes it SAVE_DELAY s after the last change
(atomic: a temporary file next to it, then os.replace). Before the first write of each
file since the server started, the file on disk is copied to $SLEAP_BACKUPS (the last
KEEP_BACKUPS per file are kept). A file changed on disk since it was read is reloaded
when nothing is pending, and never overwritten otherwise (the save fails; reload drops
the changes). Every change is logged to $SLEAP_LABELS_LOG.

A labelled frame keeps one user instance (SLEAP's single-instance models); writing one
replaces it, linked to the frame's predicted instance (from_predicted) when there is one.
A frame of a chunk not in the file yet adds its video (copied into SLEAP_DATA).

    GET  /api/sleap/files                          the .slp files, newest training file first
    GET  /api/sleap/labels?file=                   skeleton, flies, the labelled and suggested
                                                   frames of the loaded experiment
    GET  /api/sleap/locate?fly=&frame_number=      chunk, local identity, frame_idx of a fly's frame
    GET  /api/sleap/frame?file=&chunk=&lid=&idx=   user / predicted instance, pose file prediction,
                                                   template (mean instance laid on the fly)
    GET  /api/sleap/image?file=&chunk=&lid=&idx=   PNG of the video frame
    POST /api/sleap/frame                          {file, chunk, lid, idx, points: [[x, y, visible]]
                                                   per node or null to delete the user instance}
    POST /api/sleap/save, /api/sleap/reload        {file}: write now / drop unsaved changes
    GET  /api/sleap/status?file=                   dirty, saving, last save, error
"""
import glob
import json
import logging
import os
import re
import shutil
import threading
import time
import warnings
from collections import OrderedDict

import cv2
import numpy as np
import sleap_io as sio
from flask import Response, jsonify, request

from flyhostel.utils import get_basedir

from idtrackerai_validator_server.deg_labels import _local_identities
from idtrackerai_validator_server.experiments import to_flat
from idtrackerai_validator_server.pose import _flies, first_frame, get_h5_file, node_names, pose_path

logger = logging.getLogger(__name__)

LABELS_DIR = os.environ.get("SLEAP_LABELS_DIR", "/flyhostel_data/fiftyone/FlyBehaviors")
BACKUP_DIR = os.environ.get("SLEAP_BACKUPS", os.path.expanduser("~/.cache/idtrackerai-validator/slp_backups"))
LOG_PATH = os.environ.get("SLEAP_LABELS_LOG", os.path.expanduser("~/.cache/idtrackerai-validator/sleap_labels.jsonl"))
SAVE_DELAY = 2.0
KEEP_BACKUPS = 20
FILE_RE = re.compile(r"^labels_(train|test)[^/]*\.slp$")
# <experiment>_<chunk>_<lid> or <experiment>_<chunk>/<lid> anywhere in a video path
VIDEO_RE = re.compile(r"(FlyHostel\d+_\d+X_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_arena\d+)?)_(\d{6})[_/](\d{3})(?=[/.]|$)")
# videos_json of the videos sleap-label adds (sleap-io only writes what it is given)
BACKEND_METADATA = {"grayscale": True, "bgr": True, "dataset": "", "input_format": ""}
# head-abdomen distance / length of the fly's dark blob (4 sd along its long axis):
# median of 400 labelled frames of labels_train_v6 (IQR 0.61-0.67, alike in every experiment)
BODY_FRACTION = 0.64


def list_files():
    """.slp files of LABELS_DIR: training files first, newest first."""
    files = []
    for path in glob.glob(os.path.join(LABELS_DIR, "*.slp")):
        name = os.path.basename(path)
        if FILE_RE.match(name):
            stat = os.stat(path)
            files.append({"name": name, "mtime": stat.st_mtime, "size": stat.st_size,
                          "train": name.startswith("labels_train")})
    files.sort(key=lambda f: (not f["train"], -f["mtime"]))
    return files


def file_path(name):
    if not name or not FILE_RE.match(name) or not os.path.exists(os.path.join(LABELS_DIR, name)):
        raise FileNotFoundError(f"No SLEAP labels file {name!r} in {LABELS_DIR}")
    return os.path.join(LABELS_DIR, name)


def video_key(filename):
    """(experiment, chunk, local identity) of a video path, or None."""
    match = VIDEO_RE.search(filename)
    return (match.group(1), int(match.group(2)), int(match.group(3))) if match else None


def folder_key(experiment, chunk, lid):
    return f"{experiment}_{chunk:06d}_{lid:03d}"


def single_animal_video(experiment, chunk, lid):
    return os.path.join(get_basedir(experiment), "flyhostel", "single_animal", f"{lid:03d}", f"{chunk:06d}.mp4")


def fly_axis(frame):
    """Centre (x, y), angle (rad) and length (4 sd) of the long axis of the dark blob
    nearest the centre of a crop (Otsu threshold, image moments), or None. The head
    may be at either end."""
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = found[-2]   # (image, contours, hierarchy) in OpenCV 3, (contours, hierarchy) in 4
    h, w = frame.shape[:2]
    best = None
    for contour in contours:
        m = cv2.moments(contour)
        if m["m00"] < 20:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        distance = np.hypot(cx - w / 2, cy - h / 2)
        if best is None or distance < best[0]:
            spread = np.sqrt(4 * m["mu11"] ** 2 + (m["mu20"] - m["mu02"]) ** 2)
            length = 4 * np.sqrt((m["mu20"] + m["mu02"] + spread) / 2 / m["m00"])
            best = (distance, cx, cy, 0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]), length)
    return None if best is None else best[1:]


def place_template(template, nodes, frame):
    """[x, y, visible] per node: the egocentric template (in head-abdomen lengths) laid
    on the fly of the frame: the head-abdomen midpoint on the blob's centre, the head
    along its long axis, scaled to BODY_FRACTION of its length. Nodes without template
    position at the centre and not visible."""
    h, w = frame.shape[:2]
    axis = fly_axis(frame)
    cx, cy, angle, length = axis if axis is not None else (w / 2, h / 2, 0.0, min(w, h) / 2)
    centre = (template[nodes.index("head")] + template[nodes.index("abdomen")]) / 2
    c, s = np.cos(angle), np.sin(angle)
    placed = (template - centre) * BODY_FRACTION * length @ np.array([[c, s], [-s, c]]) + [cx, cy]
    return [[_round(x), _round(y), True] if np.isfinite(x) else [_round(w / 2), _round(h / 2), False] for x, y in placed]


def _round(value, digits=2):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def _log(entry):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as handle:
            handle.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}) + "\n")
    except OSError as error:
        logger.warning("Could not log to %s: %s", LOG_PATH, error)


class LabelsFile:
    """One .slp file in memory, with an index (experiment, chunk, lid) -> video and
    (video, frame_idx) -> labeled frame, saved in the background after changes."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.dirty = 0              # changes not written yet
        self.saving = False
        self.last_saved = None
        self.error = None
        self.backed_up = False
        self._timer = None
        self._load()

    def _load(self):
        started = time.time()
        self.mtime = os.path.getmtime(self.path)
        self.labels = sio.load_slp(self.path, open_videos=False)
        self.skeleton = self.labels.skeleton
        self.nodes = list(self.skeleton.node_names)
        self.videos = {}
        for video in self.labels.videos:
            key = video_key(video.filename)
            if key is not None:
                self.videos.setdefault(key, video)
        self.frames = {(id(lf.video), int(lf.frame_idx)): lf for lf in self.labels.labeled_frames}
        self.suggested = {(id(s.video), int(s.frame_idx)) for s in self.labels.suggestions}
        self.template = self._egocentric_template()
        self.dirty = 0
        self.error = None
        logger.info("Read %s in %.1f s: %s", self.path, time.time() - started, self.labels)

    def refresh(self):
        """Read the file again when it changed on disk and nothing is pending."""
        with self.lock:
            try:
                mtime = os.path.getmtime(self.path)
            except OSError:
                return
            if mtime != self.mtime and not self.dirty and not self.saving:
                logger.info("%s changed on disk, reading it again", self.path)
                self._load()

    def _egocentric_template(self):
        """Mean of the user instances in the fly's frame of reference (thorax at the
        origin, head along +x; rotated only, so left stays left), in head-abdomen
        lengths (the videos differ in scale), NaN for nodes never visible: where a new
        instance without prediction starts (place_template)."""
        names = self.nodes
        if not {"thorax", "head", "abdomen"} <= set(names):
            return None
        thorax, head, abdomen = names.index("thorax"), names.index("head"), names.index("abdomen")
        aligned = []
        for lf in self.labels.labeled_frames:
            for inst in lf.user_instances:
                xy = np.array(inst.points["xy"], dtype=float)
                xy[~inst.points["visible"]] = np.nan
                axis = xy[head] - xy[thorax]
                body = np.hypot(*(xy[head] - xy[abdomen]))
                if not np.isfinite(axis).all() or not axis.any() or not body > 0:
                    continue
                angle = np.arctan2(axis[1], axis[0])
                c, s = np.cos(-angle), np.sin(-angle)
                aligned.append((xy - xy[thorax]) @ np.array([[c, s], [-s, c]]) / body)
        if not aligned:
            return None
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # nodes never visible: NaN
            return np.nanmean(np.array(aligned), axis=0)

    def resolve(self, video):
        """Video path on disk (relative paths are relative to the .slp's folder)."""
        filename = video.filename
        return filename if os.path.isabs(filename) else os.path.join(os.path.dirname(self.path), filename)

    def ensure_video(self, experiment, chunk, lid):
        """The file's video of that chunk of that fly, added (copied into SLEAP_DATA) if missing."""
        key = (experiment, chunk, lid)
        video = self.videos.get(key)
        if video is not None:
            return video
        source = single_animal_video(experiment, chunk, lid)
        if not os.path.exists(source):
            raise FileNotFoundError(f"No single-animal video of local identity {lid} in chunk {chunk}: {source}")
        name = folder_key(*key)
        target = os.path.join(os.path.dirname(self.path), "SLEAP_DATA", name, f"{name}.mp4")
        if not (os.path.exists(target) and os.path.getsize(target) == os.path.getsize(source)):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            tmp = f"{target}.partial"
            shutil.copy2(source, tmp)
            os.replace(tmp, target)
            logger.info("Copied %s to %s", source, target)
        video = sio.Video(filename=target, backend_metadata={"filename": target, **BACKEND_METADATA}, open_backend=False)
        self.labels.videos.append(video)
        self.videos[key] = video
        return video

    def frame_of(self, key, idx):
        video = self.videos.get(key)
        return video, (self.frames.get((id(video), idx)) if video is not None else None)

    def set_user_instance(self, key, idx, points):
        """Replace the frame's user instance by `points` ([x, y, visible] per node, x/y
        null where never placed), or delete it (points None)."""
        with self.lock:
            if points is None:
                video, lf = self.frame_of(key, idx)
                if lf is None or not lf.user_instances:
                    return False
                lf.instances = [inst for inst in lf.instances if isinstance(inst, sio.PredictedInstance)]
                if not lf.instances:
                    self.labels.labeled_frames.remove(lf)
                    del self.frames[(id(video), idx)]
            else:
                if len(points) != len(self.nodes):
                    raise ValueError(f"{len(points)} points for {len(self.nodes)} nodes")
                xy = np.array([[np.nan if p[0] is None else p[0], np.nan if p[1] is None else p[1]] for p in points], dtype=float)
                visible = np.array([bool(p[2]) for p in points]) & np.isfinite(xy).all(axis=1)
                video = self.ensure_video(*key)
                lf = self.frames.get((id(video), idx))
                if lf is None:
                    lf = sio.LabeledFrame(video=video, frame_idx=idx)
                    self.labels.labeled_frames.append(lf)
                    self.frames[(id(video), idx)] = lf
                predicted = next(iter(lf.predicted_instances), None)
                instance = sio.Instance.from_numpy(xy, skeleton=self.skeleton, from_predicted=predicted)
                # sleap-label keeps the position of points marked not visible
                instance.points["xy"] = xy
                instance.points["visible"] = visible
                instance.points["complete"] = np.isfinite(xy).all(axis=1)
                kept = [inst for inst in lf.instances if isinstance(inst, sio.PredictedInstance)]
                lf.instances = [instance] + kept
            self.dirty += 1
            self.error = None
            self._schedule()
            return True

    def _schedule(self):
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(SAVE_DELAY, self._save_quietly)
        self._timer.daemon = True
        self._timer.start()

    def _save_quietly(self):
        try:
            self.save()
        except Exception as error:
            logger.exception("Could not save %s", self.path)
            self.error = str(error)

    def _backup(self):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        name, ext = os.path.splitext(os.path.basename(self.path))
        shutil.copy2(self.path, os.path.join(BACKUP_DIR, f"{name}.{time.strftime('%Y%m%d-%H%M%S')}{ext}"))
        backups = sorted(glob.glob(os.path.join(BACKUP_DIR, f"{glob.escape(name)}.*{ext}")))
        for old in backups[:-KEEP_BACKUPS]:
            os.remove(old)

    def save(self):
        with self.lock:
            if not self.dirty:
                return False
            if self._timer is not None:
                self._timer.cancel()
            if os.path.getmtime(self.path) != self.mtime:
                raise RuntimeError(f"{os.path.basename(self.path)} changed on disk since it was read: "
                                   "not overwritten (Reload drops the changes made here)")
            self.saving = True
            try:
                started = time.time()
                if not self.backed_up:
                    self._backup()
                    self.backed_up = True
                folder, name = os.path.split(self.path)
                tmp = os.path.join(folder, f".{name}.saving.slp")
                sio.save_slp(self.labels, tmp, verbose=False)
                os.replace(tmp, self.path)
                self.mtime = os.path.getmtime(self.path)
                self.dirty = 0
                self.error = None
                self.last_saved = time.time()
                logger.info("Saved %s in %.1f s", self.path, time.time() - started)
                return True
            finally:
                self.saving = False

    def status(self):
        return {"file": os.path.basename(self.path), "dirty": self.dirty, "saving": self.saving,
                "last_saved": self.last_saved, "error": self.error, "mtime": self.mtime}


_files = {}
_files_lock = threading.Lock()


def get_file(name):
    path = file_path(name)
    with _files_lock:
        labels_file = _files.get(path)
        if labels_file is None:
            labels_file = _files[path] = LabelsFile(path)
    labels_file.refresh()
    return labels_file


class VideoFrames:
    """Frames of SLEAP videos (cv2, as sleap-label 1.3.3 reads them), the last few
    captures and decoded frames cached."""

    def __init__(self, n_captures=6, n_frames=64):
        self._caps = OrderedDict()   # path -> [capture, next index]
        self._frames = OrderedDict()
        self._lock = threading.Lock()
        self._n_caps, self._n_frames = n_captures, n_frames

    def get(self, path, index):
        key = (path, index)
        with self._lock:
            if key in self._frames:
                self._frames.move_to_end(key)
                return self._frames[key]
            entry = self._caps.get(path)
            if entry is None:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"No video {path}")
                entry = self._caps[path] = [cv2.VideoCapture(path), 0]
                while len(self._caps) > self._n_caps:
                    self._caps.popitem(last=False)[1][0].release()
            self._caps.move_to_end(path)
            cap = entry[0]
            if entry[1] != index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            entry[1] = index + 1 if ok else -1
            if not ok:
                raise ValueError(f"Frame {index} of {path} cannot be read")
            frame = frame[:, :, 0] if frame.ndim == 3 else frame
            self._frames[key] = frame
            while len(self._frames) > self._n_frames:
                self._frames.popitem(last=False)
            return frame


FRAMES = VideoFrames()


class NoExperiment(Exception):
    pass


def register_sleap_labels(app, get_state):
    """get_state() -> None (no database loaded) or a dict with experiment and chunksize."""

    def context():
        state = get_state()
        if state is None:
            raise NoExperiment("No experiment with database loaded")
        return to_flat(state["experiment"]), int(state["chunksize"]), state

    def identity_of(experiment, chunk, lid, chunksize):
        """Fly id of a local identity in a chunk, or None."""
        table = _local_identities(experiment, chunk, chunksize)
        identity = next((i for i, l in table.items() if l == lid), None)
        return None if identity is None else f"{experiment}__{identity:02d}"

    def image_path(labels_file, experiment, chunk, lid):
        """The file's video of the chunk (what sleap-label shows), else the single-animal video."""
        video = labels_file.videos.get((experiment, chunk, lid))
        path = labels_file.resolve(video) if video is not None else None
        return path if path is not None and os.path.exists(path) else single_animal_video(experiment, chunk, lid)

    def key_args(source):
        return int(source["chunk"]), int(source["lid"]), int(source["idx"])

    def errors(handler):
        def wrapped(*args, **kwargs):
            try:
                return handler(*args, **kwargs)
            except NoExperiment as error:
                return jsonify({"error": str(error)}), 409
            except (KeyError, ValueError, TypeError) as error:   # before LookupError, KeyError's base
                return jsonify({"error": f"Bad request: {error}"}), 400
            except (LookupError, FileNotFoundError) as error:
                return jsonify({"error": str(error)}), 404
            except Exception as error:
                logger.exception("SLEAP labels request failed")
                return jsonify({"error": str(error)}), 500
        wrapped.__name__ = handler.__name__
        return wrapped

    def points_of(instance, names, scores=False):
        out = []
        for point in instance.points:
            x, y = point["xy"]
            if scores:
                out.append(None if not point["visible"] or not np.isfinite(x) else [_round(x, 4), _round(y, 4), _round(point["score"], 3)])
            else:
                # 4 decimals: points sent back unchanged are written back (almost) as they were
                out.append([_round(x, 4), _round(y, 4), bool(point["visible"])])
        return out

    def pose_file_points(state, fly, frame_number, nodes):
        """[x, y, confidence] per node from the fly's compiled pose file, or None (a missing
        file is not compiled here: get_h5_file would)."""
        nn = fly.split("__")[1]
        if not os.path.exists(pose_path(state["experiment"], nn)):
            return None
        h5_file = get_h5_file(nn, state["experiment"])
        if h5_file is None:
            return None
        i = frame_number - first_frame(h5_file, state["chunksize"])
        if not 0 <= i < h5_file["point_scores"].shape[2]:
            return None
        index = {name: k for k, name in enumerate(node_names(h5_file))}
        xy = h5_file["tracks"][0, :, :, i]
        scores = h5_file["point_scores"][0, :, i]
        out = []
        for name in nodes:
            k = index.get(name)
            if k is None or not np.isfinite(xy[0, k]):
                out.append(None)
            else:
                out.append([_round(xy[0, k]), _round(xy[1, k]), _round(scores[k], 3)])
        return out if any(out) else None

    @app.route("/api/sleap/files", methods=["GET"])
    @errors
    def sleap_files():
        files = list_files()
        loaded = {os.path.basename(path): f.status() for path, f in _files.items()}
        return jsonify({"folder": LABELS_DIR, "files": [{**f, "loaded": loaded.get(f["name"])} for f in files]})

    @app.route("/api/sleap/labels", methods=["GET"])
    @errors
    def sleap_labels():
        experiment, chunksize, state = context()
        labels_file = get_file(request.args["file"])
        with labels_file.lock:
            frames = []
            for (exp, chunk, lid), video in labels_file.videos.items():
                if exp != experiment:
                    continue
                fly = identity_of(exp, chunk, lid, chunksize)
                indices = {idx for (vid, idx) in labels_file.frames if vid == id(video)}
                indices |= {idx for (vid, idx) in labels_file.suggested if vid == id(video)}
                for idx in sorted(indices):
                    lf = labels_file.frames.get((id(video), idx))
                    frames.append({
                        "chunk": chunk, "lid": lid, "idx": idx, "fly": fly, "frame_number": chunk * chunksize + idx,
                        "user": bool(lf and lf.user_instances), "predicted": bool(lf and lf.predicted_instances),
                        "suggested": (id(video), idx) in labels_file.suggested,
                    })
            frames.sort(key=lambda f: (f["fly"] or "", f["frame_number"]))
            labels = labels_file.labels
            return jsonify({
                **labels_file.status(),
                "experiment": experiment,
                "nodes": labels_file.nodes,
                "edges": [[e.source.name, e.destination.name] for e in labels_file.skeleton.edges],
                "flies": [fly for fly, _ in _flies(state["experiment"])],
                "frames": frames,
                "totals": {"videos": len(labels.videos), "frames": len(labels.labeled_frames),
                           "user_frames": sum(1 for lf in labels.labeled_frames if lf.user_instances),
                           "suggestions": len(labels.suggestions)},
            })

    @app.route("/api/sleap/locate", methods=["GET"])
    @errors
    def sleap_locate():
        experiment, chunksize, _ = context()
        fly, frame_number = request.args["fly"], int(request.args["frame_number"])
        if fly.split("__")[0] != experiment:
            raise LookupError(f"{fly} is not a fly of the loaded experiment")
        chunk, idx = divmod(frame_number, chunksize)
        lid = _local_identities(experiment, chunk, chunksize).get(int(fly.split("__")[1]))
        if lid is None:
            raise LookupError(f"{fly} has no local identity in chunk {chunk}")
        return jsonify({"fly": fly, "chunk": chunk, "lid": lid, "idx": idx, "frame_number": frame_number,
                        "has_video": os.path.exists(single_animal_video(experiment, chunk, lid))})

    @app.route("/api/sleap/frame", methods=["GET"])
    @errors
    def sleap_frame():
        experiment, chunksize, state = context()
        chunk, lid, idx = key_args(request.args)
        labels_file = get_file(request.args["file"])
        fly = identity_of(experiment, chunk, lid, chunksize)
        frame_number = chunk * chunksize + idx
        with labels_file.lock:
            video, lf = labels_file.frame_of((experiment, chunk, lid), idx)
            user = next(iter(lf.user_instances), None) if lf else None
            predicted = next(iter(lf.predicted_instances), None) if lf else None
            out = {
                "chunk": chunk, "lid": lid, "idx": idx, "fly": fly, "frame_number": frame_number,
                "video": video.filename if video is not None else None,
                "suggested": video is not None and (id(video), idx) in labels_file.suggested,
                "user": points_of(user, labels_file.nodes) if user else None,
                "predicted": {"points": points_of(predicted, labels_file.nodes, scores=True),
                              "score": _round(predicted.score, 3)} if predicted else None,
            }
        out["pose_file"] = pose_file_points(state, fly, frame_number, labels_file.nodes) if fly else None
        out["template"] = None
        if labels_file.template is not None:
            try:
                out["template"] = place_template(labels_file.template, labels_file.nodes,
                                                 FRAMES.get(image_path(labels_file, experiment, chunk, lid), idx))
            except (FileNotFoundError, ValueError) as error:
                logger.warning("No template for chunk %s, local id %s, index %s: %s", chunk, lid, idx, error)
        return jsonify(out)

    @app.route("/api/sleap/image", methods=["GET"])
    @errors
    def sleap_image():
        experiment, _, _ = context()
        chunk, lid, idx = key_args(request.args)
        labels_file = get_file(request.args["file"])
        ok, png = cv2.imencode(".png", FRAMES.get(image_path(labels_file, experiment, chunk, lid), idx))
        if not ok:
            raise RuntimeError("Could not encode the frame")
        return Response(png.tobytes(), mimetype="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @app.route("/api/sleap/frame", methods=["POST"])
    @errors
    def sleap_frame_post():
        experiment, chunksize, _ = context()
        body = request.get_json(force=True)
        chunk, lid, idx = key_args(body)
        if not 0 <= idx < chunksize:
            raise ValueError(f"frame index {idx} outside the chunk")
        labels_file = get_file(body["file"])
        points = body.get("points")
        changed = labels_file.set_user_instance((experiment, chunk, lid), idx, points)
        if changed:
            _log({"file": labels_file.path, "experiment": experiment, "chunk": chunk, "lid": lid, "idx": idx,
                  "action": "delete" if points is None else "set",
                  "visible": None if points is None else sum(1 for p in points if p[2] and p[0] is not None)})
        _, lf = labels_file.frame_of((experiment, chunk, lid), idx)
        return jsonify({**labels_file.status(), "changed": changed,
                        "user": bool(lf and lf.user_instances), "predicted": bool(lf and lf.predicted_instances)})

    @app.route("/api/sleap/save", methods=["POST"])
    @errors
    def sleap_save():
        labels_file = get_file(request.get_json(force=True)["file"])
        try:
            labels_file.save()
        except RuntimeError as error:
            labels_file.error = str(error)
            return jsonify({**labels_file.status()}), 409
        return jsonify(labels_file.status())

    @app.route("/api/sleap/reload", methods=["POST"])
    @errors
    def sleap_reload():
        labels_file = get_file(request.get_json(force=True)["file"])
        with labels_file.lock:
            if labels_file._timer is not None:
                labels_file._timer.cancel()
            dropped = labels_file.dirty
            labels_file._load()
        if dropped:
            _log({"file": labels_file.path, "action": "reload", "dropped": dropped})
        return jsonify({**labels_file.status(), "dropped": dropped})

    @app.route("/api/sleap/status", methods=["GET"])
    @errors
    def sleap_status():
        return jsonify(get_file(request.args["file"]).status())
