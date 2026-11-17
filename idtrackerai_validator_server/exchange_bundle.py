"""
exchange_bundle.py  —  the "minimal" kind of the Drive exchange (gdrive_exchange.py): what the
Annotate tab needs of an experiment's scenes, instead of the whole folder (60 GB → a few hundred MB).

A bundle is laid out as an experiment folder, so the server opens it as one:

    bundle.json                       kind "minimal", scenes, frames, first_frame, per chunk the frames of its clip
    <flat>.db                         the database cut down to the scene frames ± DB_MARGIN (tables with a
                                      frame_number column), small tables whole, big ones without frame_number empty
    metadata.yaml, <date>.conf        the store's metadata and the idtrackerai config
    index.db                          imgstore's `frames` table of the clips: chunk, frame_idx (in the clip),
                                      frame_number, frame_time
    <chunk:06d>.mp4                   per chunk with scenes a clip of its scene frames ± IMAGE_MARGIN, in frame
                                      order (H.264 yuv420p, crf CRF, a keyframe every GOP frames: frame-exact
                                      seeks with OpenCV, ~45 dB PSNR, ~15 kB per 2100 px frame)
    flyhostel/validation/scenes.json  the scenes as the Annotate tab lists them here (scene_qc.csv + the unsure
                                      chunk boundaries, which need the whole database)
    flyhostel/validation/contours.db  `contours` (frame_number, zlib of json): the blobs idtrackerai segments in the
                                      original frames (simplified to 1 px), for the boxes proposed from the
                                      tracking (the clips' pixels differ slightly, and annotators may not have
                                      idtrackerai)

The server reads it with lite_store.LiteVideoCapture(exact=True) (a frame outside the bundle is
blank), takes the scenes from scenes.json and the contours from contours.db (backend.bundle_info,
app._default_contours, box_annotations.read_scenes).

The bundle is built in $EXCHANGE_CACHE/bundles/<flat>/ and kept: publishing again re-encodes only
the clips whose frames changed and segments only frames without contours.
"""
import os
import json
import time
import shutil
import sqlite3
import logging
import subprocess
import zlib

import cv2
import numpy as np
import yaml

from idtrackerai_validator_server.compat import HAS_IDTRACKERAI, get_number_of_animals
from idtrackerai_validator_server.experiments import to_flat

logger = logging.getLogger(__name__)

BUNDLE_FILE = "bundle.json"
SCENES_FILE = "scenes.json"
CONTOURS_FILE = "contours.db"
FORMAT = 1
IMAGE_MARGIN = 2           # frames shown around each scene (the tab steps a frame or two out of it)
DB_MARGIN = 1              # rows kept around the image frames (fragment events look at the frame before / after)
SMALL_TABLE_ROWS = 100000  # tables without frame_number copied whole up to this many rows
SKIP_TABLES = {"IMG_SNAPSHOTS"}
CRF = int(os.environ.get("EXCHANGE_CLIP_CRF", "18"))
GOP = 24
SEEK_GAP = 300             # a gap of more frames is a seek, else frames are read forward
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")


def bundle_info(basedir):
    """bundle.json of a minimal bundle, else None"""
    try:
        with open(os.path.join(basedir, BUNDLE_FILE)) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _validation(basedir):
    return os.path.join(basedir, "flyhostel", "validation")


def read_bundle_scenes(basedir):
    path = os.path.join(_validation(basedir), SCENES_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def bundle_contours(basedir, frame_number):
    """Contours of a bundle's frame (contours.db), [] when it has none"""
    path = os.path.join(_validation(basedir), CONTOURS_FILE)
    if not os.path.exists(path):
        return []
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    try:
        row = con.execute("SELECT contours FROM contours WHERE frame_number = ?", (int(frame_number),)).fetchone()
    finally:
        con.close()
    if not row:
        return []
    value = row[0]
    return json.loads(zlib.decompress(value) if isinstance(value, bytes) else value)


# ---------------------------------------------------------------- export

def scene_list(basedir, experiment, dbfile, chunksize, n_animals):
    """The scenes the Annotate tab lists for the full experiment"""
    from idtrackerai_validator_server import box_annotations as boxes_mod
    scenes, source = boxes_mod.read_scenes(basedir, experiment, n_animals)
    boundaries = boxes_mod.unsure_boundaries(dbfile, chunksize, n_animals)
    return boxes_mod.add_boundary_scenes(scenes, boundaries, chunksize), source


def _frames_of(scenes, margin, lo, hi):
    out = set()
    for sc in scenes:
        start = max(lo, sc["start"] - margin)
        end = min(hi, sc["start"] + sc["length"] - 1 + margin)
        out.update(range(start, end + 1))
    return out


def cut_database(src, dst, frames):
    """dst: src with only the rows of `frames` in tables having frame_number (same schema and indices)"""
    if os.path.exists(dst):
        os.remove(dst)
    out = sqlite3.connect(f"file:{dst}", uri=True)
    out.execute("ATTACH DATABASE ? AS src", (f"file:{src}?mode=ro",))
    try:
        out.execute("CREATE TEMP TABLE wanted (frame_number INTEGER PRIMARY KEY)")
        out.executemany("INSERT INTO wanted VALUES (?)", [(f,) for f in sorted(frames)])
        objects = out.execute("SELECT type, name, tbl_name, sql FROM src.sqlite_master WHERE sql IS NOT NULL "
                              "AND name NOT LIKE 'sqlite_%' ORDER BY type = 'index'").fetchall()
        counts = {}
        for kind, name, table, sql in objects:
            if kind != "table":
                continue
            out.execute(sql)
            if name in SKIP_TABLES:
                continue
            columns = [row[1] for row in out.execute(f'PRAGMA src.table_info("{name}")')]
            if "frame_number" in columns:
                out.execute(f'INSERT INTO main."{name}" SELECT * FROM src."{name}" '
                            f'WHERE frame_number IN (SELECT frame_number FROM temp.wanted)')
            else:
                n = out.execute(f'SELECT COUNT(*) FROM (SELECT 1 FROM src."{name}" LIMIT {SMALL_TABLE_ROWS + 1})').fetchone()[0]
                if n <= SMALL_TABLE_ROWS:
                    out.execute(f'INSERT INTO main."{name}" SELECT * FROM src."{name}"')
            counts[name] = out.execute(f'SELECT COUNT(*) FROM main."{name}"').fetchone()[0]
        for kind, name, table, sql in objects:
            if kind == "index":
                out.execute(sql)
        out.commit()
        return counts
    finally:
        out.close()


class _SourceReader:
    """Frames of the original recording in increasing order: read forward, seek over big gaps"""

    def __init__(self, basedir, extension):
        self.basedir, self.extension = basedir, extension
        self.chunk, self.capture, self.pos = None, None, None

    def read(self, chunk, frame_idx):
        if chunk != self.chunk:
            if self.capture is not None:
                self.capture.release()
            path = os.path.join(self.basedir, f"{int(chunk):06d}{self.extension}")
            self.capture = cv2.VideoCapture(path)
            if not self.capture.isOpened():
                raise FileNotFoundError(f"cannot open {path}")
            self.chunk, self.pos = chunk, 0
        if frame_idx < self.pos or frame_idx - self.pos > SEEK_GAP:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.pos = frame_idx
        while self.pos < frame_idx:
            self.capture.grab()
            self.pos += 1
        ok, frame = self.capture.read()
        self.pos += 1
        if not ok:
            raise ValueError(f"cannot read frame {frame_idx} of chunk {chunk}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def close(self):
        if self.capture is not None:
            self.capture.release()


def _encoder(path, width, height, framerate):
    return subprocess.Popen(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}",
         "-r", str(framerate), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF), "-g", str(GOP),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _simplify(contours):
    out = []
    for contour in contours:
        poly = np.asarray(contour, dtype=np.int32).reshape((-1, 1, 2))
        if len(poly) > 3:
            poly = cv2.approxPolyDP(poly, 1.0, True)
        out.append(poly.reshape((-1, 2)).tolist())
    return out


def export(basedir, outdir, progress=lambda fraction, text=None: None):
    """Build (or update) the minimal bundle of the experiment in `basedir` into `outdir` -> bundle.json's content"""
    from idtrackerai_validator_server import backend
    flat = to_flat(basedir)
    dbfile = os.path.join(basedir, flat + ".db")
    if not os.path.exists(dbfile):
        raise FileNotFoundError(f"{dbfile} not found: a minimal bundle needs the tracking database")
    with open(os.path.join(basedir, "metadata.yaml")) as handle:
        store = yaml.safe_load(handle)["__store"]
    chunksize, framerate = int(float(store["chunksize"])), float(store["framerate"])
    extension = store.get("extension", ".mp4")
    n_animals = int(get_number_of_animals(flat))
    os.makedirs(_validation(outdir), exist_ok=True)
    previous = bundle_info(outdir) or {}

    progress(0.01, "scenes")
    scenes, source = scene_list(basedir, flat, dbfile, chunksize, n_animals)
    if not scenes:
        raise RuntimeError("no scenes to annotate: prepare them in the Annotate tab first")
    with sqlite3.connect(f"file:{os.path.join(basedir, 'index.db')}?mode=ro", uri=True) as con:
        lo, hi = con.execute("SELECT MIN(frame_number), MAX(frame_number) FROM frames").fetchone()
    image_frames = _frames_of(scenes, IMAGE_MARGIN, lo, hi)
    db_frames = _frames_of(scenes, IMAGE_MARGIN + DB_MARGIN, lo, hi)

    progress(0.03, "database")
    counts = cut_database(dbfile, os.path.join(outdir, flat + ".db"), db_frames)

    # source frames: chunk, frame_idx, frame_time
    index = {}
    with sqlite3.connect(f"file:{os.path.join(basedir, 'index.db')}?mode=ro", uri=True) as con:
        con.execute("CREATE TEMP TABLE wanted (frame_number INTEGER PRIMARY KEY)")
        con.executemany("INSERT INTO wanted VALUES (?)", [(f,) for f in sorted(image_frames)])
        for chunk, frame_idx, frame_number, frame_time in con.execute(
                "SELECT chunk, frame_idx, frame_number, frame_time FROM frames "
                "WHERE frame_number IN (SELECT frame_number FROM temp.wanted) ORDER BY frame_number"):
            index[int(frame_number)] = (int(chunk), int(frame_idx), frame_time)
    by_chunk = {}
    for fn in sorted(index):
        by_chunk.setdefault(index[fn][0], []).append(fn)

    config = backend.load_idtrackerai_config(basedir) if HAS_IDTRACKERAI else None
    contours_db = sqlite3.connect(os.path.join(_validation(outdir), CONTOURS_FILE))
    contours_db.execute("CREATE TABLE IF NOT EXISTS contours (frame_number INTEGER PRIMARY KEY, contours BLOB)")
    have_contours = {r[0] for r in contours_db.execute("SELECT frame_number FROM contours")}

    total = sum(len(v) for v in by_chunk.values())
    done, chunks_out, encoded = 0, {}, 0
    reader = _SourceReader(basedir, extension)
    width, height = int(store["imgshape"][1]), int(store["imgshape"][0])
    try:
        for chunk in sorted(by_chunk):
            frames = by_chunk[chunk]
            clip = os.path.join(outdir, f"{chunk:06d}.mp4")
            same = (previous.get("chunks", {}).get(str(chunk)) == frames and os.path.exists(clip))
            need_contours = config is not None and any(fn not in have_contours for fn in frames)
            if same and not need_contours:
                chunks_out[str(chunk)] = frames
                done += len(frames)
                progress(0.05 + 0.9 * done / total)
                continue
            encoder = None if same else _encoder(clip + ".part.mp4", width, height, framerate)
            try:
                for fn in frames:
                    _, frame_idx, _ = index[fn]
                    image = reader.read(chunk, frame_idx)
                    if image.shape != (height, width):
                        raise ValueError(f"frame {fn} is {image.shape}, metadata.yaml says {(height, width)}")
                    if encoder is not None:
                        encoder.stdin.write(np.ascontiguousarray(image).tobytes())
                    if config is not None and fn not in have_contours:
                        try:
                            found = _simplify(backend.process_frame(image, config))
                        except Exception as error:
                            logger.warning("No contours for frame %s: %s", fn, error)
                            found = []
                        contours_db.execute("INSERT OR REPLACE INTO contours VALUES (?, ?)",
                                            (fn, zlib.compress(json.dumps(found).encode(), 6)))
                    done += 1
                    if done % 50 == 0:
                        progress(0.05 + 0.9 * done / total)
            finally:
                if encoder is not None:
                    encoder.stdin.close()
                    error = encoder.stderr.read().decode(errors="replace")
                    if encoder.wait() != 0:
                        raise RuntimeError(f"ffmpeg failed on chunk {chunk}: {error.strip()[-300:]}")
            if encoder is not None:
                os.replace(clip + ".part.mp4", clip)
                encoded += 1
            contours_db.commit()
            chunks_out[str(chunk)] = frames
    finally:
        reader.close()
        contours_db.commit()
        contours_db.close()

    # clips of chunks without scenes any more
    for name in os.listdir(outdir):
        if name.endswith(".mp4") and name[:6].isdigit() and str(int(name[:6])) not in chunks_out:
            os.remove(os.path.join(outdir, name))

    progress(0.96, "index")
    tmp = os.path.join(outdir, "index.db.tmp")
    if os.path.exists(tmp):
        os.remove(tmp)
    with sqlite3.connect(tmp) as con:
        con.execute("CREATE TABLE frames (chunk INTEGER, frame_idx INTEGER, frame_number INTEGER, frame_time REAL)")
        con.execute("CREATE INDEX frames_fn ON frames (frame_number)")
        con.executemany("INSERT INTO frames VALUES (?, ?, ?, ?)",
                        [(int(c), i, fn, index[fn][2]) for c, fns in chunks_out.items() for i, fn in enumerate(fns)])
    os.replace(tmp, os.path.join(outdir, "index.db"))
    shutil.copy2(os.path.join(basedir, "metadata.yaml"), os.path.join(outdir, "metadata.yaml"))
    for name in os.listdir(basedir):
        if name.endswith(".conf"):
            shutil.copy2(os.path.join(basedir, name), os.path.join(outdir, name))
    problems = os.path.join(_validation(basedir), "validator_problem_frames.csv")
    if os.path.exists(problems):
        shutil.copy2(problems, os.path.join(_validation(outdir), "validator_problem_frames.csv"))

    with open(os.path.join(_validation(outdir), SCENES_FILE), "w") as handle:
        json.dump(scenes, handle)
    info = {"format": FORMAT, "kind": "minimal", "experiment": flat, "created_at": time.time(),
            "scene_source": source.get("kind"), "n_scenes": len(scenes), "frames": total,
            "first_frame": min(index) if index else None, "image_margin": IMAGE_MARGIN,
            "has_contours": config is not None, "rows": counts, "encoded_clips": encoded,
            "chunks": chunks_out}
    with open(os.path.join(outdir, BUNDLE_FILE), "w") as handle:
        json.dump(info, handle)
    progress(1.0)
    return info
