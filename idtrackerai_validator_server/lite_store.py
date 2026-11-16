"""
lite_store.py  —  frames of an imgstore recording without imgstore (compat.py: an annotator's install).

A store folder holds metadata.yaml (`__store`: chunksize, framerate, imgshape, extension),
the chunk videos <chunk:06d><extension> and index.db, imgstore's index (`frames`: chunk,
frame_idx, frame_number, frame_time). Its `chunks.chunk_path` are absolute paths of the
computer that made it, so the videos are looked for next to metadata.yaml instead.

LiteVideoCapture has the part of imgstore.interface.VideoCapture the server uses:
get_image(frame_number) -> (gray image, (frame_number, frame_time)) and `_cap` with
frame_min, frame_max, _metadata, index_db_path. One OpenCV capture is kept open and read
forward; any other frame is a seek (CAP_PROP_POS_FRAMES).
"""
import os
import sqlite3
import threading

import cv2
import yaml


class _Store:

    def __init__(self, store_path):
        self.basedir = os.path.dirname(os.path.abspath(store_path))
        with open(store_path) as handle:
            self._metadata = yaml.safe_load(handle)["__store"]
        self.index_db_path = os.path.join(self.basedir, "index.db")
        if not os.path.exists(self.index_db_path):
            raise FileNotFoundError(f"{self.index_db_path} not found: without imgstore the index cannot be made here")
        with self._connect() as con:
            lo, hi = con.execute("SELECT MIN(frame_number), MAX(frame_number) FROM frames").fetchone()
        if lo is None:
            raise ValueError(f"{self.index_db_path} lists no frames")
        self.frame_min, self.frame_max = float(lo), float(hi)
        self.extension = self._metadata.get("extension", ".mp4")

    def _connect(self):
        return sqlite3.connect(f"file:{self.index_db_path}?mode=ro", uri=True, timeout=10)

    def locate(self, frame_number):
        """(chunk, frame_idx, frame_time) of a frame number, or of the nearest frame listed before it"""
        with self._connect() as con:
            row = con.execute("SELECT chunk, frame_idx, frame_time, frame_number FROM frames WHERE frame_number <= ? "
                              "ORDER BY frame_number DESC LIMIT 1", (frame_number,)).fetchone()
            if row is None:
                row = con.execute("SELECT chunk, frame_idx, frame_time, frame_number FROM frames "
                                  "ORDER BY frame_number LIMIT 1").fetchone()
        return row

    def video(self, chunk):
        return os.path.join(self.basedir, f"{int(chunk):06d}{self.extension}")


class LiteVideoCapture:

    def __init__(self, store_path, chunk=None):
        self._cap = _Store(store_path)
        self._lock = threading.Lock()
        self._video = None       # (chunk, cv2.VideoCapture, index of the next frame it reads)

    def _open(self, chunk):
        if self._video and self._video[0] == chunk:
            return self._video
        if self._video:
            self._video[1].release()
        path = self._cap.video(chunk)
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open {path}")
        self._video = [chunk, capture, 0]
        return self._video

    def get_image(self, frame_number):
        chunk, frame_idx, frame_time, frame_number = self._cap.locate(frame_number)
        with self._lock:
            video = self._open(chunk)
            if video[2] != frame_idx:
                video[1].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = video[1].read()
            video[2] = frame_idx + 1 if ok else -1
        if not ok:
            raise ValueError(f"cannot read frame {frame_idx} of {self._cap.video(chunk)}")
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame, (int(frame_number), frame_time)

    def release(self):
        if self._video:
            self._video[1].release()
            self._video = None
