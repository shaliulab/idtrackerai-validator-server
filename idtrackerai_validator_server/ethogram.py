"""
ethogram.py  —  per-fly ethogram endpoints (merged from the behavior-viewer project).

Each fly's ethogram is a pre-rendered plotly figure produced by
flyhostel.data.pose.ethogram.draw_ethogram and stored, together with an HLS movie,
under the experiment tree:

    <basedir>/motionmapper/<NN>/<experiment>__<NN>.json
    <basedir>/motionmapper/<NN>/<experiment>__<NN>.feather
    <basedir>/motionmapper/<NN>/movie/movie.m3u8

The x axis of the figure is seconds since the start of the movie, which itself
starts at the first chunk present in the feather file.

Time conventions (same as flyhostel's sleep tables and the ethogram's ZT ticks):
    frame_number   frames since the beginning of the recording (chunk 0)
    zt             seconds since ZT0 of the first day of recording
                   = frame_time / 1000 + offset,
                   frame_time = ms since the o'clock hour the recording started in
                   offset = that hour (s since midnight) - reference_hour * 3600
                   (same as flyhostel.utils.load_meta_info's t_after_ref)
    movie_time     seconds since the start of the fly's movie
                   = (frame_number - first_chunk * chunksize) / framerate
"""
import io
import os
import os.path
import re
import sqlite3
import logging
import tempfile
import subprocess
from contextlib import closing
from functools import lru_cache

import pandas as pd
from flask import jsonify, request, send_file, send_from_directory

from flyhostel.utils import (
    get_basedir,
    get_chunksize,
    get_framerate,
    get_identities,
)

from idtrackerai_validator_server.utils import load_sleep_data

logger = logging.getLogger(__name__)

FLY_PATTERN = re.compile(r"^FlyHostel\d+_\d+X_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}__\d{2}$")

# Seconds added before and after a sleep bout in the exported video.
SLEEP_BOUT_PADDING = 5
# Two asleep rows further apart than this many seconds belong to different bouts.
SLEEP_BOUT_MAX_GAP = 2


def _fly_dir(fly):
    experiment, identity = fly.split("__")
    return os.path.join(get_basedir(experiment), "motionmapper", identity)


def _figure_path(fly):
    return os.path.join(_fly_dir(fly), f"{fly}.json")


def _movie_dir(fly):
    return os.path.join(_fly_dir(fly), "movie")


@lru_cache(maxsize=256)
def _properties(fly):
    experiment = fly.split("__")[0]
    framerate = float(get_framerate(experiment))
    chunksize = int(get_chunksize(experiment))

    feather_path = os.path.join(_fly_dir(fly), f"{fly}.feather")
    first_chunk = None
    if os.path.exists(feather_path):
        frame_numbers = pd.read_feather(feather_path, columns=["frame_number"])["frame_number"]
        if len(frame_numbers):
            first_chunk = int(frame_numbers.iloc[0]) // chunksize

    return {
        "framerate": framerate,
        "chunksize": chunksize,
        "first_chunk": first_chunk,
        "has_movie": os.path.exists(os.path.join(_movie_dir(fly), "movie.m3u8")),
    }


@lru_cache(maxsize=256)
def _playlist(fly):
    """[(segment path, start in movie time, duration)] of the HLS movie.

    The playlist has #EXT-X-DISCONTINUITY tags (timestamps restart in every
    segment), so ffmpeg cannot seek in the .m3u8: we address segments ourselves."""
    segments = []
    start = 0.0
    duration = None
    with open(os.path.join(_movie_dir(fly), "movie.m3u8")) as playlist:
        for line in playlist:
            line = line.strip()
            if line.startswith("#EXTINF:"):
                duration = float(line[len("#EXTINF:"):].split(",")[0])
            elif line and not line.startswith("#") and duration is not None:
                segments.append((os.path.join(_movie_dir(fly), line), start, duration))
                start += duration
                duration = None
    return segments


def _movie_duration(fly):
    segments = _playlist(fly)
    return segments[-1][1] + segments[-1][2] if segments else 0.0


def _segments_between(fly, start, end):
    """Segments overlapping [start, end] of movie time."""
    return [seg for seg in _playlist(fly) if seg[1] + seg[2] > start and seg[1] <= end]


def _connect(experiment):
    dbfile = os.path.join(get_basedir(experiment), f"{experiment}.db")
    return closing(sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True))


@lru_cache(maxsize=64)
def _zt_offset(experiment):
    """Seconds to add to frame_time / 1000 to get seconds since ZT0."""
    with _connect(experiment) as conn:
        date_time = conn.execute("SELECT value FROM METADATA WHERE field = 'date_time'").fetchone()[0]
        ethoscope_metadata = conn.execute("SELECT value FROM METADATA WHERE field = 'ethoscope_metadata'").fetchone()[0]

    reference_hour = pd.read_csv(io.StringIO(ethoscope_metadata), index_col=0)["reference_hour"].unique()
    assert len(reference_hour) == 1, f"Multiple reference hours in {experiment}: {reference_hour}"
    start_time = int(float(date_time))
    start_hour = (start_time - start_time % 3600) % (24 * 3600)
    return start_hour - float(reference_hour[0]) * 3600


def frame_to_zt(experiment, frame_number):
    with _connect(experiment) as conn:
        row = conn.execute("SELECT frame_time FROM STORE_INDEX WHERE frame_number = ?", (int(frame_number),)).fetchone()
    if row is None:
        raise ValueError(f"Frame {frame_number} not found in {experiment}")
    return row[0] / 1000 + _zt_offset(experiment)


def zt_to_frame(experiment, zt):
    frame_time = (float(zt) - _zt_offset(experiment)) * 1000
    with _connect(experiment) as conn:
        row = conn.execute(
            "SELECT frame_number FROM STORE_INDEX WHERE frame_time >= ? ORDER BY frame_time LIMIT 1",
            (frame_time,)
        ).fetchone()
    if row is None:
        raise ValueError(f"ZT {zt} s is after the end of {experiment}")
    return int(row[0])


def _movie_time(fly, frame_number):
    props = _properties(fly)
    if props["first_chunk"] is None:
        raise ValueError(f"Cannot place frames in the movie of {fly}: no feather file")
    return (frame_number - props["first_chunk"] * props["chunksize"]) / props["framerate"]


def locate(fly, frame_number=None, zt=None):
    """Resolve a frame number or a ZT time into all three time conventions."""
    experiment = fly.split("__")[0]
    if frame_number is None:
        frame_number = zt_to_frame(experiment, zt)
    zt = frame_to_zt(experiment, frame_number)
    movie_time = _movie_time(fly, frame_number)
    in_movie = _properties(fly)["has_movie"] and 0 <= movie_time <= _movie_duration(fly)
    return {
        "frame_number": int(frame_number),
        "zt": zt,
        "movie_time": movie_time,
        "in_movie": bool(in_movie),
    }


def sleep_bouts(fly):
    """[(first_frame, last_frame + 1 s)] of every sleep bout of the fly, sorted."""
    experiment, identity = fly.split("__")
    framerate = _properties(fly)["framerate"]
    frames = load_sleep_data(experiment, int(identity))
    bouts = []
    for frame_number in frames:
        if bouts and frame_number - bouts[-1][1] <= SLEEP_BOUT_MAX_GAP * framerate:
            bouts[-1][1] = frame_number
        else:
            bouts.append([frame_number, frame_number])
    # each asleep row stands for (about) one second
    return [(start, int(end + framerate)) for start, end in bouts]


def find_sleep_bout(fly, frame_number):
    """The bout the fly is in at `frame_number`, or the next one. None if there is none."""
    for start, end in sleep_bouts(fly):
        if end > frame_number:
            return {
                "start_frame": start,
                "end_frame": end,
                "current": start <= frame_number,
                "duration": (end - start) / _properties(fly)["framerate"],
            }
    return None


def _ffmpeg(*args):
    process = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode(errors="replace").strip())
    return process.stdout


def register_ethogram(app, get_selected_experiment):
    """Attach the ethogram routes to an existing Flask `app`.
    `get_selected_experiment` is a 0-arg callable returning the current experiment."""

    def _check_fly(fly):
        if not FLY_PATTERN.match(fly):
            return jsonify({"error": f"invalid fly id {fly}"}), 400
        return None

    @app.route("/api/ethogram/flies", methods=["GET"])
    def ethogram_flies():
        """Flies of the loaded experiment, flagged by whether an ethogram exists."""
        exp = get_selected_experiment()
        if not exp:
            return jsonify([])
        experiment = exp.replace("/", "_")
        flies = [f"{experiment}__{str(identity).zfill(2)}" for identity in get_identities(experiment)]
        return jsonify([
            {"fly": fly, "available": os.path.exists(_figure_path(fly))}
            for fly in flies
        ])

    @app.route("/api/ethogram/<fly>/figure", methods=["GET"])
    def ethogram_figure(fly):
        err = _check_fly(fly)
        if err:
            return err
        path = _figure_path(fly)
        if not os.path.exists(path):
            return jsonify({"error": f"ethogram not found: {path}"}), 404
        return send_from_directory(os.path.dirname(path), os.path.basename(path), mimetype="application/json")

    @app.route("/api/ethogram/<fly>/properties", methods=["GET"])
    def ethogram_properties(fly):
        err = _check_fly(fly)
        if err:
            return err
        try:
            return jsonify(_properties(fly))
        except Exception as error:
            logger.error("Could not compute ethogram properties for %s: %s", fly, error)
            return jsonify({"error": str(error)}), 500

    @app.route("/api/ethogram/<fly>/movie/<path:filename>", methods=["GET"])
    def ethogram_movie(fly, filename):
        err = _check_fly(fly)
        if err:
            return err
        mimetype = None
        if filename.endswith(".m3u8"):
            mimetype = "application/x-mpegURL"
        elif filename.endswith(".ts"):
            mimetype = "video/MP2T"
        return send_from_directory(_movie_dir(fly), filename, mimetype=mimetype)

    def _time_args():
        """Exactly one of ?frame_number= (frames since chunk 0) or ?zt= (s since ZT0)."""
        frame_number = request.args.get("frame_number", type=int)
        zt = request.args.get("zt", type=float)
        if (frame_number is None) == (zt is None):
            raise ValueError("Pass exactly one of ?frame_number= or ?zt=")
        return frame_number, zt

    @app.route("/api/ethogram/<fly>/locate", methods=["GET"])
    def ethogram_locate(fly):
        """Convert ?frame_number= or ?zt= into frame_number, zt and movie_time."""
        err = _check_fly(fly)
        if err:
            return err
        try:
            return jsonify(locate(fly, *_time_args()))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @app.route("/api/ethogram/<fly>/movie_frame", methods=["GET"])
    def ethogram_movie_frame(fly):
        """JPEG of the fly's movie at ?frame_number= or ?zt=.
        e.g. /api/ethogram/FlyHostel1_6X_2025-10-02_16-00-00__01/movie_frame?zt=40000"""
        err = _check_fly(fly)
        if err:
            return err
        try:
            position = locate(fly, *_time_args())
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        if not position["in_movie"]:
            return jsonify({"error": f"Frame {position['frame_number']} is not covered by the movie of {fly}", **position}), 404

        segment, segment_start, _ = _segments_between(fly, position["movie_time"], position["movie_time"])[0]
        try:
            # Seek after -i (decode and discard): input seeking silently returns
            # nothing on some movies' segments, and segments are only ~10 s long.
            jpeg = _ffmpeg(
                "-i", segment,
                "-ss", f"{position['movie_time'] - segment_start:.3f}",
                "-frames:v", "1", "-f", "image2", "-c:v", "mjpeg", "-q:v", "2", "pipe:1",
            )
            if not jpeg:
                raise RuntimeError(f"ffmpeg extracted no frame from {segment} at {position['movie_time'] - segment_start:.3f} s")
        except RuntimeError as error:
            logger.error("Could not extract frame of %s: %s", fly, error)
            return jsonify({"error": str(error)}), 500

        response = send_file(io.BytesIO(jpeg), mimetype="image/jpeg")
        response.headers["X-Frame-Number"] = str(position["frame_number"])
        response.headers["X-ZT"] = f"{position['zt']:.3f}"
        return response

    @app.route("/api/ethogram/<fly>/sleep_bout", methods=["GET"])
    def ethogram_sleep_bout(fly):
        """The sleep bout ongoing at ?frame_number=, or else the next one."""
        err = _check_fly(fly)
        if err:
            return err
        frame_number = request.args.get("frame_number", type=int)
        if frame_number is None:
            return jsonify({"error": "frame_number is required"}), 400
        bout = find_sleep_bout(fly, frame_number)
        if bout is None:
            return jsonify({"error": f"{fly} has no more sleep bouts after frame {frame_number}"}), 404
        return jsonify(bout)

    @app.route("/api/ethogram/<fly>/sleep_bout/video", methods=["GET"])
    def ethogram_sleep_bout_video(fly):
        """MP4 cut from the fly's movie spanning the bout returned by /sleep_bout."""
        err = _check_fly(fly)
        if err:
            return err
        frame_number = request.args.get("frame_number", type=int)
        if frame_number is None:
            return jsonify({"error": "frame_number is required"}), 400
        bout = find_sleep_bout(fly, frame_number)
        if bout is None:
            return jsonify({"error": f"{fly} has no more sleep bouts after frame {frame_number}"}), 404
        if not _properties(fly)["has_movie"]:
            return jsonify({"error": f"{fly} has no movie"}), 404

        duration = _movie_duration(fly)
        start = _movie_time(fly, bout["start_frame"]) - SLEEP_BOUT_PADDING
        end = _movie_time(fly, bout["end_frame"]) + SLEEP_BOUT_PADDING
        if end < 0 or start > duration:
            return jsonify({"error": f"Sleep bout {bout['start_frame']}-{bout['end_frame']} is not covered by the movie of {fly}"}), 404
        start, end = max(0.0, start), min(duration, end)

        # Concatenate whole segments spanning the bout (+ padding). Stream copy is
        # fast even for bouts of hours, but can only cut at keyframes: keeping whole
        # segments guarantees the bout is included, at the cost of <= 1 segment
        # (10 s) of extra context on each side.
        segments = _segments_between(fly, start, end)
        concat_list = ["ffconcat version 1.0"] + [f"file '{segment}'" for segment, _, _ in segments]

        with tempfile.TemporaryDirectory() as workdir:
            list_path = os.path.join(workdir, "segments.ffconcat")
            path = os.path.join(workdir, "bout.mp4")
            with open(list_path, "w") as handle:
                handle.write("\n".join(concat_list) + "\n")
            try:
                _ffmpeg(
                    "-f", "concat", "-safe", "0", "-i", list_path,
                    "-c", "copy", "-movflags", "+faststart", path,
                )
                video = open(path, "rb")   # the open handle outlives the directory
            except RuntimeError as error:
                logger.error("Could not cut sleep bout of %s: %s", fly, error)
                return jsonify({"error": str(error)}), 500

        return send_file(
            video, mimetype="video/mp4", as_attachment=True,
            download_name=f"{fly}_sleep_{bout['start_frame']}-{bout['end_frame']}.mp4",
        )
