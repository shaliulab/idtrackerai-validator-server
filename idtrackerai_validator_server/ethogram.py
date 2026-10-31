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

A fly with the feather but no JSON gets its figure drawn on first request (first 24 h,
as behavior-viewer intended), written next to the feather.

Arena flies show instead their activity ethogram (active / inactive per second), made by
sleep_centroid.nf (flyhostel.data.sleep.activity_ethogram) when it exists, and their sleep
bouts come from the same centroid sleep (utils.sleep_method):

    <basedir>/sleep/<NN>/<experiment>__<NN>_ethogram_centroid.feather

A fly without motionmapper feather (no movie) gets its time axis from that file: movie
time 0 = the chunk of its first frame.

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
import json
import os
import os.path
import re
import sqlite3
import logging
import tempfile
import threading
import zipfile
from contextlib import closing
from functools import lru_cache

import numpy as np
import pandas as pd
from flask import jsonify, request, send_file, send_from_directory

from flyhostel.utils import (
    get_basedir,
    get_chunksize,
    get_framerate,
    get_identities,
)
from flyhostel.utils.experiment import split_arena
from flyhostel.data.sleep import ACTIVITY_STATES, ethogram_path

# shared with FlyHostelLoader.record_all_sleep_bouts
from flyhostel.data.pose.loaders.sleep_videos import (
    SLEEP_BOUT_PADDING,
    movie_first_chunk,
    read_playlist,
    playlist_duration,
    segments_between,
    run_ffmpeg,
    cut_movie,
    bout_video_name as _bout_video_name,
)

from idtrackerai_validator_server.jobs import start_job
from idtrackerai_validator_server.experiments import to_flat
from idtrackerai_validator_server.utils import sleep_bouts as _sleep_bouts, find_sleep_bout as _find_sleep_bout

logger = logging.getLogger(__name__)

# <fly>_sleep_bout_t0-XXXX_t1-YYYY.mp4 (see flyhostel's bout_video_name), t0 / t1 in s since ZT0
SAVED_VIDEO_PATTERN = re.compile(r"^(?P<fly>.+)_sleep_bout_t0-(?P<t0>-?\d+)_t1-(?P<t1>-?\d+)\.mp4$")

FLY_PATTERN = re.compile(r"^FlyHostel\d+_\d+X_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(_arena\d+)?__\d{2}$")
# hours of the feather drawn when the figure has to be made here
FIGURE_HOURS = 24
_figure_locks = {}
_figure_locks_lock = threading.Lock()



def _fly_dir(fly):
    experiment, identity = fly.split("__")
    return os.path.join(get_basedir(experiment), "motionmapper", identity)


def _figure_path(fly):
    return os.path.join(_fly_dir(fly), f"{fly}.json")


def _feather_path(fly):
    return os.path.join(_fly_dir(fly), f"{fly}.feather")


def _activity_path(fly):
    """Activity ethogram of an arena fly (sleep_centroid.nf), None for other flies."""
    experiment = fly.split("__")[0]
    if split_arena(experiment)[1] is None:
        return None
    return ethogram_path(get_basedir(experiment), fly)


def _has_activity(fly):
    path = _activity_path(fly)
    return path is not None and os.path.exists(path)


def _ensure_figure(fly):
    """Path of the fly's figure JSON, drawn from its feather first if missing."""
    path = _figure_path(fly)
    if os.path.exists(path):
        return path
    with _figure_locks_lock:
        lock = _figure_locks.setdefault(fly, threading.Lock())
    with lock:
        if os.path.exists(path):
            return path
        feather = _feather_path(fly)
        if not os.path.exists(feather):
            raise FileNotFoundError(f"ethogram not found: {path} (and no {os.path.basename(feather)} to draw it)")
        from flyhostel.data.pose.ethogram.plot import draw_ethogram   # plotly, sklearn: only when needed
        props = _properties(fly)
        df = pd.read_feather(feather)
        # x = seconds since the start of the movie, as the movie and the pre-rendered figures
        t0 = props["first_chunk"] * props["chunksize"]
        df = df.loc[df["frame_number"] < df["frame_number"].iloc[0] + FIGURE_HOURS * 3600 * props["framerate"]]
        logger.info("Drawing the ethogram of %s from %s", fly, feather)
        fig = draw_ethogram(df, props["chunksize"], props["framerate"], x_var="seconds", t0=t0)
        tmp = path + ".tmp"
        fig.write_json(tmp)
        os.replace(tmp, path)
    return path


# Rows of the browsable ethogram (/track), bottom to top. The other inactive
# labels count as inactive; background (and anything else) is left blank.
TRACK_BEHAVIORS = ["inactive", "inactive+micromovement", "feed", "groom", "walk"]
TRACK_MERGED = {"inactive+pe": "inactive", "inactive+rejection": "inactive"}


def _track_from_feather(fly, path):
    """Winner behavior of every second of movie (most frequent prediction2, ties to
    the first of TRACK_BEHAVIORS) and the mean score of its frames."""
    props = _properties(fly)
    df = pd.read_feather(path, columns=["frame_number", "prediction2", "score"])
    labels = df["prediction2"].replace(TRACK_MERGED)
    # 0 = none of TRACK_BEHAVIORS, 1.. = TRACK_BEHAVIORS
    codes = pd.Categorical(labels, categories=TRACK_BEHAVIORS).codes.astype(np.int64) + 1
    t0 = props["first_chunk"] * props["chunksize"]
    bins = np.floor((df["frame_number"].to_numpy() - t0) / props["framerate"]).astype(np.int64)
    keep = bins >= 0
    bins, codes = bins[keep], codes[keep]
    score = np.nan_to_num(df["score"].to_numpy(dtype=float)[keep])
    n_bins = int(bins.max()) + 1 if len(bins) else 0
    n_codes = len(TRACK_BEHAVIORS) + 1
    index = bins * n_codes + codes
    counts = np.bincount(index, minlength=n_bins * n_codes).reshape(n_bins, n_codes)
    sums = np.bincount(index, weights=score, minlength=n_bins * n_codes).reshape(n_bins, n_codes)
    # behaviors before "none": a second is blank only when most of it is
    winner = np.argmax(np.roll(counts, -1, axis=1), axis=1) + 1
    winner[winner == n_codes] = 0
    rows = np.arange(n_bins)
    winner_counts = counts[rows, winner]
    winner[winner_counts == 0] = 0
    scores = np.where(winner_counts > 0, sums[rows, winner] / np.maximum(winner_counts, 1), 0)
    return winner - 1, scores


def _track_from_activity(fly, path):
    """Activity (index in ACTIVITY_STATES, -1 = none) of every second of movie, from the
    activity ethogram: each row (one ZT second) fills the movie seconds up to the next row."""
    props = _properties(fly)
    df = pd.read_feather(path, columns=["frame_number", "t", "behavior"])
    codes = pd.Categorical(df["behavior"], categories=ACTIVITY_STATES).codes.astype(np.int64)
    t0 = props["first_chunk"] * props["chunksize"]
    starts = np.floor((df["frame_number"].to_numpy() - t0) / props["framerate"]).astype(np.int64)
    # a row reaches the next one when that is the next second, otherwise only its own second
    contiguous = np.append(np.diff(df["t"].to_numpy()) == 1, False)
    ends = np.where(contiguous, np.append(starts[1:], 0), starts + 1)
    ends = np.maximum(ends, starts + 1)
    keep = starts >= 0
    starts, ends, codes = starts[keep], ends[keep], codes[keep]
    n_bins = int(ends.max()) if len(ends) else 0
    out = np.full(n_bins, -1, dtype=np.int64)
    lengths = ends - starts
    index = np.repeat(starts, lengths) + (np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths))
    out[index] = np.repeat(codes, lengths)
    return out, np.where(out >= 0, 1.0, 0.0)


def _track_from_figure(path):
    """Same as _track_from_feather, read from a pre-rendered figure (1 s markers whose
    alpha is the score) when the feather is gone."""
    with open(path) as handle:
        figure = json.load(handle)
    points = []
    for trace in figure["data"]:
        if trace.get("type") != "scattergl" or not trace.get("x"):
            continue
        behavior = TRACK_MERGED.get(trace.get("name"), trace.get("name"))
        if behavior not in TRACK_BEHAVIORS:
            continue
        colors = trace.get("marker", {}).get("color")
        if not isinstance(colors, list):
            colors = [None] * len(trace["x"])
        for x, color in zip(trace["x"], colors):
            try:
                score = float(color.rsplit(",", 1)[1].rstrip(")"))
            except (AttributeError, IndexError, ValueError):
                score = 1.0
            points.append((int(x), TRACK_BEHAVIORS.index(behavior), score))
    n_bins = max(p[0] for p in points) + 1 if points else 0
    codes = np.full(n_bins, -1, dtype=np.int64)
    scores = np.zeros(n_bins)
    for x, code, score in sorted(points, key=lambda p: -p[1]):   # ties to the first behavior
        if x >= 0:
            codes[x], scores[x] = code, score
    return codes, scores


@lru_cache(maxsize=16)
def _track(fly, source, mtime):
    """`mtime` keys the cache: a redrawn feather / figure is read again."""
    experiment = fly.split("__")[0]
    activity = source == _activity_path(fly)
    if activity:
        codes, scores = _track_from_activity(fly, source)
    elif source.endswith(".feather"):
        codes, scores = _track_from_feather(fly, source)
    else:
        codes, scores = _track_from_figure(source)
    props = _properties(fly)
    zt0 = None
    if props["first_chunk"] is not None:
        try:
            zt0 = frame_to_zt(experiment, props["first_chunk"] * props["chunksize"])
        except Exception as error:   # no STORE_INDEX row / metadata: ticks fall back to movie time
            logger.warning("No ZT for the movie start of %s: %s", fly, error)
    return {
        "behaviors": ACTIVITY_STATES if activity else TRACK_BEHAVIORS,
        # activity: from the centroid (no classifier score); behavior: from the pose
        "kind": "activity" if activity else "behavior",
        "bin": 1,
        "codes": codes.tolist(),
        "scores": np.round(scores * 100).astype(int).tolist(),   # %
        "zt0": zt0,
        "source": os.path.basename(source),
    }


def track(fly):
    """Per-second behavior of the fly (see _track_from_feather), x = movie time.
    Arena flies: their activity ethogram when it exists (_track_from_activity)."""
    for source in (_activity_path(fly), _feather_path(fly), _figure_path(fly)):
        if source is not None and os.path.exists(source):
            return _track(fly, source, os.path.getmtime(source))
    raise FileNotFoundError(f"no {os.path.basename(_feather_path(fly))} or {os.path.basename(_figure_path(fly))} for {fly}")


def _saved_videos_dir(fly):
    """Where "save on server" / FlyHostelLoader.record_all_sleep_bouts() write sleep bout videos."""
    return os.path.join(get_basedir(fly.split("__")[0]), "flyhostel", "videos", fly)


def _movie_dir(fly):
    return os.path.join(_fly_dir(fly), "movie")


@lru_cache(maxsize=256)
def _properties(fly):
    experiment = fly.split("__")[0]
    framerate = float(get_framerate(experiment))
    chunksize = int(get_chunksize(experiment))

    experiment, identity = fly.split("__")
    first_chunk = movie_first_chunk(get_basedir(experiment), experiment, identity, chunksize)
    if first_chunk is None and _has_activity(fly):
        # no movie: the time axis starts at the chunk of the activity ethogram's first frame
        first_chunk = int(pd.read_feather(_activity_path(fly), columns=["frame_number"])["frame_number"].min()) // chunksize

    return {
        "framerate": framerate,
        "chunksize": chunksize,
        "first_chunk": first_chunk,
        "has_movie": os.path.exists(os.path.join(_movie_dir(fly), "movie.m3u8")),
    }


@lru_cache(maxsize=256)
def _playlist(fly):
    return read_playlist(_movie_dir(fly))


def _movie_duration(fly):
    return playlist_duration(_playlist(fly))


def _segments_between(fly, start, end):
    return segments_between(_playlist(fly), start, end)


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
        raise ValueError(f"Cannot place frames in the ethogram of {fly}: no feather file")
    return (frame_number - props["first_chunk"] * props["chunksize"]) / props["framerate"]


def locate(fly, frame_number=None, zt=None):
    """Resolve a frame number or a ZT time into all three time conventions.
    movie_time is None when the fly has no motionmapper feather to place frames in its movie."""
    experiment = fly.split("__")[0]
    if frame_number is None:
        frame_number = zt_to_frame(experiment, zt)
    zt = frame_to_zt(experiment, frame_number)
    try:
        movie_time = _movie_time(fly, frame_number)
    except ValueError:
        movie_time = None
    in_movie = (movie_time is not None and _properties(fly)["has_movie"]
                and 0 <= movie_time <= _movie_duration(fly))
    return {
        "frame_number": int(frame_number),
        "zt": zt,
        "movie_time": movie_time,
        "in_movie": bool(in_movie),
    }


def find_sleep_bout(fly, frame_number, direction="current"):
    """See utils.find_sleep_bout. None if there is no such bout."""
    experiment, identity = fly.split("__")
    framerate = _properties(fly)["framerate"]
    bout = _find_sleep_bout(_sleep_bouts(experiment, int(identity), framerate), frame_number, direction)
    if bout is None:
        return None
    start, end = bout
    return {
        "start_frame": start,
        "end_frame": end,
        "current": start <= frame_number < end,
        "duration": (end - start) / framerate,
        "movie_time": _movie_time(fly, start),
    }


def bout_video_name(fly, bout):
    """<fly>_sleep_bout_t0-XXXX_t1-YYYY.mp4, with t0 / t1 in seconds since ZT0."""
    experiment = fly.split("__")[0]
    t0 = frame_to_zt(experiment, bout["start_frame"])
    try:
        t1 = frame_to_zt(experiment, bout["end_frame"])
    except ValueError:   # the bout ends with the recording
        t1 = t0 + bout["duration"]
    return _bout_video_name(fly, t0, t1)


def cut_bout_video(fly, bout, path, on_progress=None):
    """Write to `path` an MP4 of the fly's movie spanning `bout` (+ padding).
    on_progress(fraction) is called while the video is written.

    Raises ValueError if the movie does not cover the bout, RuntimeError if ffmpeg fails.
    """
    if not _properties(fly)["has_movie"]:
        raise ValueError(f"{fly} has no movie")
    start = _movie_time(fly, bout["start_frame"]) - SLEEP_BOUT_PADDING
    end = _movie_time(fly, bout["end_frame"]) + SLEEP_BOUT_PADDING
    try:
        cut_movie(_playlist(fly), start, end, path, on_progress=on_progress)
    except ValueError:
        raise ValueError(f"Sleep bout {bout['start_frame']}-{bout['end_frame']} is not covered by the movie of {fly}")


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
        experiment = to_flat(exp)
        flies = [f"{experiment}__{str(identity).zfill(2)}" for identity in get_identities(experiment)]
        return jsonify([
            {"fly": fly,
             "available": os.path.exists(_figure_path(fly)) or os.path.exists(_feather_path(fly)) or _has_activity(fly),
             "drawn": os.path.exists(_figure_path(fly)),
             "activity": _has_activity(fly)}
            for fly in flies
        ])

    @app.route("/api/ethogram/<fly>/figure", methods=["GET"])
    def ethogram_figure(fly):
        err = _check_fly(fly)
        if err:
            return err
        try:
            path = _ensure_figure(fly)
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except Exception as error:
            logger.exception("Could not draw the ethogram of %s", fly)
            return jsonify({"error": f"could not draw the ethogram: {error}"}), 500
        return send_from_directory(os.path.dirname(path), os.path.basename(path), mimetype="application/json")

    @app.route("/api/ethogram/<fly>/track", methods=["GET"])
    def ethogram_track(fly):
        """Per-second behavior codes (index in `behaviors`, -1 = none) and scores (%), x = movie time."""
        err = _check_fly(fly)
        if err:
            return err
        try:
            # compact: jsonify indents in debug mode, doubling the size
            return app.response_class(json.dumps(track(fly), separators=(",", ":")), mimetype="application/json")
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except Exception as error:
            logger.exception("Could not read the ethogram of %s", fly)
            return jsonify({"error": f"could not read the ethogram: {error}"}), 500

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

    @app.route("/api/ethogram/<fly>/saved_videos", methods=["GET"])
    def ethogram_saved_videos(fly):
        """Sleep bout videos saved on the server for the fly (<basedir>/flyhostel/videos/<fly>/),
        sorted by start, each placed on the ethogram's time axis."""
        err = _check_fly(fly)
        if err:
            return err
        experiment = fly.split("__")[0]
        directory = _saved_videos_dir(fly)
        if not os.path.isdir(directory):
            return jsonify({"directory": directory, "videos": []})

        props = _properties(fly)
        videos = []
        for name in sorted(os.listdir(directory)):
            match = SAVED_VIDEO_PATTERN.match(name)
            if not match or match["fly"] != fly:
                continue
            t0, t1 = float(match["t0"]), float(match["t1"])
            try:
                start_frame = zt_to_frame(experiment, t0)
            except ValueError as error:
                logger.warning("Cannot place %s: %s", name, error)
                continue
            videos.append({
                "name": name,
                "t0": t0,
                "t1": t1,
                "duration": t1 - t0,
                "start_frame": start_frame,
                "movie_time": _movie_time(fly, start_frame) if props["first_chunk"] is not None else None,
            })
        videos.sort(key=lambda video: video["t0"])
        return jsonify({"directory": directory, "videos": videos})

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
            jpeg = run_ffmpeg(
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

    @app.route("/api/ethogram/<fly>/sleep_bout/<direction>", methods=["GET"])
    def ethogram_sleep_bout_nav(fly, direction):
        """Start of the next / previous sleep bout relative to ?frame_number=.
        An ongoing bout is skipped in both directions."""
        err = _check_fly(fly)
        if err:
            return err
        if direction not in ("prev", "next"):
            return jsonify({"error": "direction must be prev or next"}), 400
        frame_number = request.args.get("frame_number", type=int)
        if frame_number is None:
            return jsonify({"error": "frame_number is required"}), 400
        bout = find_sleep_bout(fly, frame_number, direction)
        if bout is None:
            where = "after" if direction == "next" else "before"
            return jsonify({"error": f"{fly} has no more sleep bouts {where} frame {frame_number}"}), 404
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

        with tempfile.TemporaryDirectory() as workdir:
            path = os.path.join(workdir, "bout.mp4")
            try:
                cut_bout_video(fly, bout, path)
            except ValueError as error:
                return jsonify({"error": str(error)}), 404
            except RuntimeError as error:
                logger.error("Could not cut sleep bout of %s: %s", fly, error)
                return jsonify({"error": str(error)}), 500
            video = open(path, "rb")   # the open handle outlives the directory

        return send_file(
            video, mimetype="video/mp4", as_attachment=True,
            download_name=bout_video_name(fly, bout),
        )

    @app.route("/api/ethogram/<fly>/sleep_bouts/videos", methods=["POST"])
    def ethogram_sleep_bout_videos_job(fly):
        """Start a background job making one MP4 per sleep bout; poll /api/jobs/<id>.

        JSON body:
            frame_number (optional): only the bout ongoing at that frame, or the next one
            save: write to <basedir>/flyhostel/videos/<fly>/ on the server instead of
                  offering the result at /api/jobs/<id>/download (an mp4 for a single
                  bout, a zip otherwise)
        """
        err = _check_fly(fly)
        if err:
            return err
        body = request.get_json(silent=True) or {}
        save = bool(body.get("save"))
        frame_number = body.get("frame_number")
        experiment, identity = fly.split("__")
        framerate = _properties(fly)["framerate"]

        if frame_number is not None:
            bout = find_sleep_bout(fly, int(frame_number))
            if bout is None:
                return jsonify({"error": f"{fly} has no more sleep bouts after frame {frame_number}"}), 404
            bouts = [bout]
        else:
            bouts = [
                {"start_frame": start, "end_frame": end, "duration": (end - start) / framerate}
                for start, end in _sleep_bouts(experiment, int(identity), framerate)
            ]
            if not bouts:
                return jsonify({"error": f"{fly} has no sleep bouts"}), 404
        if not _properties(fly)["has_movie"]:
            return jsonify({"error": f"{fly} has no movie"}), 404

        names = [bout_video_name(fly, bout) for bout in bouts]
        directory = _saved_videos_dir(fly) if save else None
        if save:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as error:
                return jsonify({"error": f"Cannot create {directory}: {error}"}), 500

        def work(job):
            if save:
                job.directory = directory
                output_dir = directory
            else:
                job.workdir = output_dir = tempfile.mkdtemp(prefix="sleep_bouts_")

            written = []
            for i, (bout, name) in enumerate(zip(bouts, names)):
                job.update(i, status="running")
                partial = os.path.join(output_dir, f".{name}.part")
                try:
                    cut_bout_video(fly, bout, partial, on_progress=lambda p, i=i: job.update(i, progress=p))
                    os.replace(partial, os.path.join(output_dir, name))
                    written.append(name)
                    job.update(i, status="done", progress=1.0)
                except (ValueError, RuntimeError) as error:
                    if os.path.exists(partial):
                        os.unlink(partial)
                    logger.warning("Skipping sleep bout video %s: %s", name, error)
                    job.update(i, status="skipped", error=str(error))

            if not written:
                raise RuntimeError("No sleep bout video could be made")
            if save:
                return
            if len(bouts) == 1:
                job.result_path, job.download_name = os.path.join(output_dir, written[0]), written[0]
                return
            archive_path = os.path.join(output_dir, f"{fly}_sleep_bouts.zip")
            # mp4 does not compress further: just store
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
                for name in written:
                    archive.write(os.path.join(output_dir, name), name)
            job.result_path, job.download_name = archive_path, os.path.basename(archive_path)

        what = "sleep bout video" if frame_number is not None else "sleep bout videos"
        job = start_job(f"{fly}: {what}", names, work)
        return jsonify(job.to_dict()), 202
