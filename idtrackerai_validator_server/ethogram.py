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
"""
import os.path
import re
import logging
from functools import lru_cache

import pandas as pd
from flask import jsonify, send_from_directory

from flyhostel.utils import (
    get_basedir,
    get_chunksize,
    get_framerate,
    get_identities,
)

logger = logging.getLogger(__name__)

FLY_PATTERN = re.compile(r"^FlyHostel\d+_\d+X_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}__\d{2}$")


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
