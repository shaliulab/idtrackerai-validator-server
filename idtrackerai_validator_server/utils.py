import os.path
import pickle
import logging
import numpy as np
import pandas as pd
from flyhostel.utils import (
    get_basedir,
)
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.sleep import sleep_path
from flyhostel.utils.experiment import split_arena
# shared with FlyHostelLoader.record_all_sleep_bouts
from flyhostel.data.pose.loaders.sleep_videos import (  # noqa: F401  (find_sleep_bout is re-exported)
    SLEEP_BOUT_MAX_GAP,
    group_sleep_bouts,
    find_sleep_bout,
)

logger = logging.getLogger(__name__)


# Interactions of every fly: interactions path -> (mtime, {identity: sorted first frames})
INTERACTIONS_CACHE: dict = {}


def interactions_path(experiment: str) -> str:
    """Interactions found by the interactions pipeline (detect_neighbors / describe_interactions of vsc-scripts):
    <basedir>/interactions/<experiment>_index.csv, one row per interaction and fly (id) with its partner (nn)"""
    return os.path.join(get_basedir(experiment), "interactions", f"{experiment}_index.csv")


def load_interactions(experiment: str) -> dict:
    """{identity: sorted first frames of the interactions of the fly, as id or as nn}, of the interactions kept by the
    pipeline (keep). {} when the experiment has no interactions index. Cached per file mtime."""
    path = interactions_path(experiment)
    if not os.path.exists(path):
        return {}
    mtime = os.path.getmtime(path)
    cached = INTERACTIONS_CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    index = pd.read_csv(path)
    if "keep" in index.columns:
        index = index.loc[index["keep"].astype(bool)]
    frames = {}
    for column in ("id", "nn"):
        # fly ids end in |<identity>
        identities = index[column].astype(str).str.rsplit("|", n=1).str[-1].astype(int)
        for identity, first_frames in index["first_frame"].groupby(identities):
            frames.setdefault(int(identity), []).append(first_frames.to_numpy())
    frames = {identity: np.unique(np.concatenate(parts)).astype(int) for identity, parts in frames.items()}
    INTERACTIONS_CACHE[path] = (mtime, frames)
    return frames


def find_interaction(first_frames, frame_number: int, direction: str):
    """First frame of the next (after frame_number) or previous (before it) interaction, None when there is none"""
    if first_frames is None or len(first_frames) == 0:
        return None
    if direction == "next":
        i = np.searchsorted(first_frames, frame_number, side="right")
        return int(first_frames[i]) if i < len(first_frames) else None
    i = np.searchsorted(first_frames, frame_number, side="left") - 1
    return int(first_frames[i]) if i >= 0 else None


# Sleep frame cache: (experiment_flat, identity_int) -> sorted list[int] of asleep frame_numbers
SLEEP_CACHE: dict = {}


def sleep_method(experiment: str, identity: int) -> str:
    """Sleep definition of a fly (flyhostel.data.sleep): arena flies use the centroid method (sleep_centroid.nf)
    when it has been computed, as their pose based behavior is unreliable on the walls; other flies the behavior."""
    if split_arena(experiment)[1] is not None:
        animal = f"{experiment}__{str(int(identity)).zfill(2)}"
        if os.path.exists(sleep_path(get_basedir(experiment), animal, method="centroid")):
            return "centroid"
    return "behavior"


def load_sleep_data(experiment: str, identity: int) -> list:
    """Return sorted list of frame_numbers where `identity` is asleep.

    Result is cached; returns [] when the feather file is missing or on any error.
    `experiment` must be the flat form (underscores, not slashes). Sleep as defined by `sleep_method`.
    """
    key = (experiment, int(identity))
    if key in SLEEP_CACHE:
        return SLEEP_CACHE[key]

    sleep_frames: list = []
    try:
        if sleep_method(experiment, identity) == "centroid":
            # read directly: FlyHostelLoader needs a validated experiment, which arenas with a splitter are not
            animal = f"{experiment}__{str(int(identity)).zfill(2)}"
            df = pd.read_feather(sleep_path(get_basedir(experiment), animal, method="centroid"), columns=["frame_number", "inactive_rule"])
            df = df.rename(columns={"inactive_rule": "asleep"})
        else:
            loader = FlyHostelLoader(experiment, int(identity))
            loader.load_sleep_data(bin_size=None, errors="warning")
            df = loader.sleep
        if df is not None and not df.empty and 'asleep' in df.columns and 'frame_number' in df.columns:
            asleep_fn = df.loc[df['asleep'] == True, 'frame_number'].dropna().astype(int)
            sleep_frames = sorted(asleep_fn.tolist())
    except Exception as exc:
        logger.warning("Sleep data unavailable for %s id=%s: %s", experiment, identity, exc)

    SLEEP_CACHE[key] = sleep_frames
    return sleep_frames


def sleep_bouts(experiment: str, identity: int, framerate: float, max_gap: float = SLEEP_BOUT_MAX_GAP) -> list:
    """[(first_frame, end_frame)] of every sleep bout (end exclusive); see flyhostel's group_sleep_bouts."""
    return group_sleep_bouts(load_sleep_data(experiment, identity), framerate, max_gap=max_gap)
