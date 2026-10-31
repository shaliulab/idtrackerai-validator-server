import os.path
import pickle
import logging
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


def load_rejections(experiment):
    """
    This function is also implemented in
    from flyhostel.data.interactions.sociability.behavior_integration.load_rejections
    """
    csv_file=os.path.join(
        get_basedir(experiment), "interactions", f"{experiment}_rejections.csv"
    )
    index_file=os.path.join(
        get_basedir(experiment), "interactions", f"{experiment}_index.csv"
    )
    features_file=os.path.join(
        get_basedir(experiment), "interactions", f"{experiment}_features.hdf5"
    )
    features=pd.read_hdf(features_file)

    rejections=pd.read_csv(csv_file)
    index=pd.read_csv(index_file)
    index=index.loc[index["keep"]]
    rejections=rejections.merge(index[["first_frame", "id", "nn"]].reset_index(), how="left", on=["first_frame", "id", "nn"])
    return rejections, features


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
