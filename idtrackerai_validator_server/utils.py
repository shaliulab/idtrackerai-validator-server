import os.path
import pickle
import logging
import pandas as pd
from flyhostel.utils import (
    get_basedir,
)
from flyhostel.data.pose.main import FlyHostelLoader

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


def load_sleep_data(experiment: str, identity: int) -> list:
    """Return sorted list of frame_numbers where `identity` is asleep.

    Result is cached; returns [] when the feather file is missing or on any error.
    `experiment` must be the flat form (underscores, not slashes).
    """
    key = (experiment, int(identity))
    if key in SLEEP_CACHE:
        return SLEEP_CACHE[key]

    sleep_frames: list = []
    try:
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


def sleep_bouts(experiment: str, identity: int, framerate: float, max_gap: float = 2) -> list:
    """[(first_frame, end_frame)] of every sleep bout, sorted. end_frame is exclusive.

    Asleep rows further apart than `max_gap` seconds belong to different bouts;
    each row stands for (about) one second, so a bout ends 1 s after its last row.
    """
    bouts = []
    for frame_number in load_sleep_data(experiment, identity):
        if bouts and frame_number - bouts[-1][1] <= max_gap * framerate:
            bouts[-1][1] = frame_number
        else:
            bouts.append([frame_number, frame_number])
    return [(start, int(end + framerate)) for start, end in bouts]


def find_sleep_bout(bouts: list, frame_number: int, direction: str = "current"):
    """Pick a bout relative to `frame_number`; None if there is none.

    current: the bout ongoing at frame_number, or else the next one
    next:    the first bout starting after frame_number (skips an ongoing bout)
    prev:    the last bout starting before the ongoing bout, or before
             frame_number when the fly is awake
    """
    if direction == "current":
        return next(((s, e) for s, e in bouts if e > frame_number), None)
    if direction == "next":
        return next(((s, e) for s, e in bouts if s > frame_number), None)
    if direction == "prev":
        ongoing = next(((s, e) for s, e in bouts if s <= frame_number < e), None)
        reference = ongoing[0] if ongoing else frame_number
        return next(((s, e) for s, e in reversed(bouts) if s < reference), None)
    raise ValueError(f"direction must be current, next or prev, not {direction}")
