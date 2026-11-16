"""
compat.py  —  the server without flyhostel, idtrackerai or imgstore (an annotator's install).

Those packages are installed from source (not on PyPI) and pull a whole analysis stack.
An annotator who only uses the Annotate tab on an experiment downloaded from Google Drive
(`gdrive_exchange.py`) needs none of it, so:

- HAS_FLYHOSTEL / HAS_IDTRACKERAI / HAS_IMGSTORE tell what is installed;
- the few flyhostel helpers the core of the server uses (experiment names, number of
  animals, identities, square size, validated tables) are taken from flyhostel when it is
  installed, else from the copies below (same behaviour for what they are used for here);
- app.py registers the modules that need flyhostel (Ethograms, PE, Pose, SLEAP, DEG labels,
  Overview, metadata sync, Data collector, sleep / interactions...) only when it imports,
  and lists what was left out in MISSING (GET /api/features);
- backend.py reads frames with lite_store.LiteVideoCapture (index.db + OpenCV) without
  imgstore, and gives no contours without idtrackerai (the Annotate tab then proposes
  boxes around the tracked centroids).
"""
import glob
import json
import os
import re
import sqlite3
import importlib.util

import yaml


def _installed(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAS_FLYHOSTEL = _installed("flyhostel")
HAS_IDTRACKERAI = _installed("idtrackerai")
HAS_IMGSTORE = _installed("imgstore")

# modules app.py did not register: name -> reason
MISSING = {}

ARENAS_FOLDER = "arenas"
ARENA_PREFIX = "arena"
# flyhostel's ARENA_SQUARE_WIDTH and the square of a recording without its metadata
DEFAULT_SQUARE_WIDTH = 200


if HAS_FLYHOSTEL:
    from flyhostel.utils.experiment import (  # noqa: F401
        experiment_to_relpath, get_experiment_identifier, split_arena, normalize_arena,
    )
    from flyhostel.utils import (  # noqa: F401
        get_identities, get_number_of_animals, get_square_width, get_square_height,
    )
    from flyhostel.data.human_validation.utils import check_if_validated  # noqa: F401
else:

    def normalize_arena(arena):
        """The arena as an int, None when the recording is not split into arenas (None, 0, NaN, empty)"""
        try:
            arena = int(arena)
        except (TypeError, ValueError):
            return None
        return arena or None

    def split_arena(experiment):
        """FlyHostel1_4X_<date>_arena2 -> (FlyHostel1_4X_<date>, 2); no arena -> (experiment, None)"""
        match = re.match(rf"(.*)_{ARENA_PREFIX}(\d+)$", experiment)
        if match is None:
            return experiment, None
        return match.group(1), normalize_arena(match.group(2))

    def experiment_to_relpath(experiment):
        """FlyHostel1_6X_<date>_<time> -> FlyHostel1/6X/<date>_<time> (arenas: .../arenas/<n>)"""
        experiment, arena = split_arena(experiment)
        tokens = experiment.split("_")
        relpath = os.path.join(tokens[0], tokens[1], "_".join(tokens[2:4]))
        if arena is not None:
            relpath = os.path.join(relpath, ARENAS_FOLDER, str(arena))
        return relpath

    def get_experiment_identifier(basedir):
        """.../FlyHostel1/6X/<date>_<time> -> FlyHostel1_6X_<date>_<time> (arenas: ..._arena<n>)"""
        parts = os.path.normpath(basedir).split(os.path.sep)
        if len(parts) >= 5 and parts[-2] == ARENAS_FOLDER:
            return "_".join(parts[-5:-2]) + f"_{ARENA_PREFIX}{parts[-1]}"
        return "_".join(parts[-3:])

    def _basedir(experiment):
        return os.path.join(os.environ["FLYHOSTEL_VIDEOS"], experiment_to_relpath(experiment))

    def get_number_of_animals(experiment):
        """The NX of the name; in an arena the conf's number of animals (else the database's idtrackerai_conf)"""
        if split_arena(experiment)[1] is None:
            return int(experiment.split("_")[1].rstrip("X"))
        basedir = _basedir(experiment)
        confs = sorted(glob.glob(os.path.join(basedir, "20*.conf")))
        if confs:
            with open(confs[0], encoding="utf8") as handle:
                conf = yaml.load(handle, yaml.SafeLoader)
        else:
            dbfile = os.path.join(basedir, experiment + ".db")
            with sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True) as con:
                conf = json.loads(con.execute("SELECT value FROM METADATA WHERE field = 'idtrackerai_conf'").fetchone()[0])
        return int(conf["_number_of_animals"]["value"])

    def get_identities(experiment):
        number_of_animals = get_number_of_animals(experiment)
        return [0] if number_of_animals == 1 else list(range(1, number_of_animals + 1))

    def get_square_width(experiment):
        """flyhostel picks 100 or 200 px from the recording's pixels per mm; the pose is not shown here
        without flyhostel, so the larger one"""
        return DEFAULT_SQUARE_WIDTH

    def get_square_height(*args, **kwargs):
        return get_square_width(*args, **kwargs)

    def check_if_validated(dbfile):
        with sqlite3.connect(dbfile) as con:
            tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        return "_VAL" if "IDENTITY_VAL" in tables else ""


def get_spaced_colors(n, black=False):
    """idtrackerai's get_spaced_colors_util: n BGR(A) colours of matplotlib's jet, times 256"""
    if HAS_IDTRACKERAI:
        from idtrackerai.utils.py_utils import get_spaced_colors_util
        return get_spaced_colors_util(n, black=black)
    import numpy as np
    from matplotlib import colormaps
    rgba = np.asarray([colormaps["jet"](i / n) for i in range(n)])
    colors = [tuple(float(v) for v in (row[2] * 256, row[1] * 256, row[0] * 256, row[3] * 0)) for row in rgba]
    if black:
        colors.insert(0, (0.0, 0.0, 0.0))
    return colors


def features():
    """What this install can do, for GET /api/features"""
    return {"flyhostel": HAS_FLYHOSTEL, "idtrackerai": HAS_IDTRACKERAI, "imgstore": HAS_IMGSTORE,
            "missing": dict(MISSING)}
