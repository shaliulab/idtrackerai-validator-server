"""Experiment names: the flat form (index.txt, flyhostel, the client) and the
folder relative to $FLYHOSTEL_VIDEOS (slash form, SELECTED_EXPERIMENT).

    FlyHostel1_6X_2023-05-23_14-00-00         <-> FlyHostel1/6X/2023-05-23_14-00-00
    FlyHostel1_4X_2026-10-13_17-00-00_arena2  <-> FlyHostel1/4X/2026-10-13_17-00-00/arenas/2

Multi-arena recordings crop each arena into its own imgstore (with its own
database) under arenas/N; see flyhostel.utils.experiment.
"""
import logging
import glob
import os
import posixpath
import re
import sqlite3
import threading

from idtrackerai_validator_server.compat import experiment_to_relpath, get_experiment_identifier
from idtrackerai_validator_server.oscompat import to_posix

# index.txt: databases of experiment folders (depth 4) and of their arenas (depth 6).
# Other folders at depth 4 are pruned, so find does not walk the experiments' contents.
INDEX_COMMAND = [
    "find", "-L", ".", "-mindepth", "4", "-maxdepth", "6", "-regextype", "posix-extended",
    "(", "-regex", r"\./[^/]+/[^/]+/[^/]+/[^/]+", "-type", "d", "-not", "-name", "arenas", "-prune", ")",
    "-o", "(", "-regex", r"\./[^/]+/[^/]+/[^/]+/(arenas/[0-9]+/)?[^/]+db", "-regex", ".*FlyHostel.*db",
    # -size +0c: an empty file is no database (sqlite makes one when a missing path is opened for writing)
    "-not", "-name", "index.db", "-size", "+0c", "-print", ")",
]


_INDEX_DEPTH4 = re.compile(r"\./[^/]+/[^/]+/[^/]+/(arenas/[0-9]+/)?[^/]+db")
_INDEX_FLYHOSTEL = re.compile(r".*FlyHostel.*db")


def index_lines(root):
    """What INDEX_COMMAND lists, in Python (Windows has no GNU find): "./<relpath>" of the
    non-empty databases of the experiment folders and of their arenas, links followed."""
    lines = []
    for pattern in (("*", "*", "*", "*db"), ("*", "*", "*", "arenas", "*", "*db")):
        for path in glob.glob(os.path.join(root, *pattern)):
            line = "./" + os.path.relpath(path, root).replace(os.sep, "/")
            if (os.path.basename(path) == "index.db" or not _INDEX_DEPTH4.fullmatch(line)
                    or not _INDEX_FLYHOSTEL.fullmatch(line)):
                continue
            try:
                if os.path.isfile(path) and os.path.getsize(path) > 0:
                    lines.append(line)
            except OSError:
                pass
    return sorted(lines)


def to_relpath(experiment):
    """Folder of the experiment relative to $FLYHOSTEL_VIDEOS (either form accepted), with "/"
    separators also on Windows."""
    experiment = to_posix(experiment)
    if "/" in experiment:
        return posixpath.normpath(experiment.strip("/"))
    return to_posix(experiment_to_relpath(experiment))


def to_flat(experiment):
    """Flat name of the experiment (either form accepted)."""
    experiment = to_posix(experiment)
    if "/" not in experiment:
        return experiment
    return get_experiment_identifier(to_relpath(experiment))


def experiment_folder(relpath):
    """Top-level experiment folder (FlyHostel<N>/<X>X/<date>) of an experiment or arena."""
    return "/".join(to_relpath(relpath).split("/")[:3])


# A validated experiment: its database holds these tables (save-human-annotations writes them).
# The Data collector, the Overview and the Annotate tab's list of experiments to validate use it.
VALIDATED_TABLES = ("ROI_0_VAL", "IDENTITY_VAL", "CONCATENATION_VAL")
_validated_cache = {}  # path -> ((mtime, size), validated)
_validated_lock = threading.Lock()
logger = logging.getLogger(__name__)


def is_validated(path, stat=None):
    """Whether a database holds every VALIDATED_TABLES table, opened read-only; cached per file
    until its mtime or size changes; None when unreadable."""
    stat = stat or os.stat(path)
    key = (stat.st_mtime, stat.st_size)
    with _validated_lock:
        cached = _validated_cache.get(path)
    if cached and cached[0] == key:
        return cached[1]
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            names = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        finally:
            con.close()
    except sqlite3.Error as error:
        logger.warning("Cannot read the tables of %s: %s", path, error)
        return None
    validated = all(table in names for table in VALIDATED_TABLES)
    with _validated_lock:
        _validated_cache[path] = (key, validated)
    return validated
