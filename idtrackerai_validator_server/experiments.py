"""Experiment names: the flat form (index.txt, flyhostel, the client) and the
folder relative to $FLYHOSTEL_VIDEOS (slash form, SELECTED_EXPERIMENT).

    FlyHostel1_6X_2023-05-23_14-00-00         <-> FlyHostel1/6X/2023-05-23_14-00-00
    FlyHostel1_4X_2026-10-13_17-00-00_arena2  <-> FlyHostel1/4X/2026-10-13_17-00-00/arenas/2

Multi-arena recordings crop each arena into its own imgstore (with its own
database) under arenas/N; see flyhostel.utils.experiment.
"""
import os

from flyhostel.utils.experiment import experiment_to_relpath, get_experiment_identifier

# index.txt: databases of experiment folders (depth 4) and of their arenas (depth 6).
# Other folders at depth 4 are pruned, so find does not walk the experiments' contents.
INDEX_COMMAND = [
    "find", "-L", ".", "-mindepth", "4", "-maxdepth", "6", "-regextype", "posix-extended",
    "(", "-regex", r"\./[^/]+/[^/]+/[^/]+/[^/]+", "-type", "d", "-not", "-name", "arenas", "-prune", ")",
    "-o", "(", "-regex", r"\./[^/]+/[^/]+/[^/]+/(arenas/[0-9]+/)?[^/]+db", "-regex", ".*FlyHostel.*db",
    "-not", "-name", "index.db", "-print", ")",
]


def to_relpath(experiment):
    """Folder of the experiment relative to $FLYHOSTEL_VIDEOS (either form accepted)."""
    if "/" in experiment:
        return os.path.normpath(experiment.strip("/"))
    return experiment_to_relpath(experiment)


def to_flat(experiment):
    """Flat name of the experiment (either form accepted)."""
    if "/" not in experiment:
        return experiment
    return get_experiment_identifier(to_relpath(experiment))


def experiment_folder(relpath):
    """Top-level experiment folder (FlyHostel<N>/<X>X/<date>) of an experiment or arena."""
    return os.path.join(*to_relpath(relpath).split(os.sep)[:3])
