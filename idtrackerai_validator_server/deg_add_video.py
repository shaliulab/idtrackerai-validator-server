"""
deg_add_video.py  —  add a single-animal chunk video to a deepethogram project.

Run by deg_labels.ensure_label_file with the Python of an environment where
deepethogram is installed ($DEG_PYTHON), not the server's: it must not import
anything from the server.

    python deg_add_video.py PROJECT VIDEO KEY

Makes <PROJECT>/DATA/<KEY>/ as deepethogram's projects.add_video_to_project does
(copy of the video, record.yaml, stats.yaml from zscore's get_video_statistics), but
keeps the video's name (<chunk>.mp4) inside a folder named after KEY, as the existing
folders do. The project's overall normalization (project_config.yaml) is left alone:
add_video_to_project would rewrite the whole config with the paths of this machine.

Then makes <chunk>_labels.csv if missing: one row per frame of the video, one column
per class of the project, all -1 (not labelled), and lists it in record.yaml.
Today's index_<date>.pkl (the shaliulab fork's cache of the records) is removed so the
GUI sees the new folder. Prints @@PROGRESS <fraction> lines while it works (the video
statistics are most of the time, ~30 s), then {"folder", "labels", "created", "frames"}
as JSON.
"""
import datetime
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from deepethogram import projects, utils, zscore
from deepethogram.file_io import VideoReader
from deepethogram.zscore import get_video_statistics

MARKER = "@@PROGRESS "


def progress(fraction):
    print(f"{MARKER}{fraction:.3f}", flush=True)


def _reporting(iterable, *args, **kwargs):
    """Stands in for the tqdm of get_video_statistics: its frames are 5–95 % of the work."""
    items = list(iterable)
    for i, item in enumerate(items):
        if i % 20 == 0:
            progress(0.05 + 0.9 * i / max(1, len(items)))
        yield item


zscore.tqdm = _reporting


def main(project_dir, video, key):
    cfg = utils.load_yaml(os.path.join(project_dir, "project_config.yaml"))
    class_names = list(cfg["project"]["class_names"])
    datadir = os.path.join(project_dir, "DATA")
    folder = os.path.join(datadir, key)
    basename = os.path.basename(video)
    chunk = os.path.splitext(basename)[0]
    created = False

    if not os.path.isdir(folder):
        partial = os.path.join(datadir, f".{key}.partial")
        shutil.rmtree(partial, ignore_errors=True)
        os.makedirs(partial)
        shutil.copy(video, os.path.join(partial, basename))
        progress(0.05)
        utils.save_dict_to_yaml(
            {"normalization": get_video_statistics(os.path.join(partial, basename), 10)},
            os.path.join(partial, "stats.yaml"),
        )
        os.rename(partial, folder)
        created = True

    labels = os.path.join(folder, f"{chunk}_labels.csv")
    with VideoReader(os.path.join(folder, basename)) as reader:
        frames = len(reader)
    if not os.path.exists(labels):
        df = pd.DataFrame(np.full((frames, len(class_names)), -1, dtype=np.int64), columns=class_names)
        df.to_csv(labels + ".tmp")
        os.replace(labels + ".tmp", labels)
        created = True

    if created:
        utils.save_dict_to_yaml(projects.parse_subdir(folder), os.path.join(folder, "record.yaml"))
        cache = os.path.join(datadir, f"index_{datetime.date.today()}.pkl")
        if os.path.exists(cache):
            os.remove(cache)

    progress(1.0)
    print(json.dumps({"folder": folder, "labels": labels, "created": created, "frames": frames}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
