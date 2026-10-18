"""
compress_slp.py  —  compress SLEAP prediction files in place, losslessly.

SLEAP writes <chunk>.mp4.predictions.slp (HDF5) without compression. Repacking with
gzip (h5repack -f GZIP=4) makes them ~3.7x smaller and changes nothing SLEAP reads:
on a 45000-frame file Labels.load_file gave identical frames, instances, points,
scores, tracks, skeleton and videos, and identical analysis .h5 exports.

Each file:
    1. h5repack into .<name>.repack-tmp.slp next to it
    2. compare both files object by object (`same_contents`): every group and dataset,
       their attributes, dtype, shape, maxshape and data bytes; the first file of
       each experiment is also loaded with SLEAP from both files ($SLEAP_PYTHON,
       default the sleap env) and compared
    3. the copy gets the original's permissions and mtime (steps that compare mtimes
       do not rerun; the backup's rsync sees a different size and copies it again),
       then replaces the original (os.replace, atomic)
Any difference or error keeps the original. Skipped: files already compressed,
symbolic links, files modified in the last --min-age hours (SLEAP may still be
writing them). Every replaced file is logged in $COMPRESS_SLP_LOG
(default ~/.cache/idtrackerai-validator/compressed_slp.jsonl).

Nothing is changed without --apply (default: dry run, counts the files):
    python -m idtrackerai_validator_server.compress_slp FlyHostel1/6X/2025-10-08_16-00-00 ...
    python -m idtrackerai_validator_server.compress_slp --all --apply
Experiments on different drives run in parallel, one file at a time per drive.
"""
import os
import sys
import glob
import json
import time
import shutil
import argparse
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

from idtrackerai_validator_server.data_management import list_experiments
from idtrackerai_validator_server.experiments import to_relpath

H5REPACK = os.environ.get("H5REPACK") or shutil.which("h5repack") or os.path.expanduser("~/mambaforge/bin/h5repack")
SLEAP_PYTHON = os.environ.get("SLEAP_PYTHON", os.path.expanduser("~/mambaforge/envs/sleap/bin/python"))
LOG_FILE = os.path.expanduser(os.environ.get(
    "COMPRESS_SLP_LOG", "~/.cache/idtrackerai-validator/compressed_slp.jsonl"))
FILTER = "GZIP=4"
TMP_SUFFIX = ".repack-tmp.slp"   # SLEAP picks its reader by extension
PATTERNS = ("flyhostel/single_animal/*/*.predictions.slp",
            "arenas/*/flyhostel/single_animal/*/*.predictions.slp")

# Run with SLEAP's python: prints "identical" when both files load as the same labels
SLEAP_CHECK = r"""
import sys, numpy as np, sleap
a, b = (sleap.Labels.load_file(p) for p in sys.argv[1:3])
assert len(a) == len(b)
assert [v.filename for v in a.videos] == [v.filename for v in b.videos]
assert [s.node_names for s in a.skeletons] == [s.node_names for s in b.skeletons]
assert [s.edge_names for s in a.skeletons] == [s.edge_names for s in b.skeletons]
assert [t.name for t in a.tracks] == [t.name for t in b.tracks]
for fa, fb in zip(a.labeled_frames, b.labeled_frames):
    assert fa.frame_idx == fb.frame_idx and len(fa) == len(fb)
    for ia, ib in zip(fa, fb):
        assert np.array_equal(ia.points_and_scores_array, ib.points_and_scores_array, equal_nan=True)
        assert ia.score == ib.score
        assert (ia.track.name if ia.track else None) == (ib.track.name if ib.track else None)
print("identical")
"""


def is_compressed(path):
    """Whether any dataset of the file already has a compression filter."""
    found = []

    def visit(_, obj):
        if isinstance(obj, h5py.Dataset) and obj.compression:
            found.append(True)
            return True
    with h5py.File(path, "r") as handle:
        handle.visititems(visit)
    return bool(found)


def _contents(handle):
    """name -> (attributes, kind, dtype, shape, maxshape, data) of every object."""
    objects = {"/": (dict(handle.attrs), "group", None, None, None, None)}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            data = obj[()]
            data = data.tobytes() if isinstance(data, np.ndarray) and data.dtype.kind != "O" else \
                (list(data.ravel()) if isinstance(data, np.ndarray) else data)
            objects[name] = (dict(obj.attrs), "dataset", obj.dtype, obj.shape, obj.maxshape, data)
        else:
            objects[name] = (dict(obj.attrs), "group", None, None, None, None)
    handle.visititems(visit)
    return objects


def _same_attrs(a, b):
    return a.keys() == b.keys() and all(
        np.array_equal(np.asarray(a[key]), np.asarray(b[key])) for key in a)


def same_contents(original, copy):
    """None when both HDF5 files hold the same objects, else what differs."""
    with h5py.File(original, "r") as fa, h5py.File(copy, "r") as fb:
        a, b = _contents(fa), _contents(fb)
    if a.keys() != b.keys():
        return f"objects differ: {sorted(set(a) ^ set(b))}"
    for name in a:
        attrs_a, *rest_a = a[name]
        attrs_b, *rest_b = b[name]
        if not _same_attrs(attrs_a, attrs_b):
            return f"{name}: attributes differ"
        for what, x, y in zip(("kind", "dtype", "shape", "maxshape", "data"), rest_a, rest_b):
            if x != y:
                return f"{name}: {what} differs"
    return None


def sleap_check(original, copy):
    """None when SLEAP loads both files as the same labels, else the error."""
    if not os.path.exists(SLEAP_PYTHON):
        return f"{SLEAP_PYTHON} not found"
    proc = subprocess.run([SLEAP_PYTHON, "-c", SLEAP_CHECK, original, copy],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
    if proc.returncode == 0 and proc.stdout.strip().endswith("identical"):
        return None
    lines = proc.stderr.strip().splitlines()
    return f"SLEAP check failed: {lines[-1] if lines else proc.returncode}"


def find_files(folder, min_age):
    """(to compress, skipped reason -> count) of an experiment folder."""
    todo, skipped = [], defaultdict(int)
    now = time.time()
    for pattern in PATTERNS:
        for path in sorted(glob.glob(os.path.join(folder, pattern))):
            if os.path.islink(path):
                skipped["symbolic link"] += 1
            elif now - os.path.getmtime(path) < min_age * 3600:
                skipped["modified recently"] += 1
            else:
                try:
                    if is_compressed(path):
                        skipped["already compressed"] += 1
                        continue
                except OSError:
                    skipped["unreadable"] += 1
                    continue
                todo.append(path)
    return todo, skipped


def compress_file(path, check_with_sleap=False):
    """Repack `path` in place. Returns (status, bytes before, bytes after)."""
    tmp = os.path.join(os.path.dirname(path), "." + os.path.basename(path) + TMP_SUFFIX)
    before = os.path.getsize(path)
    st = os.stat(path)
    try:
        proc = subprocess.run([H5REPACK, "-f", FILTER, path, tmp],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3600)
        if proc.returncode != 0:
            lines = proc.stderr.strip().splitlines()
            return f"h5repack failed: {lines[-1] if lines else proc.returncode}", before, before
        after = os.path.getsize(tmp)
        if after >= before:
            return "not smaller", before, before
        problem = same_contents(path, tmp) or (sleap_check(path, tmp) if check_with_sleap else None)
        if problem:
            return f"kept, {problem}", before, before
        if (os.path.getsize(path), os.path.getmtime(path)) != (before, st.st_mtime):
            return "kept, changed while compressing", before, before
        shutil.copystat(path, tmp)
        os.replace(tmp, path)
        return "compressed", before, after
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def compress_experiment(root, experiment, apply=False, min_age=24, out=print, log=None):
    folder = os.path.join(root, experiment)
    todo, skipped = find_files(folder, min_age)
    summary = {"experiment": experiment, "compressed": 0, "before": 0, "after": 0,
               "candidates": len(todo), "skipped": dict(skipped), "kept": defaultdict(int)}
    skipped_text = ", ".join(f"{n} {why}" for why, n in skipped.items())
    if not apply:
        size = sum(os.path.getsize(p) for p in todo)
        summary["before"] = size
        out(f"{experiment}: would compress {len(todo)} files ({size / 1e9:.2f} GB)"
            + (f"; skipped {skipped_text}" if skipped_text else ""))
        return summary
    for i, path in enumerate(todo):
        status, before, after = compress_file(path, check_with_sleap=(i == 0))
        if status == "compressed":
            summary["compressed"] += 1
            summary["before"] += before
            summary["after"] += after
            if log is not None:
                log(path, before, after)
        else:
            summary["kept"][status] += 1
            out(f"{experiment}: {os.path.relpath(path, folder)}: {status}")
            if i == 0 and status.startswith("kept"):
                out(f"{experiment}: stopped, its first file (also checked with SLEAP) failed")
                break
    out(f"{experiment}: compressed {summary['compressed']}/{len(todo)} files, "
        f"{summary['before'] / 1e9:.2f} -> {summary['after'] / 1e9:.2f} GB"
        + (f"; skipped {skipped_text}" if skipped_text else ""))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compress SLEAP .predictions.slp files losslessly "
                                                 "(h5repack -f GZIP=4, verified). Dry run unless --apply.")
    parser.add_argument("experiments", nargs="*", help="FlyHostel<N>/<X>X/<date>_<time> (or underscore form)")
    parser.add_argument("--all", action="store_true", help="every experiment folder of $FLYHOSTEL_VIDEOS")
    parser.add_argument("--from-file", help="file with one experiment per line")
    parser.add_argument("--apply", action="store_true", help="really compress (default: dry run)")
    parser.add_argument("--min-age", type=float, default=24,
                        help="skip files modified in the last N hours (default 24)")
    args = parser.parse_args(argv)

    root = os.environ["FLYHOSTEL_VIDEOS"]
    experiments = list(args.experiments)
    if args.from_file:
        with open(args.from_file) as handle:
            experiments += [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    experiments = [to_relpath(e) for e in experiments]
    if args.all:
        experiments += list_experiments(root)
    if not experiments:
        parser.error("no experiments given (or --all)")
    if not os.path.exists(H5REPACK):
        parser.error(f"h5repack not found ({H5REPACK}); set $H5REPACK")
    if args.apply and not os.path.exists(SLEAP_PYTHON):
        parser.error(f"SLEAP's python not found ({SLEAP_PYTHON}); set $SLEAP_PYTHON")
    if not args.apply:
        print("Dry run: nothing is changed (pass --apply)")

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log_handle = open(LOG_FILE, "a") if args.apply else None

    def log(path, before, after):
        log_handle.write(json.dumps({"compressed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "file": path,
                                     "before": before, "after": after, "filter": FILTER}) + "\n")
        log_handle.flush()

    by_drive = defaultdict(list)
    for experiment in dict.fromkeys(experiments):
        try:
            by_drive[os.stat(os.path.join(root, experiment)).st_dev].append(experiment)
        except OSError:
            print(f"{experiment}: no such folder")

    def run_drive(queue):
        return [compress_experiment(root, e, apply=args.apply, min_age=args.min_age, log=log) for e in queue]

    try:
        with ThreadPoolExecutor(max(1, len(by_drive))) as pool:
            summaries = [s for result in pool.map(run_drive, by_drive.values()) for s in result]
    finally:
        if log_handle:
            log_handle.close()

    before = sum(s["before"] for s in summaries)
    if args.apply:
        after = sum(s["after"] for s in summaries)
        kept = sum(sum(s["kept"].values()) for s in summaries)
        print(f"\nCompressed {sum(s['compressed'] for s in summaries)} files: "
              f"{before / 1e12:.3f} -> {after / 1e12:.3f} TB; {kept} kept unchanged")
    else:
        print(f"\nWould compress {sum(s['candidates'] for s in summaries)} files, "
              f"{before / 1e12:.3f} TB now (~{before / 3.7 / 1e12:.3f} TB after)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
