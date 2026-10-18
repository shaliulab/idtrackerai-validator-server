"""
trim_raw.py  —  delete the raw chunk files of an experiment that no analysis uses and
that Dropbox has an identical copy of.

Raw chunk files (NNNNNN.mp4 / .npz / .png / .avi in the experiment folder) are
"outside tracking" when their chunk is before the first or after the last chunk with
rows in ROI_0 of the experiment's database. Those are deleted only when Dropbox
($DROPBOX_REMOTE/<experiment>/, see dropbox.py) has the same file with the same size
and the same content hash, the hash being computed from the local file right before:

    1. list the experiment folder on Dropbox (sizes and Dropbox content hashes)
    2. hash every candidate file locally, compare with its Dropbox entry
    3. list the folder on Dropbox again, and delete a file only if its entry is
       unchanged and the local file has the same size and mtime as when it was hashed

Any difference, a file missing on Dropbox or an rclone error keeps the file. Every
deleted file is written to <experiment>/deleted_raw_chunks.jsonl (path, size, mtime,
hash, Dropbox path), from which it can be restored:
    rclone copy <dropbox path> <experiment folder>

Skipped experiments: without database or without tracking (nothing to compare the
chunks with), multi-arena recordings (arenas/<n>/ are cropped from the top-level
chunks, each with its own tracked range), folders that resolve into the backup.

Nothing is deleted without --delete (default: dry run, prints what would go):
    python -m idtrackerai_validator_server.trim_raw FlyHostel1/6X/2025-10-08_16-00-00 ...
    python -m idtrackerai_validator_server.trim_raw --from-file experiments.txt --delete
Experiments on different drives run in parallel, one at a time per drive.
"""
import os
import sys
import json
import time
import logging
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from idtrackerai_validator_server import dropbox
from idtrackerai_validator_server.data_management import (
    BACKUP_ROOT, RAW_RE, compact_ranges, flat_name, tracked_chunks,
)
from idtrackerai_validator_server.experiments import to_relpath

logger = logging.getLogger(__name__)

LOG_NAME = "deleted_raw_chunks.jsonl"


def candidates(folder, tracked):
    """[(name, chunk)] of the raw chunk files outside the tracked chunks (first, last)."""
    found = []
    for entry in os.scandir(folder):
        match = RAW_RE.fullmatch(entry.name)
        if match and entry.is_file(follow_symlinks=False):
            chunk = int(match.group(1))
            if not tracked[0] <= chunk <= tracked[1]:
                found.append((entry.name, chunk))
    return sorted(found)


def skip_reason(root, experiment):
    """Why `experiment` must not be trimmed, or None."""
    folder = os.path.join(root, experiment)
    if not os.path.isdir(folder):
        return "no such folder"
    backup = os.path.realpath(BACKUP_ROOT)
    if os.path.realpath(folder).startswith(backup + os.sep):
        return f"folder resolves into the backup ({os.path.realpath(folder)})"
    if os.path.isdir(os.path.join(folder, "arenas")):
        return "multi-arena recording"
    if not os.path.exists(os.path.join(folder, f"{flat_name(experiment)}.db")):
        return "no database"
    return None


def trim_experiment(root, experiment, delete=False, out=print):
    """Delete (or with delete=False list) the verified raw chunk files outside tracking.
    Returns a summary dict."""
    summary = {"experiment": experiment, "deleted": 0, "freed": 0, "kept": {}, "skipped": None}
    reason = skip_reason(root, experiment)
    folder = os.path.join(root, experiment)
    tracked = None
    if reason is None:
        tracked, _ = tracked_chunks(os.path.join(folder, f"{flat_name(experiment)}.db"))
        if tracked is None:
            reason = "no tracking in ROI_0"
    if reason:
        summary["skipped"] = reason
        out(f"{experiment}: skipped, {reason}")
        return summary

    files = candidates(folder, tracked)
    out(f"{experiment}: tracked chunks {tracked[0]}-{tracked[1]}, "
        f"{len(files)} raw files outside (chunks {compact_ranges({c for _, c in files}) or '-'})")
    if not files:
        return summary

    kept = defaultdict(list)
    try:
        remote = dropbox.list_remote(experiment)
    except Exception as error:      # rclone failed: keep everything
        summary["skipped"] = f"rclone: {error}"
        out(f"{experiment}: skipped, {summary['skipped']}")
        return summary
    if remote is None:
        summary["skipped"] = "folder not on Dropbox"
        out(f"{experiment}: skipped, {summary['skipped']}")
        return summary

    hashed = []                     # (name, size, mtime, remote entry)
    for name, _ in files:
        path = os.path.join(folder, name)
        entry = remote.get(name)
        st = os.stat(path, follow_symlinks=False)
        status = dropbox.compare(path, entry, verify=True)
        if status == dropbox.VERIFIED:
            hashed.append((name, st.st_size, st.st_mtime, entry, st.st_blocks * 512))
        else:
            kept[status].append(name)

    if delete and hashed:
        try:
            again = dropbox.list_remote(experiment) or {}
        except Exception as error:
            again = {}
            out(f"{experiment}: second Dropbox listing failed ({error}), nothing deleted")
        log_path = os.path.join(folder, LOG_NAME)
        with open(log_path, "a") as log:
            for name, size, mtime, entry, used in hashed:
                path = os.path.join(folder, name)
                st = os.stat(path, follow_symlinks=False)
                if again.get(name) != entry:
                    kept["changed on Dropbox"].append(name)
                    continue
                if (st.st_size, st.st_mtime) != (size, mtime):
                    kept["changed locally"].append(name)
                    continue
                os.remove(path)
                log.write(json.dumps({
                    "deleted_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "file": name, "size": size,
                    "mtime": mtime, "dropbox_hash": entry[1],
                    "dropbox_path": f"{dropbox.REMOTE}/{experiment}/{name}",
                }) + "\n")
                log.flush()
                summary["deleted"] += 1
                summary["freed"] += used
    else:
        summary["deleted"] = len(hashed)
        summary["freed"] = sum(used for *_, used in hashed)

    summary["kept"] = {status: names for status, names in kept.items()}
    verb = "deleted" if delete else "would delete"
    out(f"{experiment}: {verb} {summary['deleted']} files ({summary['freed'] / 1e9:.2f} GB)"
        + (f", log in {os.path.join(folder, LOG_NAME)}" if delete and summary["deleted"] else ""))
    for status, names in kept.items():
        out(f"{experiment}:   kept {len(names)} ({status}): {', '.join(names[:10])}"
            + (" ..." if len(names) > 10 else ""))
    return summary


def drive_of(root, experiment):
    """Device the experiment folder is on (the FlyHostel<N> links point to different drives)."""
    try:
        return os.stat(os.path.join(root, experiment)).st_dev
    except OSError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Delete raw chunk files outside the tracked chunks that are verified on Dropbox ("
                    + dropbox.REMOTE + "). Dry run unless --delete.")
    parser.add_argument("experiments", nargs="*", help="FlyHostel<N>/<X>X/<date>_<time> (or underscore form)")
    parser.add_argument("--from-file", help="file with one experiment per line")
    parser.add_argument("--delete", action="store_true", help="really delete (default: dry run)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    root = os.environ["FLYHOSTEL_VIDEOS"]
    experiments = list(args.experiments)
    if args.from_file:
        with open(args.from_file) as handle:
            experiments += [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    experiments = [to_relpath(e) for e in experiments]
    if not experiments:
        parser.error("no experiments given")
    if not args.delete:
        print("Dry run: nothing is deleted (pass --delete)")

    by_drive = defaultdict(list)
    for experiment in dict.fromkeys(experiments):
        by_drive[drive_of(root, experiment)].append(experiment)

    def run_drive(queue):
        return [trim_experiment(root, experiment, delete=args.delete) for experiment in queue]

    with ThreadPoolExecutor(max(1, len(by_drive))) as pool:
        summaries = [s for result in pool.map(run_drive, by_drive.values()) for s in result]

    deleted = sum(s["deleted"] for s in summaries)
    freed = sum(s["freed"] for s in summaries)
    kept = sum(len(v) for s in summaries for v in s["kept"].values())
    skipped = [s for s in summaries if s["skipped"]]
    print(f"\n{'Deleted' if args.delete else 'Would delete'} {deleted} files, "
          f"{freed / 1e12:.3f} TB; kept {kept} unverified files; {len(skipped)} experiments skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
