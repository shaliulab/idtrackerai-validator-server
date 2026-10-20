"""
upload_slp_dropbox.py  —  replace the Dropbox copies of the .slp files compress_slp.py
compressed with the compressed files.

compress_slp.py repacked ~260k SLEAP prediction files in place (lossless, verified,
~3.7x smaller), but the Dropbox mirror of $FLYHOSTEL_VIDEOS ($DROPBOX_REMOTE, see
dropbox.py) still holds the uncompressed originals. For every file in its log
($COMPRESS_SLP_LOG) the Dropbox entry is listed (rclone lsjson --hash) and
    not on Dropbox        skipped (never uploaded; not this script's job)
    local changed         skipped: the local file no longer has its compressed size
    already compressed    Dropbox has the local file's size
    to upload             Dropbox has the size the file had before compression
    unexpected size       skipped: Dropbox has some other version
With --apply the files "to upload" are copied with rclone copy --files-from (Dropbox
keeps the replaced version in its version history), then listed again: each one must
have the local size and the Dropbox content hash of the local file (computed by reading
it), or it is reported. Every verified upload is logged in $UPLOAD_SLP_LOG
(default ~/.cache/idtrackerai-validator/uploaded_slp.jsonl).

Nothing is uploaded without --apply (default: dry run, counts the files):
    python -m idtrackerai_validator_server.upload_slp_dropbox [FlyHostel1/6X/2025-10-08_16-00-00 ...]
    python -m idtrackerai_validator_server.upload_slp_dropbox --apply
No experiment given: every experiment of the compression log.
"""
import os
import sys
import json
import time
import tempfile
import argparse
import subprocess
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

from idtrackerai_validator_server import dropbox
from idtrackerai_validator_server.compress_slp import LOG_FILE as COMPRESS_LOG
from idtrackerai_validator_server.experiments import to_relpath

UPLOAD_LOG = os.path.expanduser(os.environ.get(
    "UPLOAD_SLP_LOG", "~/.cache/idtrackerai-validator/uploaded_slp.jsonl"))
SA_DIR = "flyhostel/single_animal"

NOT_ON_DROPBOX = "not on Dropbox"
LOCAL_CHANGED = "local changed"
DONE = "already compressed"
TO_UPLOAD = "to upload"
UNEXPECTED = "unexpected size"


def read_compress_log(root):
    """single_animal folder (relative to root) -> {name inside it: (before, after)}."""
    folders = defaultdict(dict)
    with open(COMPRESS_LOG) as handle:
        for line in handle:
            entry = json.loads(line)
            relpath = dropbox.to_relpath(entry["file"], root)
            folder, _, name = relpath.partition(SA_DIR + "/")
            folders[folder + SA_DIR][name] = (entry["before"], entry["after"])
    return folders


def classify(root, folder, files):
    """name -> status of the logged files of one single_animal folder."""
    remote = dropbox.list_remote(folder, recursive=True) or {}
    statuses = {}
    for name, (before, after) in files.items():
        try:
            size = os.path.getsize(os.path.join(root, folder, name))
        except OSError:
            size = None
        entry = remote.get(name)
        if size != after:
            statuses[name] = LOCAL_CHANGED
        elif entry is None:
            statuses[name] = NOT_ON_DROPBOX
        elif entry[0] == after:
            statuses[name] = DONE
        elif entry[0] == before:
            statuses[name] = TO_UPLOAD
        else:
            statuses[name] = UNEXPECTED
    return statuses


def upload(root, folder, names, transfers):
    """Copy `names` of a single_animal folder to Dropbox. Returns the names verified there."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(names) + "\n")
        files_from = handle.name
    try:
        proc = subprocess.run(
            [dropbox.RCLONE, "copy", "--files-from-raw", files_from, "--no-traverse",
             "--transfers", str(transfers), os.path.join(root, folder), f"{dropbox.REMOTE}/{folder}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    finally:
        os.remove(files_from)
    error = None
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        error = f"rclone copy failed: {lines[-1] if lines else proc.returncode}"
    remote = dropbox.list_remote(folder, recursive=True) or {}
    verified = [name for name in names
                if dropbox.compare(os.path.join(root, folder, name), remote.get(name), verify=True)
                == dropbox.VERIFIED]
    return verified, error


def process(root, folder, files, apply, transfers, out, log):
    statuses = classify(root, folder, files)
    counts = Counter(statuses.values())
    todo = sorted(name for name, status in statuses.items() if status == TO_UPLOAD)
    saving = sum(files[name][0] - files[name][1] for name in todo)
    summary = {"folder": folder, "counts": counts, "todo": len(todo), "saving": saving,
               "uploaded": 0, "failed": 0}
    text = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    if not apply or not todo:
        out(f"{folder}: {text} ({saving / 1e9:.2f} GB to free)")
        return summary
    verified, error = upload(root, folder, todo, transfers)
    for name in verified:
        log(os.path.join(root, folder, name), *files[name])
    summary["uploaded"] = len(verified)
    summary["failed"] = len(todo) - len(verified)
    out(f"{folder}: uploaded {len(verified)}/{len(todo)}"
        + (f", {summary['failed']} NOT verified" if summary["failed"] else "")
        + (f" ({error})" if error else "") + f"; before: {text}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Upload the .slp files compressed by compress_slp.py "
                                                 "over their uncompressed Dropbox copies. Dry run unless --apply.")
    parser.add_argument("experiments", nargs="*", help="FlyHostel<N>/<X>X/<date>_<time> (default: all logged)")
    parser.add_argument("--apply", action="store_true", help="really upload (default: dry run)")
    parser.add_argument("--jobs", type=int, default=4, help="folders processed at once (default 4)")
    parser.add_argument("--transfers", type=int, default=4, help="rclone transfers per folder (default 4)")
    args = parser.parse_args(argv)

    root = os.environ["FLYHOSTEL_VIDEOS"]
    folders = read_compress_log(root)
    if args.experiments:
        wanted = tuple(to_relpath(e).rstrip("/") + "/" for e in args.experiments)
        folders = {f: files for f, files in folders.items() if f.startswith(wanted)}
    if not args.apply:
        print("Dry run: nothing is uploaded (pass --apply)")

    os.makedirs(os.path.dirname(UPLOAD_LOG), exist_ok=True)
    log_handle = open(UPLOAD_LOG, "a") if args.apply else None

    def log(path, before, after):
        log_handle.write(json.dumps({"uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "file": path,
                                     "remote": f"{dropbox.REMOTE}/{dropbox.to_relpath(path, root)}",
                                     "before": before, "after": after}) + "\n")
        log_handle.flush()

    def run(item):
        folder, files = item
        try:
            return process(root, folder, files, args.apply, args.transfers, print, log)
        except Exception as error:  # one folder failing does not stop the others
            print(f"{folder}: {error}")
            return None

    try:
        with ThreadPoolExecutor(args.jobs) as pool:
            summaries = [s for s in pool.map(run, sorted(folders.items())) if s]
    finally:
        if log_handle:
            log_handle.close()

    counts = sum((s["counts"] for s in summaries), Counter())
    print("\n" + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())))
    print(f"{len(summaries)}/{len(folders)} folders listed")
    if args.apply:
        print(f"Uploaded and verified {sum(s['uploaded'] for s in summaries)} files, "
              f"{sum(s['failed'] for s in summaries)} not verified")
    else:
        print(f"Would upload {sum(s['todo'] for s in summaries)} files, "
              f"freeing {sum(s['saving'] for s in summaries) / 1e12:.3f} TB on Dropbox")
    return 0


if __name__ == "__main__":
    sys.exit(main())
