"""
dropbox_files.py  —  the files of a Dropbox folder next to their local copies, and
deleting them on Dropbox, for the "Files on Dropbox" card of the Data management tab
(the Dropbox web interface has no regex and acts on one page of files at a time).

A folder is a path relative to $DROPBOX_REMOTE, which mirrors $FLYHOSTEL_VIDEOS (see
dropbox.py), usually an experiment folder FlyHostel<N>/<X>X/<date>_<time>. Listing it
(POST /api/dropbox/files/list, a job: `rclone lsjson -R --hash`, ~6 s for 3000 files)
compares each Dropbox file with the file at the same path in $FLYHOSTEL_VIDEOS:
    no local copy, different size, same size (contents not compared yet),
    verified / different content (local Dropbox hash known from an earlier delete)
The last listing of each folder is kept in memory (GET /api/dropbox/files).

The client selects files with a regex and sends their paths. Deleting
(POST /api/dropbox/files/delete, a job, one at a time) removes a file on Dropbox only
when the local copy is verified right before, as in trim_raw.py:
    1. list the folder on Dropbox again (sizes and Dropbox content hashes)
    2. hash each local copy (~150 MB/s) and compare with its Dropbox entry
    3. list the folder again; delete only the files whose Dropbox entry and local
       size/mtime did not change, with `rclone delete --files-from-raw`
    4. list the folder again: a file counts as deleted when it is gone
A local copy that is a link into $FLYHOSTEL_BACKUP does not count (it would leave a
single copy). With "safe": false (safe mode off in the client) steps 2 and the local
checks are skipped: every given file is deleted, also those with no local copy. Deleted files go to Dropbox's deleted files (recoverable from the web
interface for a while) and are logged to deleted_from_dropbox.jsonl next to
$DATA_MANAGEMENT_CACHE. Local hashes are kept in dropbox_hashes.json there, keyed by
real path, size and mtime.
"""
import os
import json
import time
import logging
import tempfile
import threading
import subprocess

from flask import jsonify, request

from idtrackerai_validator_server import dropbox
from idtrackerai_validator_server import data_management as dm
from idtrackerai_validator_server.jobs import start_job

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.dirname(dm.CACHE_FILE)
HASH_CACHE = os.path.join(CACHE_DIR, "dropbox_hashes.json")
DELETE_LOG = os.path.join(CACHE_DIR, "deleted_from_dropbox.jsonl")
# files per rclone delete call
DELETE_BATCH = 500

NO_LOCAL = "no local copy"
IN_BACKUP = "local copy in backup"
CHANGED_DROPBOX = "changed on Dropbox"
CHANGED_LOCAL = "changed locally"

_listings = {}              # folder -> last listing
_listings_lock = threading.Lock()
_hashes = {}                # real path -> [size, mtime, Dropbox hash]
_hashes_lock = threading.Lock()
_delete_lock = threading.Lock()


def normalize_folder(folder):
    """Folder relative to $DROPBOX_REMOTE, without leading/trailing slashes; ValueError
    when empty or leaving it."""
    folder = (folder or "").strip().strip("/")
    parts = folder.split("/")
    if not folder or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"Not a folder of {dropbox.REMOTE}: {folder!r}")
    return folder


def cached_hash(path, st):
    """Dropbox hash of the local file from the cache, if its size and mtime match."""
    with _hashes_lock:
        entry = _hashes.get(os.path.realpath(path))
    return entry[2] if entry and entry[:2] == [st.st_size, st.st_mtime] else None


def local_hash(path, st):
    found = cached_hash(path, st)
    if found is None:
        found = dropbox.content_hash(path)
        with _hashes_lock:
            _hashes[os.path.realpath(path)] = [st.st_size, st.st_mtime, found]
    return found


def local_status(local, remote_entry):
    """(status, local size) of the local copy `local` of a Dropbox file."""
    try:
        st = os.stat(local)
    except OSError:
        return NO_LOCAL, None
    if not os.path.isfile(local):
        return NO_LOCAL, None
    if os.path.realpath(local).startswith(os.path.realpath(dm.BACKUP_ROOT) + os.sep):
        return IN_BACKUP, st.st_size
    if st.st_size != remote_entry[0]:
        return dropbox.DIFFERENT_SIZE, st.st_size
    known = cached_hash(local, st)
    if known is None or remote_entry[1] is None:
        return dropbox.SAME_SIZE, st.st_size
    return (dropbox.VERIFIED if known == remote_entry[1] else dropbox.DIFFERENT_CONTENT), st.st_size


def list_folder(root, folder):
    remote = dropbox.list_remote(folder, recursive=True)
    if remote is None:
        raise RuntimeError(f"{dropbox.REMOTE}/{folder} not found on Dropbox")
    files = []
    for path, entry in sorted(remote.items()):
        status, local_size = local_status(os.path.join(root, folder, path), entry)
        files.append({"path": path, "size": entry[0], "local_size": local_size, "status": status})
    return {"folder": folder, "remote": f"{dropbox.REMOTE}/{folder}", "local": os.path.join(root, folder),
            "listed_at": time.time(), "files": files}


def rclone_delete(folder, paths):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("".join(p + "\n" for p in paths))
        listing = handle.name
    try:
        proc = subprocess.run(
            [dropbox.RCLONE, "delete", "--files-from-raw", listing, f"{dropbox.REMOTE}/{folder}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3600)
    finally:
        os.unlink(listing)
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        logger.warning("rclone delete in %s: %s", folder, proc.stderr)
        return lines[-1] if lines else f"rclone exited with {proc.returncode}"
    return None


def delete_files(root, folder, paths, job, safe=True):
    """Delete `paths` (relative to folder) on Dropbox if their local copy is verified.
    safe=False deletes them whatever is here (still only if their Dropbox entry did not
    change between the two listings). Items of `job`: 0 list, 1 hash, 2 delete."""
    kept = {}

    def keep(reason, path):
        kept.setdefault(reason, []).append(path)

    job.update(0, status="running")
    remote = dropbox.list_remote(folder, recursive=True) or {}
    job.update(0, progress=0.5)

    job.update(1, status="running")
    todo = [p for p in paths if p in remote]
    for p in paths:
        if p not in remote:
            keep(dropbox.MISSING, p)
    total = sum(remote[p][0] for p in todo) or 1
    read = 0
    verified = []               # (path, local, size, mtime, entry); size None: not verified
    for path in ([] if safe else todo):
        verified.append((path, None, None, None, remote[path]))
    for path in (todo if safe else []):
        entry = remote[path]
        local = os.path.join(root, folder, path)
        status, _ = local_status(local, entry)
        if status in (dropbox.SAME_SIZE, dropbox.VERIFIED, dropbox.DIFFERENT_CONTENT):
            st = os.stat(local)
            if entry[1] is not None and local_hash(local, st) == entry[1]:
                verified.append((path, local, st.st_size, st.st_mtime, entry))
            else:
                keep(dropbox.DIFFERENT_CONTENT, path)
        else:
            keep(status, path)
        read += entry[0]
        job.update(1, progress=read / total)
    dm._save_json(HASH_CACHE, _hashes, _hashes_lock)
    job.update(1, status="done", progress=1.0)

    again = dropbox.list_remote(folder, recursive=True) or {}
    job.update(0, status="done", progress=1.0)
    job.update(2, status="running")
    ready = []
    for path, local, size, mtime, entry in verified:
        if local is None:
            if again.get(path) != entry:
                keep(CHANGED_DROPBOX, path)
            else:
                ready.append((path, None, entry))
            continue
        try:
            st = os.stat(local)
        except OSError:
            keep(CHANGED_LOCAL, path)
            continue
        if again.get(path) != entry:
            keep(CHANGED_DROPBOX, path)
        elif (st.st_size, st.st_mtime) != (size, mtime):
            keep(CHANGED_LOCAL, path)
        else:
            ready.append((path, local, entry))

    errors = []
    for start in range(0, len(ready), DELETE_BATCH):
        error = rclone_delete(folder, [p for p, _, _ in ready[start:start + DELETE_BATCH]])
        if error:
            errors.append(error)
        job.update(2, progress=0.95 * min(len(ready), start + DELETE_BATCH) / max(1, len(ready)))

    after = (dropbox.list_remote(folder, recursive=True) or {}) if ready else again
    deleted = [(p, local, entry) for p, local, entry in ready if p not in after]
    for p, _, _ in ready:
        if p in after:
            keep("rclone could not delete it", p)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(DELETE_LOG, "a") as log:
        for path, local, entry in deleted:
            log.write(json.dumps({
                "deleted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dropbox_path": f"{dropbox.REMOTE}/{folder}/{path}", "size": entry[0],
                "dropbox_hash": entry[1], "local_copy": local and os.path.realpath(local),
                "safe": safe,
            }) + "\n")
    job.update(2, status="done", progress=1.0)

    with _listings_lock:
        if folder in _listings:     # reflect the deletion without listing again
            gone = {p for p, _, _ in deleted}
            _listings[folder]["files"] = [f for f in _listings[folder]["files"] if f["path"] not in gone]
    return {
        "folder": folder, "requested": len(paths), "deleted": len(deleted),
        "freed": sum(entry[0] for _, _, entry in deleted),
        "kept": {reason: sorted(names) for reason, names in kept.items()},
        "errors": errors, "log": DELETE_LOG, "safe": safe,
    }


def register_dropbox_files(app):
    root = os.environ["FLYHOSTEL_VIDEOS"]
    dm._load_json(HASH_CACHE, _hashes)

    @app.route("/api/dropbox/files", methods=["GET"])
    def get_dropbox_files():
        """Last listing of ?folder=, null if never listed."""
        try:
            folder = normalize_folder(request.args.get("folder"))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        with _listings_lock:
            listing = _listings.get(folder)
        return jsonify({"listing": listing, "remote": dropbox.REMOTE, "deleting": _delete_lock.locked()})

    @app.route("/api/dropbox/files/list", methods=["POST"])
    def list_dropbox_files():
        """{"folder"}: job listing it on Dropbox and comparing with the local copies."""
        try:
            folder = normalize_folder((request.get_json(silent=True) or {}).get("folder"))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        def work(job):
            job.update(0, status="running")
            listing = list_folder(root, folder)
            with _listings_lock:
                _listings[folder] = listing
            job.update(0, status="done", progress=1.0)
            job.result = {"folder": folder, "files": len(listing["files"])}

        return jsonify(start_job(f"List {folder} on Dropbox", [folder], work).to_dict())

    @app.route("/api/dropbox/files/delete", methods=["POST"])
    def delete_dropbox_files():
        """{"folder", "paths": [...]}: job deleting those files on Dropbox (see delete_files)."""
        body = request.get_json(silent=True) or {}
        try:
            folder = normalize_folder(body.get("folder"))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        paths = body.get("paths") or []
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
            return jsonify({"error": "No files given"}), 400
        if any(p.startswith("/") or ".." in p.split("/") or "\n" in p for p in paths):
            return jsonify({"error": "Paths must be relative to the folder"}), 400
        paths = list(dict.fromkeys(paths))
        safe = body.get("safe") is not False   # unsafe only when asked explicitly
        if not _delete_lock.acquire(blocking=False):
            return jsonify({"error": "A Dropbox deletion is already running"}), 409

        def work(job):
            try:
                job.result = delete_files(root, folder, paths, job, safe=safe)
            finally:
                _delete_lock.release()

        try:
            job = start_job(f"Delete {len(paths)} files of {folder} on Dropbox"
                            + ("" if safe else " (safe mode off)"),
                            ["list Dropbox", "verify local copies" if safe else "no local check",
                             "delete on Dropbox"], work)
        except Exception:
            _delete_lock.release()
            raise
        return jsonify(job.to_dict())
