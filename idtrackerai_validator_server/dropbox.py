"""
dropbox.py  —  is a file of $FLYHOSTEL_VIDEOS backed up to Dropbox?

The lab's Dropbox team folder Data/flyhostel_data/videos mirrors $FLYHOSTEL_VIDEOS
(the Dropbox links of the metadata sheets point there). The Dropbox client does not
run on this computer, so the folder is read with rclone: $DROPBOX_REMOTE, default
dropbox:/Data/flyhostel_data/videos (the leading slash reaches the team space;
dropbox:Data is not found).

A local file is compared with the file at the same relative path on Dropbox:
    missing            no such file on Dropbox
    different size
    same size          sizes match, contents not compared
    verified           also the Dropbox content hash matches, computed locally by
                       reading the file (SHA-256 of the SHA-256 of each 4 MiB block,
                       rclone's DropboxHash)
    different content  same size, different hash
Listing a folder (rclone lsjson --hash: names, sizes and hashes, nothing downloaded)
takes ~7 s for an experiment folder; hashing reads ~150 MB/s.

From the shell:
    python -m idtrackerai_validator_server.dropbox [--verify] FILE...
"""
import os
import sys
import json
import hashlib
import argparse
import subprocess
from collections import defaultdict

REMOTE = os.environ.get("DROPBOX_REMOTE", "dropbox:/Data/flyhostel_data/videos").rstrip("/")
RCLONE = os.environ.get("RCLONE", "rclone")
BLOCK_SIZE = 4 * 1024 * 1024

MISSING = "missing"
DIFFERENT_SIZE = "different size"
SAME_SIZE = "same size"
VERIFIED = "verified"
DIFFERENT_CONTENT = "different content"


def content_hash(path):
    """Dropbox content hash of a local file."""
    digests = []
    with open(path, "rb") as handle:
        while True:
            block = handle.read(BLOCK_SIZE)
            if not block:
                break
            digests.append(hashlib.sha256(block).digest())
    return hashlib.sha256(b"".join(digests)).hexdigest()


def list_remote(relpath, recursive=False, timeout=900):
    """name -> (size, content hash) of the files in REMOTE/relpath (names relative to it),
    None when the folder is not on Dropbox. Raises RuntimeError when rclone fails."""
    cmd = [RCLONE, "lsjson", "--files-only", "--hash", f"{REMOTE}/{relpath.strip('/')}"]
    if recursive:
        cmd.insert(2, "-R")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if proc.returncode != 0:
        if "directory not found" in proc.stderr:
            return None
        lines = proc.stderr.strip().splitlines()
        raise RuntimeError(f"rclone lsjson {relpath} failed: {lines[-1] if lines else proc.returncode}")
    return {entry["Path"]: (entry["Size"], (entry.get("Hashes") or {}).get("DropboxHash"))
            for entry in json.loads(proc.stdout)}


def compare(path, remote_entry, verify=False):
    """Status of the local file `path` given its Dropbox entry (size, hash) or None."""
    if remote_entry is None:
        return MISSING
    if os.path.getsize(path) != remote_entry[0]:
        return DIFFERENT_SIZE
    if not verify or remote_entry[1] is None:
        return SAME_SIZE
    return VERIFIED if content_hash(path) == remote_entry[1] else DIFFERENT_CONTENT


def to_relpath(path, root=None):
    """Path of a local file relative to $FLYHOSTEL_VIDEOS, also when given through the
    drive the FlyHostel<N> link points to (/media/vibflysleep/DATA1/FlyHostel1/...)."""
    root = os.path.abspath(root or os.environ["FLYHOSTEL_VIDEOS"])
    path = os.path.abspath(path)
    if path.startswith(root + os.sep):
        return os.path.relpath(path, root)
    for name in os.listdir(root):
        target = os.path.realpath(os.path.join(root, name))
        if os.path.isdir(target) and path.startswith(target + os.sep):
            return os.path.join(name, os.path.relpath(path, target))
    raise ValueError(f"{path} is not in {root}")


def check_files(paths, verify=False, root=None):
    """[(path, status)] for local files, listing each folder on Dropbox once."""
    by_folder = defaultdict(list)
    for path in paths:
        relpath = to_relpath(path, root)
        by_folder[os.path.dirname(relpath)].append((path, os.path.basename(relpath)))
    results = []
    for folder, files in by_folder.items():
        remote = list_remote(folder) or {}
        results += [(path, compare(path, remote.get(name), verify)) for path, name in files]
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Is each file backed up to Dropbox (" + REMOTE + ")?")
    parser.add_argument("files", nargs="+", help="files in $FLYHOSTEL_VIDEOS")
    parser.add_argument("--verify", action="store_true",
                        help="also compare the contents (reads each file whose size matches)")
    args = parser.parse_args(argv)
    ok = True
    for path, status in check_files(args.files, verify=args.verify):
        print(f"{status:17}  {path}")
        ok &= status in (VERIFIED, SAME_SIZE)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
