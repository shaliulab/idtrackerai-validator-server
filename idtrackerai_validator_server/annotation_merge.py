"""
annotation_merge.py  —  merge two copies of an Annotate tab database (validator_annotations.db).

Copies of one experiment's annotations travel through Google Drive (gdrive_exchange.py):
the publisher's database is the reference; each annotator starts from a snapshot of it,
annotates, uploads their copy; the publisher imports it; annotators pull the publisher's
newer snapshots (other annotators' merged work, new solver proposals).

What changed where comes from `log` (every save, relabel and delete of a frame is a row; a
delete has boxes "null"), compared by log rowid, not by time: each computer has its own
clock, rowids only grow. The frame's content is read from `frames` / `boxes`.

- A source's changes: frames with a log row after `remote_since` (the rowid in the source's
  log where its copy of us ended, or what was already imported from it), leaving out the rows
  the source itself imported from us (its `exchange_imports` labelled with our name): our
  own work coming back.
- Our changes: frames with a log row after `local_since` (rowid in our log of what the
  source had seen of us), leaving out what we imported from that source.
- Per frame of the source: same content -> nothing to do; changed only there -> taken;
  changed on both sides with different content -> conflict, resolved by `prefer`
  ("newest" by the frames' `updated` time, "local", "remote") or per frame.

Imported frames are written as a save by the source's author (`frames.updated` the
source's, a `log` row with the source's time) and remembered in `exchange_imports`
(source label, frame, our log rowid); `exchange_sources` keeps per source the rowid of its
log imported up to. Never VACUUM these databases: it may renumber rowids.
"""
import json
import time
import sqlite3

from idtrackerai_validator_server.box_annotations import connect_path

PREFER = ("newest", "local", "remote")


def ensure_tables(con):
    con.executescript("""
        CREATE TABLE IF NOT EXISTS exchange_imports (source TEXT, frame_number INTEGER, log_rowid INTEGER);
        CREATE INDEX IF NOT EXISTS exchange_imports_src ON exchange_imports (source, frame_number);
        CREATE TABLE IF NOT EXISTS exchange_sources (source TEXT PRIMARY KEY, watermark INTEGER, imported_at REAL);
    """)


def max_log_rowid(con):
    return con.execute("SELECT COALESCE(MAX(rowid), 0) FROM log").fetchone()[0]


def watermark(con, source):
    ensure_tables(con)
    row = con.execute("SELECT watermark FROM exchange_sources WHERE source = ?", (source,)).fetchone()
    return row[0] if row else None


def _imported_rowids(con, source):
    """log rowids of `con` that came from importing `source`"""
    return {r[0] for r in con.execute("SELECT log_rowid FROM exchange_imports WHERE source = ?", (source,))}


def changed_frames(con, since, exclude_source=None):
    """{frame: (rowid, t, author)} of the last log row after rowid `since`, ignoring the rows imported from
    `exclude_source`"""
    ensure_tables(con)
    skip = _imported_rowids(con, exclude_source) if exclude_source else set()
    out = {}
    for rowid, t, author, fn in con.execute(
            "SELECT rowid, t, author, frame_number FROM log WHERE rowid > ? ORDER BY rowid", (since,)):
        if rowid not in skip:
            out[fn] = (rowid, t, author)
    return out


def frame_state(con, frame_number):
    """None (not annotated) or {"author", "updated", "boxes": [...]}"""
    meta = con.execute("SELECT author, updated FROM frames WHERE frame_number = ?", (frame_number,)).fetchone()
    if meta is None:
        return None
    boxes = [{"x0": a, "y0": b, "x1": c, "y1": d, "label": lab or "", **({"track": tr} if tr else {})}
             for a, b, c, d, lab, tr in con.execute(
                 "SELECT x0, y0, x1, y1, label, track FROM boxes WHERE frame_number = ? ORDER BY rowid", (frame_number,))]
    return {"author": meta[0], "updated": meta[1], "boxes": boxes}


def _key(state):
    if state is None:
        return None
    return sorted((round(b["x0"], 1), round(b["y0"], 1), round(b["x1"], 1), round(b["y1"], 1), b["label"], b.get("track"))
                  for b in state["boxes"])


def plan(local_path, remote_path, source, me, remote_since, local_since):
    """What importing `remote_path` (labelled `source`) into `local_path` (whose owner the source calls `me`)
    would do (remote_path must be a copy: tables may be added to it). remote_since: rowid of the source's log already known here (None: the stored watermark, else 0);
    local_since: rowid of our log the source had seen."""
    local = connect_path(local_path)
    remote = connect_path(remote_path)      # a downloaded copy: the bookkeeping tables may be added to it
    try:
        ensure_tables(local)
        ensure_tables(remote)
        stored = watermark(local, source)
        since = max(remote_since or 0, stored or 0)
        theirs = changed_frames(remote, since, exclude_source=me)
        ours = changed_frames(local, local_since or 0, exclude_source=source)
        take, same, conflicts = [], 0, []
        for fn in sorted(theirs):
            remote_state = frame_state(remote, fn)
            local_state = frame_state(local, fn)
            if _key(remote_state) == _key(local_state):
                same += 1
                continue
            entry = {"frame_number": fn, "remote": remote_state, "local": local_state,
                     "remote_author": theirs[fn][2], "remote_t": theirs[fn][1]}
            if fn in ours:
                entry["local_author"], entry["local_t"] = ours[fn][2], ours[fn][1]
                conflicts.append(entry)
            else:
                take.append(entry)
        return {"source": source, "remote_since": since, "remote_max": max_log_rowid(remote),
                "take": take, "same": same, "conflicts": conflicts}
    finally:
        local.close()
        remote.close()


def _winner(entry, prefer):
    if prefer == "local":
        return "local"
    if prefer == "remote":
        return "remote"
    lt = (entry["local"] or {}).get("updated") or entry.get("local_t") or 0
    rt = (entry["remote"] or {}).get("updated") or entry.get("remote_t") or 0
    return "remote" if rt > lt else "local"


def apply(local_path, the_plan, prefer="newest", overrides=None, lock=None):
    """Write the plan into local_path. overrides: {frame_number: "local" | "remote"} for conflicts.
    -> {"taken", "conflicts_remote", "conflicts_local", "watermark"}"""
    if prefer not in PREFER:
        raise ValueError(f"prefer must be one of {PREFER}")
    overrides = {int(k): v for k, v in (overrides or {}).items()}
    writes = list(the_plan["take"])
    kept = 0
    for entry in the_plan["conflicts"]:
        if overrides.get(entry["frame_number"], _winner(entry, prefer)) == "remote":
            writes.append(entry)
        else:
            kept += 1
    source = the_plan["source"]

    def write():
        con = connect_path(local_path)
        try:
            with con:
                ensure_tables(con)
                for entry in writes:
                    fn, state = entry["frame_number"], entry["remote"]
                    con.execute("DELETE FROM boxes WHERE frame_number = ?", (fn,))
                    if state is None:
                        con.execute("DELETE FROM frames WHERE frame_number = ?", (fn,))
                        boxes = None
                    else:
                        boxes = state["boxes"]
                        con.executemany(
                            "INSERT INTO boxes (frame_number, x0, y0, x1, y1, label, track) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            [(fn, b["x0"], b["y0"], b["x1"], b["y1"], b["label"], b.get("track")) for b in boxes])
                        con.execute("INSERT OR REPLACE INTO frames VALUES (?, ?, ?, ?)",
                                    (fn, state["author"], state["updated"], len(boxes)))
                    cur = con.execute("INSERT INTO log VALUES (?, ?, ?, ?)",
                                      (entry.get("remote_t") or time.time(), entry.get("remote_author") or source,
                                       fn, json.dumps(boxes)))
                    con.execute("INSERT INTO exchange_imports VALUES (?, ?, ?)", (source, fn, cur.lastrowid))
                con.execute("INSERT OR REPLACE INTO exchange_sources VALUES (?, ?, ?)",
                            (source, max(the_plan["remote_max"], watermark(con, source) or 0), time.time()))
        finally:
            con.close()

    if lock is not None:
        with lock:
            write()
    else:
        write()
    return {"taken": len(the_plan["take"]), "conflicts_remote": len(writes) - len(the_plan["take"]),
            "conflicts_local": kept, "watermark": the_plan["remote_max"]}


def copy_proposals(local_path, remote_path, lock=None):
    """Replace the solver proposals of local_path by remote_path's when its last proposal run is newer.
    -> True when copied"""
    remote = sqlite3.connect(f"file:{remote_path}?mode=ro", uri=True)
    try:
        theirs = remote.execute("SELECT MAX(t) FROM proposal_runs").fetchone()[0]
        if theirs is None:
            return False
        frames = remote.execute("SELECT * FROM proposal_frames").fetchall()
        props = remote.execute("SELECT frame_number, x0, y0, x1, y1, label, note, confidence FROM proposals").fetchall()
        runs = remote.execute("SELECT t, model, scenes, frames, summary FROM proposal_runs").fetchall()
    except sqlite3.OperationalError:
        return False
    finally:
        remote.close()

    def write():
        con = connect_path(local_path)
        try:
            mine = con.execute("SELECT MAX(t) FROM proposal_runs").fetchone()[0]
            if mine is not None and mine >= theirs:
                return False
            with con:
                con.execute("DELETE FROM proposal_frames")
                con.execute("DELETE FROM proposals")
                con.execute("DELETE FROM proposal_runs")
                con.executemany("INSERT INTO proposal_frames VALUES (?, ?)", frames)
                con.executemany("INSERT INTO proposals VALUES (?, ?, ?, ?, ?, ?, ?, ?)", props)
                con.executemany("INSERT INTO proposal_runs VALUES (?, ?, ?, ?, ?)", runs)
            return True
        finally:
            con.close()

    if lock is not None:
        with lock:
            return write()
    return write()


def snapshot(src_path, dst_path, lock=None):
    """Consistent copy of an annotations database (sqlite backup). -> max log rowid of the copy"""
    def copy():
        src = connect_path(src_path)
        dst = sqlite3.connect(dst_path)
        try:
            src.backup(dst)
            return max_log_rowid(dst)
        finally:
            src.close()
            dst.close()
    if lock is not None:
        with lock:
            return copy()
    return copy()


def reset_exchange(path):
    """Forget the import bookkeeping of a copy received from someone else (it is theirs, not ours)"""
    con = connect_path(path)
    try:
        with con:
            ensure_tables(con)
            con.execute("DELETE FROM exchange_imports")
            con.execute("DELETE FROM exchange_sources")
    finally:
        con.close()


def summary(path):
    """{"frames": saved frames, "authors": {author: frames}, "last_rowid"} of an annotations database"""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        authors = dict(con.execute("SELECT COALESCE(author, ''), COUNT(*) FROM frames GROUP BY author").fetchall())
        return {"frames": sum(authors.values()), "authors": authors, "last_rowid": max_log_rowid(con)}
    finally:
        con.close()
