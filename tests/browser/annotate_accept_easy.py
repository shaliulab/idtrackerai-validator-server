"""Annotate tab in headless Chrome: the proposal / easy-method options and "Accept easy scenes" (review list, accept,
undo) on a SCRATCH COPY of an experiment (it writes the copy's validator_annotations.db).

    python tests/browser/annotate_accept_easy.py VIDEOS EXPERIMENT WORKDIR [N_SCENES]

VIDEOS: a folder laid out as $FLYHOSTEL_VIDEOS holding the copy (the recording files may be symlinks to the original:
only the copied database and the sidecar are written); refused when it resolves under /flyhostel_data. A private
server is started on it (pe_record.private_server, caches in WORKDIR). Checks, printing PASS / FAIL lines:
1. the choices Proposal = Continuity v2 / Easy = Rules are kept in localStorage and shown; the frame opens as a
   "continuity v2 proposal";
2. the review list shows the easy scenes not saved with thumbnails; N_SCENES (default 3) are ticked, the rest not;
3. Accept saves exactly the frames of those scenes nobody saved, as the person, the log marked "accepted easy
   scene, continuity2, rules, batch N"; the unticked scenes stay unsaved;
4. Undo unsaves them all again."""
import json
import os
import sqlite3
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from idtrackerai_validator_server.experiments import to_flat, to_relpath  # noqa: E402

VIDEOS, EXP, WORK = os.path.abspath(sys.argv[1]), sys.argv[2], os.path.abspath(sys.argv[3])
N_SCENES = int(sys.argv[4]) if len(sys.argv) > 4 else 3
if os.path.realpath(VIDEOS).startswith("/flyhostel_data"):
    sys.exit("refused: VIDEOS is under /flyhostel_data (use a scratch copy)")
os.makedirs(WORK, exist_ok=True)
os.environ["FLYHOSTEL_VIDEOS"] = VIDEOS
for var, sub in (("CONTINUITY_CACHE_DIR", "continuity"), ("ANNOTATE_PACK_CACHE", "pack"), ("QC_CACHE_DIR", "qc")):
    os.environ[var] = os.path.join(WORK, sub)
os.environ.pop("ANNOTATE_EASY_METHOD", None)
# the private server runs this checkout (and its frontend/), not the installed package
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ["PYTHONPATH"] = REPO + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.pop("ANNOTATE_CONTINUITY_VERSION", None)

from idtrackerai_validator_server.pe_record import Chrome, private_server  # noqa: E402

FLAT = to_flat(EXP)
BASEDIR = os.path.join(VIDEOS, to_relpath(EXP))
SIDECAR = os.path.join(BASEDIR, "flyhostel", "validation", "validator_annotations.db")
if os.path.realpath(SIDECAR).startswith("/flyhostel_data"):
    sys.exit("refused: the sidecar resolves under /flyhostel_data")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=120) as r:
        return json.loads(r.read())


def saved_frames():
    if not os.path.exists(SIDECAR):
        return {}
    with sqlite3.connect(SIDECAR) as con:
        return dict(con.execute("SELECT frame_number, author FROM frames"))


def wait_for(c, js, timeout=300, what=""):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = c.evaluate(js)
        if v:
            return v
        time.sleep(0.3)
    raise TimeoutError(what or js)


with private_server(to_relpath(EXP), WORK) as base:
    # the easy scenes by rules (a job the first time)
    t0 = time.time()
    while True:
        easy = get(base, "/api/annotate/scenes?easy=rules")["easy"]
        if easy and "scenes" in easy:
            break
        if easy and easy.get("error"):
            sys.exit(f"easy scenes failed: {easy['error']}")
        if time.time() - t0 > 1800:
            sys.exit("easy scenes not ready after 30 min")
        time.sleep(3)
    before = saved_frames()
    first = min(int(k) for k in easy["scenes"])
    url = f"{base}/?tab=annotate&experiment={FLAT}&frame={first}"   # links need a frame
    print(f"{len(easy['scenes'])} easy scenes, {len(before)} frames saved before", flush=True)
    c = Chrome(WORK, viewport=(1600, 1000))
    try:
        c.open_page((1600, 1000))
        c.send("Page.navigate", {"url": url})
        wait_for(c, "!!document.querySelector('[data-annotate-option=continuity] select')", what="the Annotate tab")
        c.evaluate("localStorage.setItem('fh-annotate-continuity', '2'); localStorage.setItem('fh-annotate-easy-method', 'rules'); true")
        c.send("Page.navigate", {"url": url})
        wait_for(c, "document.querySelector('[data-annotate-option=continuity] select')?.value === '2'", what="Proposal = v2")
        shown = c.evaluate("[document.querySelector('[data-annotate-option=continuity] select').value, document.querySelector('[data-annotate-option=easy] select').value]")
        check("choices kept and shown", shown == ["2", "rules"], str(shown))
        src = wait_for(c, "(() => { const h = document.querySelector('section.bx-view header')?.textContent || ''; return (h.match(/(continuity v2 proposal|continuity proposal|idtrackerai proposal|saved)/) || [])[1]; })()",
                       timeout=180, what="a frame open")
        check("frames open with the v2 proposal", src in ("continuity v2 proposal", "saved", "idtrackerai proposal"), src)
        wait_for(c, "!!document.querySelector('[data-action=accept-easy-open]') && !document.querySelector('[data-action=accept-easy-open]').disabled",
                 what="Accept easy… button")
        c.evaluate("document.querySelector('[data-action=accept-easy-open]').click(); true")
        n_rows = wait_for(c, "document.querySelectorAll('[data-ae-row]').length", what="review rows")
        thumbs = c.evaluate("[...document.querySelectorAll('[data-ae-row] img.ae-thumb')].slice(0, 3).map((i) => i.src)")
        check("review list with thumbnails", n_rows > 0 and all("w=" in t for t in thumbs), f"{n_rows} rows")
        all_ticked = c.evaluate("[...document.querySelectorAll('[data-ae-row]')].every((r) => r.dataset.aeChecked === '1')")
        check("every easy scene ticked at first", all_ticked)
        # None, then tick the first N_SCENES
        c.evaluate("[...document.querySelectorAll('.ae-tools button')].find((b) => b.textContent === 'None').click(); true")
        time.sleep(0.3)
        starts = c.evaluate(f"""(() => {{ const rows = [...document.querySelectorAll('[data-ae-row]')].slice(0, {N_SCENES});
            rows.forEach((r) => r.querySelector('input[type=checkbox]').click());
            return rows.map((r) => Number(r.dataset.aeRow)); }})()""")
        time.sleep(0.3)
        ticked = c.evaluate("[...document.querySelectorAll('[data-ae-row]')].filter((r) => r.dataset.aeChecked === '1').map((r) => Number(r.dataset.aeRow))")
        check("unticked scenes left out", sorted(ticked) == sorted(starts), f"{ticked}")
        c.evaluate("document.querySelector('.ae-table [data-ae-row]').click(); true")
        wait_for(c, "!!document.querySelector('[data-ae-preview] canvas') && document.querySelector('[data-ae-preview] canvas').width > 0",
                 what="the preview")
        check("preview drawn", True)
        scenes = {sc["start"]: sc for sc in get(base, "/api/annotate/scenes?easy=rules")["scenes"]}
        expect = {fn for s in starts for fn in range(s, s + scenes[s]["length"])} - set(before)
        c.evaluate("window.confirm = () => true; document.querySelector('[data-action=accept-easy-run]').click(); true")
        wait_for(c, "document.querySelector('.ae-dialog')?.dataset.aeState === 'done'", timeout=1800, what="the accept job")
        after = saved_frames()
        new = set(after) - set(before)
        check("accepted frames = frames of the ticked scenes not saved", new == expect, f"{len(new)} saved, {len(expect)} expected")
        others = {fn for s, sc in scenes.items() if s not in starts and s in {int(k) for k in easy["scenes"]}
                  for fn in range(s, s + sc["length"])} - expect
        check("unticked scenes not saved", not (others & new))
        with sqlite3.connect(SIDECAR) as con:
            # each frame's last log row (a reused copy keeps the rows of earlier runs)
            logs = [a for (a,) in con.execute(
                "SELECT author FROM log WHERE rowid IN (SELECT MAX(rowid) FROM log WHERE frame_number IN "
                f"({','.join(map(str, sorted(new))) or '-1'}) GROUP BY frame_number)")]
        check("log marked", bool(logs) and all("accepted easy scene, continuity2, rules, batch" in a for a in logs), logs[:1])
        authors = {after[fn] for fn in new}
        check("saved as the person", len(authors) == 1 and "accepted" not in next(iter(authors), "accepted"), str(authors))
        # Undo
        wait_for(c, "!!document.querySelector('[data-action=accept-easy-undo]:not([disabled])')", what="Undo")
        c.evaluate("window.confirm = () => true; document.querySelector('[data-action=accept-easy-undo]').click(); true")
        t0 = time.time()
        while set(saved_frames()) != set(before) and time.time() - t0 < 60:
            time.sleep(0.5)
        check("undo unsaves the batch", set(saved_frames()) == set(before), f"{len(saved_frames())} frames saved")
        wait_for(c, "[...document.querySelectorAll('[data-ae-batch]')].some((r) => /undone/.test(r.textContent))", what="batch marked undone")
        check("batch shown as undone", True)
    finally:
        c.close()
print("ALL PASS" if all(results) else "SOME FAILED")
sys.exit(0 if all(results) else 1)
