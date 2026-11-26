"""Touch gestures in headless Chrome with a phone's touch screen (CDP Input.dispatchTouchEvent → pointer events of
type touch), printing PASS / FAIL lines; exit status 1 when one failed.

    VALIDATOR_USERNAME=... VALIDATOR_PASSWORD=... \\
    python tests/browser/touch_gestures.py BASE_URL EXPERIMENT FRAME ANNOTATE_FRAME [--device iphone14]

BASE_URL: a running server on a SCRATCH COPY (refused when the loaded experiment's folder resolves under
/flyhostel_data): nothing is saved (the edits of the Annotate frame are undone and the page closed without leaving
the frame), but a mistake would write the copy's validator_annotations.db. EXPERIMENT: flat name; FRAME: a Group
viewer frame; ANNOTATE_FRAME: a frame of a scene. Checks:
Group viewer  1. a swipe to the left steps frames forward (the Frame field grows), the page does not scroll sideways;
              2. a two-finger pinch zooms the frame (its "×" reset button appears), a double tap goes back;
Annotate      3. New shape, then a one-finger drag draws a box (one more row in the boxes table);
              4. a one-finger drag inside the new box moves it (its x changes), nothing else is added;
              5. a two-finger pinch adds / moves nothing;
              6. the touch bar's label button labels the selected box, its delete removes it, its undo brings it back,
                 undo again removes it.
Data collector (Segmentation settings on the experiment's folder; nothing saved)
              7. + Add → Polygon, three taps add three points, a long press removes the nearest, a pinch zooms
                 without adding points, Cancel leaves the toml unchanged."""
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idtrackerai_validator_server.pe_record import Chrome  # noqa: E402
from mobile_layout import DEVICES, login, navigate, wait_app  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)


def touch(c, kind, points):
    c.send("Input.dispatchTouchEvent", {"type": kind, "touchPoints": [{"x": x, "y": y, "id": i} for i, (x, y) in points]})


def drag(c, a, b, steps=12, dt=0.02):
    touch(c, "touchStart", [(0, a)])
    for k in range(1, steps + 1):
        t = k / steps
        touch(c, "touchMove", [(0, (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))])
        time.sleep(dt)
    touch(c, "touchEnd", [])
    time.sleep(0.3)


def pinch(c, center, d0, d1, steps=10):
    cx, cy = center

    def pts(d):
        return [(0, (cx - d / 2, cy)), (1, (cx + d / 2, cy))]
    touch(c, "touchStart", pts(d0)[:1])
    touch(c, "touchStart", pts(d0))
    for k in range(1, steps + 1):
        touch(c, "touchMove", pts(d0 + (d1 - d0) * k / steps))
        time.sleep(0.02)
    # lift the fingers one at a time (an end with both at once left a touch point stuck in Chrome's emulator)
    touch(c, "touchEnd", pts(d1)[:1])
    touch(c, "touchEnd", [])
    time.sleep(0.3)


def tap(c, p):
    touch(c, "touchStart", [(0, p)])
    time.sleep(0.05)
    touch(c, "touchEnd", [])
    time.sleep(0.12)


def rect(c, selector):
    # the element's position once the page stops moving (cards above it load late and push it down)
    last = None
    for _ in range(40):
        now = _rect(c, selector)
        if now and now == last:
            # the compositor hit-tests touches: let it draw the scrolled page first
            c.evaluate("new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))", await_promise=True)
            time.sleep(0.2)
            return now
        last = now
        time.sleep(0.3)
    return last


def _rect(c, selector):
    return c.evaluate("(() => { const e = document.querySelector(%s); if (!e) return null; e.scrollIntoView({block: 'center'});"
                      " const r = e.getBoundingClientRect(); return {x: r.left, y: r.top, w: r.width, h: r.height}; })()"
                      % json.dumps(selector))


def wait(c, js, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = c.evaluate(js)
        if v:
            return v
        time.sleep(0.25)
    return None


BOXES_JS = """[...(document.querySelector('.bx-view')?.parentElement.querySelectorAll('.sl-side .sl-table tbody tr') || [])]
  .map((tr) => [...tr.children].map((td) => td.textContent.trim()))"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("experiment")
    ap.add_argument("frame", type=int)
    ap.add_argument("annotate_frame", type=int)
    ap.add_argument("--device", default="iphone14")
    ap.add_argument("--only", choices=["group_viewer", "annotate", "segmentation"])
    ap.add_argument("--collector-experiment", help="folder (slash form) opened in the Data collector's Segmentation "
                    "settings (default: EXPERIMENT in slash form); its toml is never saved")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    w, h, dpr, _, ua = DEVICES[args.device]
    # one browser per section: after a pinch, a page loaded later in the same emulated browser got its touches
    # hit-tested as page scrolls (headless Chrome's emulator only; the app does not reload between tabs)
    for name, section in (("group_viewer", group_viewer), ("annotate", annotate), ("segmentation", segmentation)):
        if args.only not in (None, name):
            continue
        c = Chrome(tempfile.mkdtemp(prefix="touch-"))
        try:
            c.open_page()
            c.send("Security.setIgnoreCertificateErrors", {"ignore": True})
            c.send("Emulation.setDeviceMetricsOverride", {"width": w, "height": h, "deviceScaleFactor": dpr, "mobile": True})
            c.send("Emulation.setTouchEmulationEnabled", {"enabled": True, "maxTouchPoints": 5})
            c.send("Emulation.setUserAgentOverride", {"userAgent": ua})
            navigate(c, base + "/")
            login(c, base)
            wait_app(c)
            info = c.evaluate("fetch('/api/current_experiment').then((r) => r.json())", await_promise=True)
            if os.path.realpath(info.get("realpath") or info.get("basedir") or "/flyhostel_data").startswith("/flyhostel_data"):
                sys.exit("refused: the server's experiment is under /flyhostel_data (use a scratch copy)")
            section(c, base, args)
        finally:
            c.close()
    print("all passed" if all(results) else f"{results.count(False)} failed", flush=True)
    sys.exit(0 if all(results) else 1)


def group_viewer(c, base, args):
    navigate(c, f"{base}/?tab=group_viewer&experiment={args.experiment}&frame={args.frame}")
    wait_app(c)
    frame_js = "Number(document.querySelector('.viewer-nav .frame-jump input')?.value)"
    wait(c, f"{frame_js} === {args.frame}")
    time.sleep(2)
    r = rect(c, ".viewer-frame canvas")
    mid = (r["x"] + r["w"] / 2, r["y"] + r["h"] / 2)
    before = c.evaluate(frame_js)
    drag(c, (mid[0] + r["w"] * 0.35, mid[1]), (mid[0] - r["w"] * 0.35, mid[1]), steps=20)
    after = wait(c, f"{frame_js} > {before} && {frame_js}", 10)
    check("group viewer: swipe left steps forward", bool(after), f"frame {before} -> {c.evaluate(frame_js)}")
    check("group viewer: no sideways page scroll after the swipe",
          c.evaluate("document.documentElement.scrollLeft === 0 && window.scrollX === 0"))
    r = rect(c, ".viewer-frame canvas")
    mid = (r["x"] + r["w"] / 2, r["y"] + r["h"] / 2)
    pinch(c, mid, 60, 220)
    zoomed = c.evaluate("document.querySelector('.frame-zoom-reset')?.textContent || ''")
    check("group viewer: pinch zooms", bool(zoomed), zoomed)
    tap(c, mid)
    tap(c, mid)
    time.sleep(0.4)
    check("group viewer: double tap returns to the whole frame", not c.evaluate("!!document.querySelector('.frame-zoom-reset')"))


def annotate(c, base, args):
    navigate(c, f"{base}/?tab=annotate&experiment={args.experiment}&frame={args.annotate_frame}")
    wait_app(c)
    ok = wait(c, f"document.querySelector('.bx-view')?.dataset.frame === '{args.annotate_frame}'"
                 " && document.querySelector('.bx-view').dataset.ready === '1'", 60)
    check("annotate: the frame opens", bool(ok))
    time.sleep(1.5)
    boxes0 = c.evaluate(BOXES_JS)
    c.evaluate("[...document.querySelectorAll('.sl-actions button')].find((b) => b.textContent.startsWith('New shape')).click()")
    time.sleep(0.3)
    r = rect(c, ".bx-stage canvas")
    # a corner of the frame, away from the flies
    a = (r["x"] + r["w"] * 0.08, r["y"] + r["h"] * 0.10)
    b = (r["x"] + r["w"] * 0.22, r["y"] + r["h"] * 0.24)
    drag(c, a, b)
    wait(c, f"({BOXES_JS}).length !== {len(boxes0)}", 3)
    boxes1 = c.evaluate(BOXES_JS)
    check("annotate: touch drag after New shape draws a box", len(boxes1) == len(boxes0) + 1,
          f"{len(boxes0)} -> {len(boxes1)} boxes")
    new = [row for row in boxes1 if row not in boxes0]
    x_before = new[0][3] if new else None
    r2 = rect(c, ".bx-stage canvas")
    shift = (r2["x"] - r["x"], r2["y"] - r["y"])
    centre = ((a[0] + b[0]) / 2 + shift[0], (a[1] + b[1]) / 2 + shift[1])
    r = r2
    drag(c, centre, (centre[0] + r["w"] * 0.12, centre[1] + r["h"] * 0.05))
    boxes2 = c.evaluate(BOXES_JS)
    moved = [row for row in boxes2 if row not in boxes0]
    check("annotate: one-finger drag inside the box moves it",
          len(boxes2) == len(boxes1) and bool(moved) and moved[0][3] != x_before,
          f"x {x_before} -> {moved[0][3] if moved else None}")
    pinch(c, (r["x"] + r["w"] / 2, r["y"] + r["h"] / 2), 80, 200)
    boxes3 = c.evaluate(BOXES_JS)
    check("annotate: a pinch adds or moves nothing", boxes3 == boxes2)
    # the new box is selected (drawn last): label it 1 from the touch bar, delete, undo
    visible_bar = c.evaluate("getComputedStyle(document.querySelector('.bx-touchbar')).display !== 'none'")
    check("annotate: the touch bar is shown on a touch screen", visible_bar)
    c.evaluate("[...document.querySelectorAll('.bx-touchbar .bx-label')].find((b) => b.textContent === '?').click()")
    time.sleep(0.3)
    c.evaluate("document.querySelector('.bx-touchbar .fa-trash').closest('button').click()")
    time.sleep(0.3)
    boxes4 = c.evaluate(BOXES_JS)
    check("annotate: touch bar delete removes the selected box", len(boxes4) == len(boxes0), f"{len(boxes4)} boxes")
    c.evaluate("document.querySelector('.bx-touchbar .fa-undo').closest('button').click()")
    time.sleep(0.3)
    boxes5 = c.evaluate(BOXES_JS)
    check("annotate: touch bar undo brings it back", len(boxes5) == len(boxes0) + 1, f"{len(boxes5)} boxes")
    # back to the frame as it opened: undo until the drawn box is gone
    for _ in range(6):
        if c.evaluate(BOXES_JS) == boxes0:
            break
        c.evaluate("document.querySelector('.bx-touchbar .fa-undo').closest('button').click()")
        time.sleep(0.3)
    check("annotate: undo leaves the frame as it opened", c.evaluate(BOXES_JS) == boxes0)


def segmentation(c, base, args):
    exp = args.collector_experiment or "/".join(args.experiment.split("_", 2)[:2]) + "/" + args.experiment.split("_", 2)[2]
    c.evaluate("localStorage.setItem('fh-collector-experiment', %s)" % json.dumps(exp))
    navigate(c, base + "/")
    wait_app(c)
    c.evaluate("[...document.querySelectorAll('nav.tabs button.tab')].find((b) => b.textContent.startsWith('Data collector')).click()")
    ok = wait(c, "(() => { const cv = document.querySelector('.sc-view canvas'); return cv && cv.width > 0; })()", 60)
    check("segmentation: the editor's frame shows", bool(ok))
    time.sleep(3)
    add = c.evaluate("(() => { const b = [...document.querySelectorAll('.sc-tab button')].find((b) => b.textContent.trim() === '+ Add');"
                     " if (!b || b.disabled) return false; b.click(); return true; })()")
    time.sleep(0.3)
    picked = add and c.evaluate("(() => { const b = [...document.querySelectorAll('.sc-roi-types button')].find((b) => /Polygon/.test(b.textContent) && !b.disabled);"
                                " if (!b) return false; b.click(); return true; })()")
    check("segmentation: + Add → Polygon starts drawing", bool(picked))
    if not picked:
        return
    pts_js = "Number((document.querySelector('.sc-tab .sc-row .sc-muted')?.textContent.match(/(\\d+) pts/) || [])[1])"
    r = rect(c, ".sc-view canvas")
    cx, cy = r["x"] + r["w"] / 2, r["y"] + r["h"] / 2
    for dx, dy in ((-60, -40), (60, -40), (0, 50)):
        tap(c, (cx + dx, cy + dy))
        time.sleep(0.4)     # not a double tap
    n = c.evaluate(pts_js)
    check("segmentation: three taps add three points", n == 3, f"{n} pts")
    touch(c, "touchStart", [(0, (cx, cy + 50))])
    time.sleep(0.8)
    touch(c, "touchEnd", [])
    time.sleep(0.3)
    n = c.evaluate(pts_js)
    check("segmentation: a long press removes the nearest point", n == 2, f"{n} pts")
    before = c.evaluate("document.querySelector('.sc-view canvas').toDataURL().length")
    pinch(c, (cx, cy), 60, 200)
    n2 = c.evaluate(pts_js)
    after = c.evaluate("document.querySelector('.sc-view canvas').toDataURL().length")
    check("segmentation: a pinch zooms without adding points", n2 == 2 and after != before, f"{n2} pts")
    c.evaluate("[...document.querySelectorAll('.sc-tab button')].find((b) => b.textContent.trim() === 'Cancel')?.click()")
    time.sleep(0.3)
    check("segmentation: Cancel leaves nothing to save",
          not c.evaluate("!!document.querySelector('.sc-tab .sc-dirty')"))


if __name__ == "__main__":
    main()
