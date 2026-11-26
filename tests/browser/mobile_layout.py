"""Every tab of the web UI at phone, tablet and desktop sizes in headless Chrome (device emulation: touch, mobile
user agent, device pixel ratio), with screenshots, failing on horizontal page overflow or JavaScript errors.

    VALIDATOR_USERNAME=... VALIDATOR_PASSWORD=... \\
    python tests/browser/mobile_layout.py BASE_URL OUTDIR [--devices iphone14,pixel7,...] [--tabs qc,annotate,...]
                                          [--no-shots] [--strict-taps]

BASE_URL: a running server (https with a self-signed certificate is fine), e.g. a scratch server on a COPY of an
experiment: the script only reads, but opening tabs starts their usual jobs (QC, easy scenes). The login is used when
the server asks for one. OUTDIR gets `<device>/<NN>_<view>.png` (the viewport), `<NN>_<view>_full.jpg` (the whole page when longer)
and `report.json`.

Per device and view, printed as PASS / FAIL lines:
- the page does not scroll sideways (documentElement.scrollWidth <= innerWidth) and no visible element sticks out of
  the window unless an ancestor scrolls or clips it (the first offenders are listed);
- no uncaught exception / console error was raised while it was shown;
- tap targets (touch devices): visible buttons, links, inputs and selects smaller than 24 CSS px in either direction
  are listed (a FAIL only with --strict-taps), and the count below 44 px is reported.
Exit status 1 when any check failed."""
import argparse
import base64
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from idtrackerai_validator_server.pe_record import Chrome  # noqa: E402

MOBILE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
             "Version/17.0 Mobile/15E148 Safari/604.1")
ANDROID_UA = ("Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Mobile Safari/537.36")
IPAD_UA = ("Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
           "Version/17.0 Mobile/15E148 Safari/604.1")
DEVICES = {
    # name: (width, height, device pixel ratio, mobile + touch, user agent)
    "iphone14": (390, 844, 3, True, MOBILE_UA),
    "pixel7": (412, 915, 2.625, True, ANDROID_UA),
    "ipad_portrait": (820, 1180, 2, True, IPAD_UA),
    "ipad_landscape": (1180, 820, 2, True, IPAD_UA),
    "desktop": (1440, 900, 1, False, None),
}

# view name, tab id (the Tab button's id is not in the DOM: matched by its label), and an optional action run
# after the tab is shown (JS returning when done)
VIEWS = [
    ("data_collector", "Data collector", None),
    ("group_viewer", "Group viewer", None),
    ("pe", "Proboscis extension", None),
    ("leg_movements", "Leg movements", None),
    ("ethograms", "Ethograms", None),
    ("ethoscope", "Ethoscope", None),
    ("qc", "QC", None),
    ("qc_swaps", "QC", "document.querySelector('.qc-swap, [class*=qc-swap]')?.scrollIntoView({block: 'start'})"),
    ("annotate", "Annotate", None),
    ("annotate_frame", "Annotate", "nav:annotate"),
    ("annotate_predownload", "Annotate",
     "(() => { const c = document.querySelector('.bx-pre'); if (!c) return 'no card';"
     " if (!c.querySelector('.bx-pre-body')) c.querySelector('.card-header').click();"
     " c.scrollIntoView({block: 'start'}); return 'ok'; })()"),
    ("accept_easy", "Annotate",
     "(() => { const b = document.querySelector('[data-action=accept-easy-open]');"
     " if (!b) return 'no button'; b.click(); return 'ok'; })()"),
    ("drive_exchange", "Drive exchange", None),
    ("pose", "Pose", None),
    ("sleap_labels", "SLEAP labels", None),
    ("overview", "Metadata overview", None),
    ("data_management", "Data management", None),
    ("monitor", "Monitor", None),
    ("help", "How to use", None),
    ("help_updates", "How to use", "document.getElementById('help-updates')?.scrollIntoView({block: 'start'})"),
]

CLICK_TAB_JS = """((label) => {
  const tabs = [...document.querySelectorAll('nav.tabs button.tab, nav.tabs [data-tab]')];
  const t = tabs.find((b) => b.textContent.trim().startsWith(label));
  if (!t) return 'no tab ' + label;
  if (t.disabled) return 'disabled';
  t.click();
  return 'ok';
})(%s)"""

CLOSE_OVERLAYS_JS = """(() => {
  for (const b of document.querySelectorAll('.ae-overlay [data-action=accept-easy-close], .ae-overlay .ae-close'))
    b.click();
  document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
  window.scrollTo(0, 0);
})()"""

MEASURE_JS = r"""((deviceWidth) => {
  const de = document.documentElement;
  // the layout viewport: a phone zooms out over content wider than the screen (innerWidth then grows)
  const W = Math.min(de.clientWidth, deviceWidth), H = window.innerHeight;
  const clips = (el) => {
    for (let p = el.parentElement; p && p !== document.body && p !== de; p = p.parentElement) {
      const s = getComputedStyle(p);
      if (/(auto|scroll|hidden|clip)/.test(s.overflowX)) {
        const r = p.getBoundingClientRect();
        if (r.right <= W + 1) return true;
      }
      if (s.position === 'fixed') return false;
    }
    return false;
  };
  const visible = (el, r) => {
    if (r.width === 0 || r.height === 0) return false;
    const s = getComputedStyle(el);
    return s.visibility !== 'hidden' && s.display !== 'none' && +s.opacity !== 0;
  };
  const describe = (el) => {
    let d = el.tagName.toLowerCase();
    if (el.id) d += '#' + el.id;
    if (el.classList.length) d += '.' + [...el.classList].slice(0, 3).join('.');
    const t = (el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 40);
    return t ? d + ' "' + t + '"' : d;
  };
  const offenders = [];
  for (const el of document.body.querySelectorAll('*')) {
    const r = el.getBoundingClientRect();
    if (r.right > W + 1 && r.left < W && visible(el, r) && !clips(el)) {
      // report the outermost offender only
      const p = el.parentElement;
      const pr = p && p.getBoundingClientRect();
      if (p && pr.right > W + 1 && !clips(p) && p !== document.body) continue;
      offenders.push({el: describe(el), right: Math.round(r.right), width: Math.round(r.width)});
      if (offenders.length >= 12) break;
    }
  }
  const small = [], under44 = [];
  for (const el of document.querySelectorAll('button, a[href], input:not([type=hidden]), select, [role=button], summary')) {
    const r = el.getBoundingClientRect();
    if (!visible(el, r) || r.bottom < 0 || r.top > H || r.right < 0 || r.left > W) continue;
    if (el.closest('.tabs') && el.offsetParent === null) continue;
    let w = r.width, h = r.height;
    // a checkbox / radio inside a label: the label is the target
    if ((el.type === 'checkbox' || el.type === 'radio') && el.closest('label')) {
      const lr = el.closest('label').getBoundingClientRect(); w = lr.width; h = lr.height;
    }
    const m = Math.min(w, h);
    if (m < 44) under44.push(describe(el));
    if (m < 24) small.push({el: describe(el), w: Math.round(w), h: Math.round(h)});
  }
  return {
    width: W, height: H, scrollWidth: de.scrollWidth, scrollHeight: de.scrollHeight,
    // a desktop scrollbar (< 20 px) is not overflow; a phone zoomed out over wide content is
    overflow: Math.max(de.scrollWidth - W, window.innerWidth - W - 20), offenders, small: small.slice(0, 15), n_small: small.length,
    n_under44: under44.length,
  };
})(%d)"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base")
    ap.add_argument("outdir")
    ap.add_argument("--devices", default=",".join(DEVICES))
    ap.add_argument("--tabs", default="")
    ap.add_argument("--experiment", help="flat name: each device opens it (link of links.py) at --frame in the Group viewer")
    ap.add_argument("--frame", type=int, help="Group viewer frame")
    ap.add_argument("--annotate-frame", type=int, help="frame of a scene: the view annotate_frame opens it in Annotate")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="localStorage entry set before the views (e.g. fh-collector-experiment=\"FlyHostel1/3X/...\")")
    ap.add_argument("--no-shots", action="store_true")
    ap.add_argument("--strict-taps", action="store_true")
    ap.add_argument("--wait", type=float, default=2.5, help="seconds a view is given to load (default 2.5)")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    wanted = [t for t in args.tabs.split(",") if t]
    views = [v for v in VIEWS if not wanted or v[0] in wanted]
    os.makedirs(args.outdir, exist_ok=True)
    report, failures = {}, 0

    def result(name, ok, detail=""):
        nonlocal failures
        failures += 0 if ok else 1
        print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)

    work = tempfile.mkdtemp(prefix="mobile-layout-")
    chrome = Chrome(work)
    try:
        chrome.open_page()
        chrome.send("Security.setIgnoreCertificateErrors", {"ignore": True})
        chrome.send("Log.enable")
        for dev in args.devices.split(","):
            w, h, dpr, mobile, ua = DEVICES[dev]
            chrome.send("Emulation.setDeviceMetricsOverride",
                        {"width": w, "height": h, "deviceScaleFactor": dpr, "mobile": mobile,
                         "screenWidth": w, "screenHeight": h})
            chrome.send("Emulation.setTouchEmulationEnabled", {"enabled": mobile, "maxTouchPoints": 5 if mobile else 1})
            chrome.send("Emulation.setEmitTouchEventsForMouse", {"enabled": mobile, "configuration": "mobile"})
            chrome.send("Emulation.setUserAgentOverride", {"userAgent": ua or "", "platform": ""} if ua
                        else {"userAgent": chrome.evaluate("navigator.userAgent").replace("HeadlessChrome", "Chrome")})
            navigate(chrome, base + "/")
            login(chrome, base)
            wait_app(chrome)
            # each device starts from a fresh browser state (the previous device's choices, e.g. a folded
            # console, would otherwise carry over), plus the --set entries
            chrome.evaluate("localStorage.clear(); sessionStorage.clear()")
            for kv in args.set:
                k, v = kv.split("=", 1)
                chrome.evaluate("localStorage.setItem(%s, %s)" % (json.dumps(k), json.dumps(v)))
            navigate(chrome, base + "/")
            wait_app(chrome)
            if args.experiment and args.frame is not None:
                navigate(chrome, f"{base}/?tab=group_viewer&experiment={args.experiment}&frame={args.frame}")
                wait_app(chrome)
                time.sleep(3)
            outdir = os.path.join(args.outdir, dev)
            os.makedirs(outdir, exist_ok=True)
            report[dev] = {}
            for view, label, action in views:
                i = [v[0] for v in VIEWS].index(view)     # file numbers stay those of the full list
                chrome.evaluate(CLOSE_OVERLAYS_JS)
                chrome.events.clear()
                if action == "nav:annotate":
                    if not (args.experiment and args.annotate_frame is not None):
                        print(f"SKIP {dev} {view}: no --experiment / --annotate-frame", flush=True)
                        continue
                    navigate(chrome, f"{base}/?tab=annotate&experiment={args.experiment}&frame={args.annotate_frame}")
                    wait_app(chrome)
                    action = None
                    time.sleep(4)
                state = chrome.evaluate(CLICK_TAB_JS % json.dumps(label))
                if state != "ok":
                    print(f"SKIP {dev} {view}: {state}", flush=True)
                    report[dev][view] = {"skipped": state}
                    continue
                time.sleep(args.wait)
                if action:
                    chrome.evaluate(action)
                    time.sleep(args.wait)
                m = chrome.evaluate(MEASURE_JS % w)
                errors = page_errors(chrome)
                m["errors"] = errors
                report[dev][view] = m
                name = f"{dev} {view}"
                result(name + " no sideways scroll", m["overflow"] <= 0 and not m["offenders"],
                       f"page {m['scrollWidth']} > {m['width']} px; " * (m["overflow"] > 0)
                       + "; ".join(f"{o['el']} right {o['right']}" for o in m["offenders"][:5]))
                result(name + " no JS errors", not errors, "; ".join(errors[:3]))
                if mobile and m["n_small"]:
                    if args.strict_taps:
                        result(name + " tap targets >= 24 px", False,
                               "; ".join(f"{s['el']} {s['w']}x{s['h']}" for s in m["small"][:5]))
                    else:
                        print(f"NOTE {name}: {m['n_small']} tap targets < 24 px, {m['n_under44']} < 44 px", flush=True)
                if not args.no_shots:
                    shot = chrome.send("Page.captureScreenshot", {"format": "png"})
                    with open(os.path.join(outdir, f"{i:02d}_{view}.png"), "wb") as f:
                        f.write(base64.b64decode(shot["data"]))
                    # the whole page too, at 1 CSS px per image px (phones' pages are long)
                    full_h = min(int(m["scrollHeight"]), 6000)
                    if full_h > h + 20:
                        shot = chrome.send("Page.captureScreenshot", {
                            "format": "jpeg", "quality": 70, "captureBeyondViewport": True,
                            "clip": {"x": 0, "y": 0, "width": w, "height": full_h, "scale": 1 / dpr}})
                        with open(os.path.join(outdir, f"{i:02d}_{view}_full.jpg"), "wb") as f:
                            f.write(base64.b64decode(shot["data"]))
            chrome.evaluate(CLOSE_OVERLAYS_JS)
    finally:
        chrome.close()
    with open(os.path.join(args.outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"{failures} failed" if failures else "all passed", flush=True)
    sys.exit(1 if failures else 0)


def navigate(chrome, url):
    # a marker on the old document: "complete" only counts once it is gone (the new document loaded)
    chrome.evaluate("window.__oldDocument = true")
    chrome.send("Page.navigate", {"url": url})
    t0 = time.time()
    while time.time() - t0 < 30:
        if chrome.evaluate("!window.__oldDocument && document.readyState === 'complete'"):
            return
        time.sleep(0.2)


def login(chrome, base):
    if "/login" not in (chrome.evaluate("location.pathname") or ""):
        return
    user, password = os.environ.get("VALIDATOR_USERNAME"), os.environ.get("VALIDATOR_PASSWORD")
    if not (user and password):
        sys.exit("the server asks for a login: set VALIDATOR_USERNAME and VALIDATOR_PASSWORD")
    ok = chrome.evaluate(
        "fetch('/login', {method: 'POST', credentials: 'same-origin', redirect: 'manual',"
        " body: new URLSearchParams({username: %s, password: %s, remember: 'on', next: '/'})})"
        ".then((r) => r.type === 'opaqueredirect' || r.ok)" % (json.dumps(user), json.dumps(password)),
        await_promise=True)
    if not ok:
        sys.exit("login refused")
    navigate(chrome, base + "/")


def wait_app(chrome, timeout=60):
    # one reload when the first load never renders (seen once in headless Chrome under load)
    for attempt in (1, 2):
        t0 = time.time()
        while time.time() - t0 < timeout / 2:
            if chrome.evaluate("document.querySelectorAll('nav.tabs button').length > 5"):
                time.sleep(2)
                return
            time.sleep(0.3)
        if attempt == 1:
            navigate(chrome, chrome.evaluate("location.href"))
    raise TimeoutError("the app did not render its tabs")


def page_errors(chrome):
    # drain what is waiting on the pipe, then keep exceptions and console errors
    try:
        while True:
            chrome.events.append(chrome._read_message(0.05))
    except TimeoutError:
        pass
    out = []
    for ev in chrome.events:
        method, p = ev.get("method"), ev.get("params", {})
        if method == "Runtime.exceptionThrown":
            d = p.get("exceptionDetails", {})
            out.append("exception: " + (d.get("exception", {}).get("description") or d.get("text", ""))[:300])
        elif method == "Runtime.consoleAPICalled" and p.get("type") == "error":
            text = " ".join(str(a.get("value", a.get("description", ""))) for a in p.get("args", []))
            # failed requests the app reports itself (404 of optional data, aborted loads) are not page errors
            if "Request failed with status code" in text or "AxiosError" in text or "canceled" in text:
                continue
            out.append("console.error: " + text[:300])
    chrome.events.clear()
    return out


if __name__ == "__main__":
    main()
