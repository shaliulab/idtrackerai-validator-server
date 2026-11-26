"""Headless Chrome at phone size: the Data collector's "idtrackerai pipeline" card adds, edits and removes a
row of a scratch copy of idtrackerai.csv, signed in from a non-loopback address (the server's LAN address).

    python tests/browser/idtrackerai_csv.py WORKDIR [HOST]

Everything happens in WORKDIR: HOME (so ~/.cache / ~/.config / the GPU and fetch queues are scratch ones),
$FLYHOSTEL_VIDEOS with one synthetic experiment, and the CSV (a copy of the real one when readable). The
real idtrackerai.csv is only read. Needs the client built into idtrackerai_validator_server/frontend.
Prints PASS / FAIL per step; exit 1 on any failure.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from idtrackerai_validator_server import lab_paths  # noqa: E402
from idtrackerai_validator_server.pe_record import Chrome  # noqa: E402

PHONE = (390, 844)
EXP = "FlyHostel2/6X/2026-11-24_17-00-00"
REAL = os.path.join(lab_paths.NEXTFLOW_PIPELINES, "idtrackerai_v3", "idtrackerai.csv")
USER, PASSWORD = "owner", "test-password"
failures = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def lan_address():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main(workdir, host=None):
    workdir = os.path.abspath(workdir)
    if os.path.realpath(workdir).startswith("/flyhostel_data"):
        sys.exit("WORKDIR must not be under /flyhostel_data")
    os.makedirs(workdir, exist_ok=True)
    home, root = os.path.join(workdir, "home"), os.path.join(workdir, "videos")
    csv = os.path.join(workdir, "idtrackerai.csv")
    for path in (home, root):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
    if os.path.exists(REAL):
        shutil.copyfile(REAL, csv)
    else:
        with open(csv, "w") as fh:
            fh.write("/flyhostel_data/videos/FlyHostel1/3X/2025-06-21_15-00-00,2025-07-04,OK\n")
    with open(csv, "rb") as fh:
        original = fh.read()
    open(os.path.join(root, "index.txt"), "w").close()
    folder = os.path.join(root, EXP)
    os.makedirs(folder)
    date = EXP.split("/")[2]
    for name in ("metadata.yaml", "index.db", "landmarks.toml", "a.pfs", date + ".toml", date + ".conf"):
        open(os.path.join(folder, name), "w").close()
    time.sleep(0.05)
    with open(os.path.join(folder, "preprocess_sample.log"), "w") as fh:
        fh.write("Total problematic frames = 0\n")
    with open(os.path.join(folder, "preprocess_sample.csv"), "w") as fh:
        fh.write("frame_number,chunk,video,frame_idx,n,x,y\n")

    host = host or lan_address()
    port = free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VALIDATOR_", "IDTRACKERAI_CSV"))}
    env.update(HOME=home, FLYHOSTEL_VIDEOS=root, IDTRACKERAI_CSV=csv, BACKEND_PORT=str(port), PYTHONUNBUFFERED="1",
               VALIDATOR_USERNAME=USER, VALIDATOR_PASSWORD=PASSWORD, VALIDATOR_SECRET_KEY="scratch",
               # this checkout's package, not the one installed in the env
               PYTHONPATH=os.path.dirname(os.path.dirname(HERE)))
    code = ("import idtrackerai_validator_server as p, sys; print('package', p.__file__, sys.prefix); "
            "from idtrackerai_validator_server.app import app; "
            f"app.run(host={host!r}, port={port}, threaded=True, debug=False, use_reloader=False)")
    log = open(os.path.join(workdir, "server.log"), "wb")
    server = subprocess.Popen([sys.executable, "-c", code], cwd=workdir, env=env, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True)
    base = f"http://{host}:{port}"
    chrome = None
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/api/auth", timeout=2)
                break
            except OSError:
                if server.poll() is not None:
                    sys.exit("server exited, see server.log")
                time.sleep(0.5)
        chrome = Chrome(workdir, viewport=PHONE)
        chrome.open_page(viewport=PHONE)
        chrome.send("Emulation.setDeviceMetricsOverride", {"width": PHONE[0], "height": PHONE[1],
                                                           "deviceScaleFactor": 2, "mobile": True})
        ev = chrome.evaluate

        def wait(expr, timeout=20):
            deadline = time.time() + timeout
            while time.time() < deadline:
                value = ev(expr)
                if value:
                    return value
                time.sleep(0.2)
            return None

        def shot(name):
            data = chrome.send("Page.captureScreenshot", {"format": "png"})["data"]
            import base64
            with open(os.path.join(workdir, name + ".png"), "wb") as fh:
                fh.write(base64.b64decode(data))

        # sign in through the form (session cookie)
        chrome.send("Page.navigate", {"url": base + "/login"})
        wait("!!document.querySelector('input[type=password]')")
        status = ev(f"""fetch('/login', {{method: 'POST', headers: {{'Content-Type': 'application/x-www-form-urlencoded',
                       'Accept': 'application/json'}}, body: 'username={USER}&password={PASSWORD}&remember=1'}})
                       .then(r => r.status)""", await_promise=True)
        check("login", status == 200, status)
        ev(f"localStorage.setItem('fh-collector-experiment', {EXP!r})")
        chrome.send("Page.navigate", {"url": base + "/"})
        check("app loads after login", bool(wait("!!document.querySelector('.tab-panel')", 40)))
        # the tab button is disabled until /api/features answered: click until the card shows
        card = wait("""(() => { const b = [...document.querySelectorAll('button.tab')].find(x => /Data collector/.test(x.textContent));
                          if (b && !b.disabled && !b.classList.contains('active')) b.click();
                          return !!document.querySelector('[data-ic-card] [data-action=ic-add]'); })()""", 40)
        check("card offers Add for an unlisted experiment", bool(card),
              ev("JSON.stringify([...document.querySelectorAll('button.tab')].map(b => [b.textContent, b.disabled, b.className]))")
              + " | " + str(ev("document.querySelector('.sc-tab')?.textContent?.slice(-600) || document.body.textContent.slice(0, 800)")))
        foot = ev("document.querySelector('[data-ic-card] .ic-foot')?.textContent")
        check("LAN login may edit", bool(foot) and "you may change it" in foot and "login owner" in foot, foot)
        ev("document.querySelector('[data-ic-card]').scrollIntoView()")
        shot("1-card")

        ev("document.querySelector('[data-action=ic-add]').click()")
        diff = wait("document.querySelector('[data-ic-diff]')?.textContent")
        check("preview shows the new line", bool(diff) and f"{root}/{EXP},1,OK" in diff, diff)
        rect = ev("JSON.stringify(document.querySelector('[data-ic-dialog]').getBoundingClientRect())")
        r = json.loads(rect)
        check("dialog fills the phone screen", r["width"] <= PHONE[0] + 1 and r["width"] >= PHONE[0] - 2, rect)
        overflow = ev("document.documentElement.scrollWidth <= window.innerWidth + 1")
        check("no sideways page scroll", bool(overflow))
        shot("2-add-dialog")
        check("CSV untouched by the preview", open(csv, "rb").read() == original)
        ev("document.querySelector('[data-action=ic-confirm]').click()")
        row = wait("document.querySelector('[data-ic-row]')?.getAttribute('data-ic-row')")
        data = open(csv, "rb").read()
        expected_tail = f"{root}/{EXP},1,OK\n".encode()
        check("row appended, the rest byte-identical",
              data.endswith(expected_tail) and data[:len(original)].rstrip(b"\n") == original.rstrip(b"\n")
              and len(data) == len(original.rstrip(b"\n")) + 1 + len(expected_tail), row)
        shot("3-listed")

        # edit: mark NEXT
        ev("document.querySelector('[data-action=ic-edit]').click()")
        wait("!!document.querySelector('[data-ic-mark]')")
        ev("""(() => { const s = document.querySelector('[data-ic-mark]');
                 const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;
                 set.call(s, 'NEXT'); s.dispatchEvent(new Event('change', {bubbles: true})); })()""")
        diff = wait(f"(t => t && t.includes(',NEXT,OK') ? t : null)(document.querySelector('[data-ic-diff]')?.textContent)")
        check("edit preview shows old and new line", bool(diff) and ",1,OK" in diff, diff)
        shot("4-edit-dialog")
        ev("document.querySelector('[data-action=ic-confirm]').click()")
        wait("!document.querySelector('[data-ic-dialog]')")
        check("row now marked NEXT", open(csv, "rb").read().endswith(f"{root}/{EXP},NEXT,OK\n".encode()))

        # someone edits the file meanwhile: the stale preview is refused
        ev("document.querySelector('[data-action=ic-remove]').click()")
        wait("!!document.querySelector('[data-ic-diff]')")
        with open(csv, "ab") as fh:
            fh.write(b"/elsewhere,LATER\n")
        ev("document.querySelector('[data-action=ic-confirm]').click()")
        err = wait("document.querySelector('[data-ic-error]')?.textContent")
        check("stale preview refused", bool(err) and "changed" in err, err)
        ev("document.querySelector('.ov-x').click()")
        ev("document.querySelector('[data-ic-card] .icon-btn').click()")
        time.sleep(1)
        ev("document.querySelector('[data-action=ic-remove]').click()")
        wait("!!document.querySelector('[data-ic-diff]')")
        time.sleep(0.5)
        ev("document.querySelector('[data-action=ic-confirm]').click()")
        wait("!!document.querySelector('[data-action=ic-add]')")
        check("row removed, the other change kept",
              open(csv, "rb").read() == original.rstrip(b"\n") + b"\n/elsewhere,LATER\n" if original.strip() else True)
        backups = os.listdir(os.path.join(home, ".cache", "idtrackerai-validator", "idtrackerai_csv_backups"))
        with open(os.path.join(home, ".cache", "idtrackerai-validator", "idtrackerai_csv.jsonl")) as fh:
            entries = [json.loads(line) for line in fh]
        check("3 backups and 3 audit lines by owner", len(backups) == 3 and [e["user"] for e in entries] == [USER] * 3,
              f"{len(backups)} {[e['user'] for e in entries]}")
    finally:
        if chrome:
            chrome.close()
        server.terminate()
        server.wait(10)
    print("FAILED: " + ", ".join(failures) if failures else "ALL PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:3]))
