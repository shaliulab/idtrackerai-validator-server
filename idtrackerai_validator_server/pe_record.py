"""
pe_record.py  —  the PE tab's "video" button, pressed by a headless Chrome.

The button (client `recordBurst.js`) records the media card of the PE tab (clip, pose
on white, distance and confidence trace SVGs with the playhead) into a 560x560 .webm
at 30 fps while the burst plays once. To get exactly that video without a person at
the browser:

1. a private server instance (`app.py` with `VALIDATOR_EXPERIMENT` = the experiment,
   127.0.0.1, a free port, no login / TLS, cwd a temporary folder) so the running
   server's loaded experiment is never switched;
2. Google Chrome headless (`$PE_RECORD_CHROME`, default google-chrome), driven over the
   DevTools protocol on a pipe (`--remote-debugging-pipe`, no extra packages), opens
   the PE tab link `/?tab=proboscis_extension&experiment=&fly=&burst=` (App.js),
   waits until the media card shows that burst (data attributes set by
   PEValidator.js / BurstVideo.js: trace, segments, pose loaded, video decodable),
   clicks the button and waits for its download;
3. ffmpeg turns the .webm into an mp4 (libx264, yuv420p, constant `fps`, faststart),
   same size and frame rate.

`record_bursts(experiment, [(fly, burst_id, output), ...])` reuses one server and one
browser for several bursts of an experiment.
"""
import contextlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

from idtrackerai_validator_server.oscompat import IS_WINDOWS, new_session

logger = logging.getLogger(__name__)

CHROME = os.environ.get("PE_RECORD_CHROME", "google-chrome")
VIEWPORT = (1920, 1080)        # the page's window: the trace SVGs are drawn at their on-screen size
SERVER_START_TIMEOUT = 300     # s; the experiment is opened at import
READY_TIMEOUT = 120            # s for the tab to show the burst
POSE_GRACE = 20                # s more for the pose once the rest is ready (bursts without .pose.json)
RECORD_FPS = 30                # recordBurst.js: canvas.captureStream(30)

_lock = threading.Lock()       # one browser recording at a time


def _ffmpeg():
    local = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    return local if os.path.exists(local) else (shutil.which("ffmpeg") or "ffmpeg")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ----------------------------------------------------------------------------------
# private server instance
# ----------------------------------------------------------------------------------

@contextlib.contextmanager
def private_server(experiment, workdir):
    """app.py serving `experiment` on 127.0.0.1:<free port>; yields its base URL"""
    port = _free_port()
    env = {k: v for k, v in os.environ.items()
           if k not in ("VALIDATOR_USERNAME", "VALIDATOR_PASSWORD", "VALIDATOR_CERTFILE", "VALIDATOR_KEYFILE")}
    env.update(VALIDATOR_EXPERIMENT=experiment, BACKEND_PORT=str(port), PYTHONUNBUFFERED="1")
    code = ("from idtrackerai_validator_server.app import app; "
            f"app.run(host='127.0.0.1', port={port}, threaded=True, debug=False, use_reloader=False)")
    log_path = os.path.join(workdir, "server.log")
    with open(log_path, "wb") as log:
        proc = subprocess.Popen([sys.executable, "-c", code], cwd=workdir, env=env,
                                stdout=log, stderr=subprocess.STDOUT, **new_session())
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + SERVER_START_TIMEOUT
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"render server exited ({proc.returncode}): {_tail(log_path)}")
            try:
                with urllib.request.urlopen(base + "/api/current_experiment", timeout=5) as r:
                    info = json.loads(r.read())
                if info.get("experiment"):
                    break
            except OSError:
                pass
            if time.time() > deadline:
                raise RuntimeError(f"render server not up after {SERVER_START_TIMEOUT} s: {_tail(log_path)}")
            time.sleep(0.5)
        logger.info("render server for %s at %s (%.1f s)", experiment, base, SERVER_START_TIMEOUT - (deadline - time.time()))
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _tail(path, n=1500):
    try:
        with open(path, "rb") as fh:
            return fh.read()[-n:].decode(errors="replace")
    except OSError:
        return ""


# ----------------------------------------------------------------------------------
# headless Chrome over the DevTools protocol (pipe transport: NUL-terminated JSON)
# ----------------------------------------------------------------------------------

class Chrome:

    def __init__(self, workdir, viewport=VIEWPORT):
        if IS_WINDOWS:     # --remote-debugging-pipe on fds 3 / 4 needs preexec_fn
            raise RuntimeError("the browser engine of the burst video runs on Linux / macOS only: use engine=draw")
        to_chrome_r, to_chrome_w = os.pipe()
        from_chrome_r, from_chrome_w = os.pipe()

        def fds():
            # Chrome reads commands on fd 3 and writes on fd 4 (moved up first: the pipe
            # ends may themselves be 3 or 4); kept open through close_fds by pass_fds
            import fcntl
            r, w = fcntl.fcntl(to_chrome_r, fcntl.F_DUPFD, 10), fcntl.fcntl(from_chrome_w, fcntl.F_DUPFD, 10)
            os.dup2(r, 3)
            os.dup2(w, 4)
            os.set_inheritable(3, True)
            os.set_inheritable(4, True)

        profile = os.path.join(workdir, "chrome-profile")
        args = [CHROME, "--headless=new", "--remote-debugging-pipe", "--no-first-run",
                "--no-default-browser-check", "--disable-extensions", "--mute-audio",
                "--autoplay-policy=no-user-gesture-required", "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows",
                f"--window-size={viewport[0]},{viewport[1]}", f"--user-data-dir={profile}", "about:blank"]
        self.proc = subprocess.Popen(args, preexec_fn=fds, pass_fds=(3, 4),
                                     stdout=subprocess.DEVNULL, stderr=open(os.path.join(workdir, "chrome.log"), "wb"),
                                     start_new_session=True)
        os.close(to_chrome_r)
        os.close(from_chrome_w)
        self._w = os.fdopen(to_chrome_w, "wb", buffering=0)
        self._r = os.fdopen(from_chrome_r, "rb", buffering=0)
        self._buf = b""
        self._id = 0
        self.events = []
        self.session = None

    def _read_message(self, timeout):
        import select
        deadline = time.time() + timeout
        while b"\0" not in self._buf:
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError("no message from Chrome")
            ready, _, _ = select.select([self._r], [], [], left)
            if ready:
                chunk = self._r.read(1 << 16)
                if not chunk:
                    raise RuntimeError("Chrome closed the DevTools pipe")
                self._buf += chunk
        msg, self._buf = self._buf.split(b"\0", 1)
        return json.loads(msg)

    def send(self, method, params=None, session=True, timeout=60):
        self._id += 1
        msg = {"id": self._id, "method": method, "params": params or {}}
        if session and self.session:
            msg["sessionId"] = self.session
        self._w.write(json.dumps(msg).encode() + b"\0")
        deadline = time.time() + timeout
        while True:
            reply = self._read_message(max(0.01, deadline - time.time()))
            if reply.get("id") == self._id:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                return reply.get("result", {})
            if "method" in reply:
                self.events.append(reply)

    def wait_event(self, predicate, timeout):
        deadline = time.time() + timeout
        while True:
            for i, ev in enumerate(self.events):
                if predicate(ev):
                    return self.events.pop(i)
            self.events.append(self._read_message(max(0.01, deadline - time.time())))

    def open_page(self, viewport=VIEWPORT):
        targets = self.send("Target.getTargets", session=False)["targetInfos"]
        page = next(t for t in targets if t["type"] == "page")
        self.session = self.send("Target.attachToTarget", {"targetId": page["targetId"], "flatten": True},
                                 session=False)["sessionId"]
        self.send("Page.enable")
        self.send("Runtime.enable")
        self.send("Emulation.setDeviceMetricsOverride",
                  {"width": viewport[0], "height": viewport[1], "deviceScaleFactor": 1, "mobile": False})

    def evaluate(self, expression, await_promise=False):
        result = self.send("Runtime.evaluate", {"expression": expression, "returnByValue": True,
                                                "awaitPromise": await_promise})
        if "exceptionDetails" in result:
            raise RuntimeError(f"page error: {result['exceptionDetails'].get('text')}")
        return result["result"].get("value")

    def close(self):
        try:
            self.send("Browser.close", session=False, timeout=5)
        except Exception:
            pass
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        for fh in (self._w, self._r):
            with contextlib.suppress(OSError):
                fh.close()


READY_JS = """(() => {
  const fly = %(fly)s, bid = %(bid)s;
  const notice = document.querySelector('.pe-notice span');
  if (notice && /not found/.test(notice.textContent)) return 'error: ' + notice.textContent;
  const m = document.querySelector('section.pe-media');
  if (!m) return 'waiting for the PE tab';
  if (m.dataset.fly !== fly || m.dataset.burstId !== bid) return 'waiting for the burst';
  if (m.dataset.traceBurst !== bid) return 'waiting for the trace';
  if (m.dataset.segments !== '1') return 'waiting for the segments';
  const v = m.querySelector('video');
  if (!v || v.readyState < 2) return 'waiting for the video';
  if (m.querySelectorAll('.pe-traces svg').length < 2) return 'waiting for the traces';
  const p = m.querySelector('[data-pose]');
  if (!p || p.dataset.pose !== fly + '_burst_' + bid) return 'pose';
  return 'ready';
})()"""


def _record_one(chrome, base, experiment, fly, burst_id, webm_dir, progress=None):
    """Open the link, click the video button, return the downloaded .webm path"""
    from urllib.parse import urlencode
    url = base + "/?" + urlencode({"tab": "proboscis_extension", "experiment": experiment,
                                   "fly": fly, "burst": int(burst_id)})
    chrome.events.clear()
    chrome.send("Page.navigate", {"url": url})
    js = READY_JS % {"fly": json.dumps(fly), "bid": json.dumps(str(int(burst_id)))}
    deadline = time.time() + READY_TIMEOUT
    pose_deadline = None
    state = None
    while True:
        state = chrome.evaluate(js)
        if state == "ready":
            break
        if state and state.startswith("error"):
            raise LookupError(f"{fly} burst {burst_id}: {state[7:]}")
        if state == "pose":
            pose_deadline = pose_deadline or time.time() + POSE_GRACE
            if time.time() > pose_deadline:
                logger.warning("%s burst %s: no pose shown after %d s, recording without", fly, burst_id, POSE_GRACE)
                break
        if time.time() > deadline:
            raise TimeoutError(f"{fly} burst {burst_id}: the PE tab is still {state} after {READY_TIMEOUT} s")
        time.sleep(0.25)
    time.sleep(0.5)    # let the panels draw a frame of this burst
    duration = chrome.evaluate("document.querySelector('section.pe-media video').duration") or 60
    if progress:
        progress(0.1)

    chrome.send("Browser.setDownloadBehavior", {"behavior": "allowAndName", "downloadPath": webm_dir,
                                                "eventsEnabled": True}, session=False)
    clicked = chrome.evaluate("""(() => { const b = document.querySelector('[data-action="pe-record-video"]');
        if (!b || b.disabled) return false; b.click(); return true; })()""")
    if not clicked:
        raise RuntimeError("the video button is missing or disabled (client bundle without PE links?)")
    begin = chrome.wait_event(lambda e: e.get("method") == "Browser.downloadWillBegin", timeout=duration * 4 + 60)
    guid = begin["params"]["guid"]
    chrome.wait_event(lambda e: e.get("method") == "Browser.downloadProgress"
                      and e["params"].get("guid") == guid and e["params"].get("state") == "completed",
                      timeout=60)
    path = os.path.join(webm_dir, guid)
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        raise RuntimeError(f"the browser recording is empty ({path})")
    return path, begin["params"].get("suggestedFilename")


def webm_to_mp4(webm, output, fps=RECORD_FPS, crf=16):
    """MediaRecorder's .webm (variable frame times) -> constant-fps H.264 yuv420p mp4"""
    tmp = output + ".part.mp4"
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-i", webm, "-an",
           "-vf", f"fps={fps:g},scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "slow",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", tmp]
    run = subprocess.run(cmd, capture_output=True)
    if run.returncode != 0:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise RuntimeError(f"ffmpeg failed: {run.stderr.decode(errors='replace')[-2000:]}")
    os.replace(tmp, output)


def record_bursts(experiment, bursts, viewport=VIEWPORT, fps=RECORD_FPS, keep_webm=False, progress=None):
    """bursts: [(fly, burst_id, output mp4)]. Returns one summary per burst."""
    from idtrackerai_validator_server.experiments import to_relpath
    relpath = to_relpath(experiment)
    out = []
    with _lock, tempfile.TemporaryDirectory(prefix="pe_record_") as workdir:
        t0 = time.time()
        with private_server(relpath, workdir) as base:
            t_server = time.time() - t0
            chrome = Chrome(workdir, viewport)
            try:
                chrome.open_page(viewport)
                for n, (fly, burst_id, output) in enumerate(bursts):
                    t1 = time.time()
                    step = (lambda p, n=n: progress((n + p) / len(bursts))) if progress else None
                    webm, suggested = _record_one(chrome, base, to_flat_name(relpath), fly, burst_id, workdir, step)
                    t_record = time.time() - t1
                    webm_to_mp4(webm, output, fps=fps)
                    if keep_webm:
                        shutil.copy(webm, os.path.splitext(output)[0] + ".webm")
                    os.unlink(webm)
                    out.append({"fly": fly, "burst_id": int(burst_id), "output": output,
                                "browser_filename": suggested, "server_start_s": round(t_server, 1),
                                "record_s": round(t_record, 1), "total_s": round(time.time() - t1, 1)})
                    if step:
                        step(1.0)
            finally:
                chrome.close()
    return out


def to_flat_name(relpath):
    from idtrackerai_validator_server.experiments import to_flat
    return to_flat(relpath)
