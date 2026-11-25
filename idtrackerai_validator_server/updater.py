"""updater.py — "Check for updates": the latest GitHub release, and applying it.

Nothing here runs on its own: GitHub is asked only when someone presses "Check for updates" in the How to use
tab (GET /api/update/check), and an update is applied only when an admin presses "Update" there.

Check: the latest release of $VALIDATOR_UPDATE_REPO (GitHub's API; when it answers "rate limit exceeded",
60 requests an hour per address without $GITHUB_TOKEN, the same is read from the release pages: the
/releases/latest redirect, releases.atom and the release's asset list) compared with the running version
(the package's, and in a git checkout its setup.py's) and commit.

Who: applying restarts the server for everyone using it, so only an admin may (`is_admin`): a request
from this computer (loopback address; VALIDATOR_ADMIN_LOCALHOST=0 turns it off), or a browser session
unlocked with $VALIDATOR_ADMIN_PASSWORD (POST /api/update/admin). Others see "ask your admin". Jobs of
jobs.py still running (pre-download builds, Prepare scenes, swap checks, integrations, ...) refuse the update
(409) unless `force`: they die with the server.

Apply, by install kind (`detect_install`):
  git      the package is an editable install of a git checkout (the lab server): the checkout must be on
           master without uncommitted changes. A helper process, outside the server (a transient
           `systemd-run --user` unit when the server is a user unit: `systemctl restart` kills the whole
           unit; else a detached process), records the commits, fetches the tag, aligns the lab's editable
           packages (flyhostel, imgstore, idtrackerai) with the release's pins (kept when already at or
           ahead of the pin; fast-forwarded when behind and clean; refused when dirty or diverged, before
           anything changed), fast-forwards master to the tag, updates the initialized submodules (not
           recursive), installs with `pip install --no-deps -e` (so pip never replaces an editable
           checkout with a copy) and then only the plain requirements that are missing, makes the web
           interface again when its commit changed (the release's frontend asset when there is one, else
           npm), restarts the server (`systemctl --user restart <unit>`, else kills it and starts the same
           command line again) and waits for /api/health to answer with the new version and commit. Any
           failure after the first change, health check included, rolls everything back to the recorded
           commits and restarts the old version.
  windows  a bundle install of scripts/windows/install.ps1 (%LOCALAPPDATA%\\idtrackerai-validator): the
           release's windows zip is downloaded and checked (size, sha256 digest GitHub gives), the zip of
           the running version too (for a rollback), the server and its console are stopped, the bundle's
           install.ps1 -Update runs, start-validator.bat starts the server again; a failed health check
           installs the previous bundle the same way.
  docker / pip   not updated from here: the answer gives the commands.

Everything is logged to $VALIDATOR_UPDATE_DIR/update.log; status.json holds the steps of the last update,
read by GET /api/update/status (also by the new server after the restart).

CLI (the helper): python -m idtrackerai_validator_server.updater run PLAN.json
"""
import hmac
import html
import importlib.metadata
import ipaddress
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.parse
import zipfile

from idtrackerai_validator_server import lab_paths
from idtrackerai_validator_server.oscompat import IS_WINDOWS, SIGKILL, new_session

logger = logging.getLogger(__name__)

PACKAGE = "idtrackerai_validator_server"
MAIN_BRANCH = "master"
CLIENT = "idtrackerai-validator-client"
UPDATE_DIR = lab_paths.UPDATE_DIR
STATUS_FILE = os.path.join(UPDATE_DIR, "status.json")
LOG_FILE = os.path.join(UPDATE_DIR, "update.log")
# how long the restarted server has to answer /api/health with the new version
HEALTH_TIMEOUT_S = float(os.environ.get("VALIDATOR_UPDATE_HEALTH_TIMEOUT", 240))
# a check older than this is made again before applying
CHECK_MAX_AGE_S = 900
# the lab's packages an update may move, by distribution name (canonical), with their GitHub repository
LAB_DEPS = {"flyhostel": "flyhostel", "imgstore-shaliulab": "imgstore", "idtrackerai-shaliulab": "idtrackerai"}
# never installed by the "plain requirements" step: the lab's packages (direct references in setup.py)
LAB_DISTS = set(LAB_DEPS) | {"imgstore", "idtrackerai", "ethoscopy", "idtrackerai-validator-server"}
FRONTEND_ASSET_RE = re.compile(r"^idtrackerai-validator-frontend-(?P<version>[^-]+)(-[0-9a-f]+)?\.tar\.gz$")
WINDOWS_ASSET_RE = re.compile(r"^idtrackerai-validator-windows-(?P<version>[^-]+)-(?P<commit>[0-9a-f]+)\.zip$")
USER_AGENT = "idtrackerai-validator-updater"


class UpdateError(Exception):
    """An update refused or failed, with a message for the person who pressed the button."""


# ---- versions -------------------------------------------------------------------------------------

def parse_version(text):
    """'v1.5.0' / '1.5' / '1.6.0rc1' / '1.6.0-beta.2' -> a sortable key; a pre-release sorts before its release.
    ValueError when there is no number."""
    match = re.match(r"^\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(.*)$", str(text or ""))
    if not match:
        raise ValueError("not a version: {!r}".format(text))
    major, minor, patch, rest = match.groups()
    pre = rest.lstrip(".-+_ ").strip()
    if rest.startswith("+"):          # build metadata: same version
        pre = ""
    return (int(major), int(minor or 0), int(patch or 0), 0 if pre else 1, pre)


def compare_versions(a, b):
    """-1 when a < b, 0 when equal, 1 when a > b."""
    ka, kb = parse_version(a), parse_version(b)
    return (ka > kb) - (ka < kb)


# ---- GitHub ---------------------------------------------------------------------------------------

def _session():
    import requests
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def _network_error(error):
    return UpdateError("Could not reach GitHub ({}): is this computer online? {}".format(
        lab_paths.UPDATE_WEB, error.__class__.__name__))


def html_to_text(text):
    """Release notes of the Atom feed (HTML) as plain text close to their markdown."""
    text = re.sub(r"(?is)<(script|style).*?</\1>", "", text or "")
    text = re.sub(r"(?i)<h(\d)[^>]*>", lambda m: "\n" + "#" * int(m.group(1)) + " ", text)
    text = re.sub(r"(?i)<li[^>]*>\s*(<p>)?", "\n- ", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</h\d>|</tr>|</ul>|</ol>", "\n", text)
    text = re.sub(r"(?i)</t[dh]>", " | ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def summarize(notes, limit=1500):
    """The start of the release notes, cut at a line end."""
    notes = (notes or "").strip()
    if len(notes) <= limit:
        return notes
    cut = notes.rfind("\n", 0, limit)
    return notes[:cut if cut > limit // 2 else limit].rstrip() + "\n…"


def _release_from_api(data):
    return {
        "tag": data["tag_name"],
        "version": data["tag_name"].lstrip("v"),
        "name": data.get("name") or data["tag_name"],
        "notes": data.get("body") or "",
        "url": data.get("html_url"),
        "published_at": data.get("published_at"),
        "assets": [{"name": a["name"], "size": a.get("size"), "digest": a.get("digest"),
                    "url": a["browser_download_url"]} for a in data.get("assets", [])],
        "source": "api",
    }


def _web_assets(session, tag):
    """Assets of a release read from its HTML list (name, url, sha256 digest; no exact size)."""
    url = "{}/{}/releases/expanded_assets/{}".format(lab_paths.UPDATE_WEB, lab_paths.UPDATE_REPO, tag)
    response = session.get(url, timeout=20)
    response.raise_for_status()
    assets = []
    for block in re.split(r"(?i)<li\b", response.text)[1:]:
        link = re.search(r'href="([^"]*/releases/download/[^"]+)"', block)
        if not link:
            continue
        href = html.unescape(link.group(1))
        digest = re.search(r"sha256:[0-9a-f]{64}", block)
        assets.append({"name": urllib.parse.unquote(href.rsplit("/", 1)[-1]), "size": None,
                       "digest": digest.group(0) if digest else None,
                       "url": urllib.parse.urljoin(lab_paths.UPDATE_WEB + "/", href)})
    return assets


def _release_from_web(session, tag=None):
    """The latest release (or `tag`) without the API: its page redirect, the Atom feed and the asset list."""
    base = "{}/{}".format(lab_paths.UPDATE_WEB, lab_paths.UPDATE_REPO)
    if tag is None:
        response = session.get(base + "/releases/latest", allow_redirects=False, timeout=20)
        location = response.headers.get("Location", "")
        if "/releases/tag/" not in location:
            raise UpdateError("GitHub lists no release of {}".format(lab_paths.UPDATE_REPO))
        tag = urllib.parse.unquote(location.rsplit("/releases/tag/", 1)[1])
    release = {"tag": tag, "version": tag.lstrip("v"), "name": tag, "notes": "",
               "url": "{}/releases/tag/{}".format(base, tag), "published_at": None, "assets": [], "source": "web"}
    feed = session.get(base + "/releases.atom", timeout=20)
    if feed.ok:
        for entry in re.findall(r"(?s)<entry>(.*?)</entry>", feed.text):
            if re.search(r'href="[^"]*/releases/tag/{}"'.format(re.escape(tag)), entry):
                title = re.search(r"(?s)<title>(.*?)</title>", entry)
                content = re.search(r"(?s)<content[^>]*>(.*?)</content>", entry)
                updated = re.search(r"<updated>(.*?)</updated>", entry)
                release["name"] = html.unescape(title.group(1)).strip() if title else tag
                release["notes"] = html_to_text(html.unescape(content.group(1))) if content else ""
                release["published_at"] = updated.group(1) if updated else None
                break
    release["assets"] = _web_assets(session, tag)
    return release


def _api_get(session, path):
    """GitHub API JSON; None when rate limited (the caller reads the web pages instead)."""
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    response = session.get(lab_paths.UPDATE_API + path, headers=headers, timeout=20)
    if response.status_code in (403, 429) and (
            response.headers.get("X-RateLimit-Remaining") == "0" or "rate limit" in response.text.lower()):
        reset = response.headers.get("X-RateLimit-Reset")
        logger.info("GitHub API rate limited (reset %s): reading the release pages", reset)
        return None
    if response.status_code == 404:
        raise UpdateError("GitHub has no release of {} ({})".format(lab_paths.UPDATE_REPO, path))
    response.raise_for_status()
    return response.json()


def latest_release(tag=None):
    """The latest release (or `tag`): tag, version, name, notes (markdown or text), url, published_at,
    assets [{name, size, digest, url}], source (api / web). UpdateError when GitHub cannot be reached."""
    import requests
    session = _session()
    path = "/repos/{}/releases/{}".format(lab_paths.UPDATE_REPO, "tags/" + tag if tag else "latest")
    try:
        data = _api_get(session, path)
        if data is not None:
            return _release_from_api(data)
        try:
            return _release_from_web(session, tag)
        except requests.HTTPError as error:
            raise UpdateError("GitHub's API is rate limited for this address (60 requests an hour; set "
                              "GITHUB_TOKEN to raise it) and the release pages failed too: {}".format(error))
    except requests.ConnectionError as error:
        raise _network_error(error)
    except requests.Timeout as error:
        raise _network_error(error)
    except requests.HTTPError as error:
        raise UpdateError("GitHub answered {}".format(error))


# ---- what is installed ------------------------------------------------------------------------------

def _canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def package_root():
    """Folder holding the idtrackerai_validator_server package (the repository in a source checkout)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _direct_url(dist):
    try:
        text = dist.read_text("direct_url.json")
        return json.loads(text) if text else None
    except (OSError, ValueError):
        return None


def _url_path(url):
    """Local path of a file:// URL (None for others)."""
    if not url or not url.startswith("file:"):
        return None
    from urllib.request import url2pathname   # Windows: /C:/x -> C:\\x
    return os.path.normpath(url2pathname(urllib.parse.urlparse(url).path))


def _same_path(a, b):
    if not a or not b:
        return False
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def installed_version():
    try:
        return importlib.metadata.version(PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_version(root):
    """version= of setup.py in a source checkout (an editable install's metadata can lag behind)."""
    try:
        with open(os.path.join(root, "setup.py"), encoding="utf-8") as f:
            match = re.search(r"""\bversion\s*=\s*['"]([^'"]+)['"]""", f.read())
        return match.group(1) if match else None
    except OSError:
        return None


def systemd_unit():
    """The systemd unit this server runs as ({unit, user}), or None (not started by systemd)."""
    if not os.environ.get("INVOCATION_ID"):
        return None
    try:
        with open("/proc/self/cgroup") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in lines:
        path = line.split(":", 2)[-1]
        for part in reversed(path.split("/")):
            if part.endswith(".service") and not part.startswith("user@"):
                return {"unit": part, "user": "/user@" in path}
    return None


def detect_install():
    """kind: git (editable install of a git checkout), windows (scripts/windows/install.ps1 bundle), docker
    or pip (any other install), with what the update needs to know."""
    root = package_root()
    try:
        dist = importlib.metadata.distribution(PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        dist = None
    direct = _direct_url(dist) if dist else None
    editable = bool(direct and direct.get("dir_info", {}).get("editable"))
    info = {"root": root, "python": sys.executable, "prefix": sys.prefix, "editable": editable,
            "platform": sys.platform, "service": systemd_unit(), "reason": None}
    forced = os.environ.get("VALIDATOR_INSTALL")
    if forced in ("git", "windows", "docker", "pip"):
        info.update(kind=forced, reason="VALIDATOR_INSTALL={}".format(forced))
    elif os.path.exists("/.dockerenv"):
        info.update(kind="docker", reason="/.dockerenv exists")
    elif editable and _same_path(_url_path(direct.get("url")), root) and os.path.isfile(os.path.join(root, "setup.py")) \
            and _git_ok(root):
        info.update(kind="git", reason="editable install of the git checkout {}".format(root))
    elif IS_WINDOWS and os.path.normcase(os.path.realpath(sys.prefix)).startswith(
            os.path.normcase(os.path.realpath(lab_paths.WINDOWS_INSTALL_DIR)) + os.sep):
        info.update(kind="windows", reason="bundle install in {}".format(lab_paths.WINDOWS_INSTALL_DIR),
                    install_dir=lab_paths.WINDOWS_INSTALL_DIR)
    else:
        info.update(kind="pip", reason="editable install outside a git checkout" if editable
                    else "installed with pip into {}".format(sys.prefix))
    if info["kind"] == "windows":
        info.setdefault("install_dir", lab_paths.WINDOWS_INSTALL_DIR)
    return info


# ---- git ------------------------------------------------------------------------------------------

def git(root, *args, timeout=120):
    """stdout of `git -C root args` (stripped); CalledProcessError / OSError when it fails."""
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=timeout,
                          check=True).stdout.strip()


def _git_ok(root):
    try:
        return _same_path(git(root, "rev-parse", "--show-toplevel", timeout=10), root)
    except (OSError, subprocess.SubprocessError):
        return False


def _is_ancestor(root, a, b):
    return subprocess.run(["git", "-C", root, "merge-base", "--is-ancestor", a, b],
                          capture_output=True, timeout=60).returncode == 0


def _has_commit(root, sha):
    return subprocess.run(["git", "-C", root, "cat-file", "-e", sha + "^{commit}"],
                          capture_output=True, timeout=30).returncode == 0


def checkout_state(root):
    """head, branch (None when detached), dirty (tracked files changed, submodules included: their changes
    would be lost by `git submodule update`)."""
    head = git(root, "rev-parse", "HEAD")
    branch = subprocess.run(["git", "-C", root, "symbolic-ref", "-q", "--short", "HEAD"],
                            capture_output=True, text=True, timeout=30).stdout.strip() or None
    porcelain = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no",
                                "--ignore-submodules=untracked"], capture_output=True, text=True, timeout=60,
                               check=True).stdout   # not stripped: " M a.txt" keeps its first column
    dirty = [line for line in porcelain.splitlines() if line.strip()]
    return {"head": head, "branch": branch, "dirty": dirty}


def git_blockers(root, state=None):
    """Why the checkout cannot be updated in place (empty list: it can)."""
    state = state or checkout_state(root)
    blockers = []
    if state["branch"] != MAIN_BRANCH:
        blockers.append("{} is {} instead of {}: check out {} (git -C {} checkout {}).".format(
            root, "on branch " + state["branch"] if state["branch"] else "not on a branch (detached HEAD)",
            MAIN_BRANCH, MAIN_BRANCH, root, MAIN_BRANCH))
    if state["dirty"]:
        blockers.append("{} has uncommitted changes ({}): commit or stash them first.".format(
            root, ", ".join(line[3:] for line in state["dirty"][:8]) + (" …" if len(state["dirty"]) > 8 else "")))
    return blockers


def read_pins(root, rev):
    """Commits a revision of the repository pins: setup.py's PINS, and the gitlinks of its submodules
    (which setup.py prefers too)."""
    pins = {}
    try:
        text = git(root, "show", "{}:setup.py".format(rev))
        block = re.search(r"(?s)^PINS\s*=\s*\{(.*?)^\}", text, re.M)
        if block:
            pins.update(re.findall(r'"([^"]+)"\s*:\s*"([0-9a-f]{40})"', block.group(1)))
    except (OSError, subprocess.SubprocessError):
        pass
    for path in ("flyhostel", CLIENT):
        try:
            out = git(root, "ls-tree", rev, path).split()
            if len(out) >= 3 and out[1] == "commit":
                pins[path] = out[2]
        except (OSError, subprocess.SubprocessError):
            pass
    return pins


def lab_deps():
    """The lab's packages installed here: name, version, editable, path (local source), url."""
    found = {}
    for dist in importlib.metadata.distributions():
        try:
            name = _canon(dist.metadata["Name"] or "")
        except Exception:   # noqa: BLE001 — broken metadata of another package
            continue
        if name not in LAB_DEPS or name in found:
            continue
        direct = _direct_url(dist) or {}
        found[name] = {"name": name, "version": dist.version, "url": direct.get("url"),
                       "editable": bool(direct.get("dir_info", {}).get("editable")),
                       "path": _url_path(direct.get("url"))}
    return found


def dep_pin(name, pins, deps):
    """The commit a release pins a lab package at, or None when it pins none."""
    if name == "flyhostel":
        return pins.get("flyhostel")
    if name == "imgstore-shaliulab":
        return pins.get("flyhostel/libraries/imgstore")
    if name == "idtrackerai-shaliulab":
        flyhostel = deps.get("flyhostel")
        pin = pins.get("flyhostel")
        if flyhostel and flyhostel.get("path") and pin and _git_ok(flyhostel["path"]) \
                and _has_commit(flyhostel["path"], pin):
            try:
                out = git(flyhostel["path"], "ls-tree", pin, "libraries/idtrackerai").split()
                if len(out) >= 3 and out[1] == "commit":
                    return out[2]
            except (OSError, subprocess.SubprocessError):
                return None
    return None


def plan_dep(dep, pin, fetch=None):
    """What the update does with a lab package installed in editable mode from a git checkout:
    keep (no pin, or already at / ahead of it), ff (behind and clean: moved to the pin) or refuse."""
    path = dep.get("path")
    out = {"name": dep["name"], "path": path, "pin": pin, "action": "keep", "reason": None}
    if not dep.get("editable") or not path:
        out["reason"] = "not an editable install"
        return out
    if not pin:
        out["reason"] = "the release pins no commit of it"
        return out
    if not _git_ok(path):
        out["reason"] = "{} is not a git checkout".format(path)
        return out
    state = checkout_state(path)
    out.update(head=state["head"], branch=state["branch"])
    if not _has_commit(path, pin) and fetch is not None:
        fetch(path)
    if not _has_commit(path, pin):
        out.update(action="refuse", reason="{} does not have commit {} (git fetch failed?)".format(path, pin[:12]))
    elif _is_ancestor(path, pin, "HEAD"):
        out["reason"] = "at {} or ahead of it".format(pin[:12])
    elif not _is_ancestor(path, "HEAD", pin):
        out.update(action="refuse", reason="{} ({} at {}) neither contains nor is behind the pinned commit {}: "
                   "merge or rebase it by hand".format(path, state["branch"] or "detached", state["head"][:12], pin[:12]))
    elif state["dirty"]:
        out.update(action="refuse", reason="{} is behind the pinned commit {} but has uncommitted changes".format(
            path, pin[:12]))
    else:
        out.update(action="ff", reason="{} → {}".format(state["head"][:12], pin[:12]))
    return out


# ---- jobs that die with the server ---------------------------------------------------------------------

def busy():
    """Jobs of this server that a restart would kill, and work that survives it (informational)."""
    from idtrackerai_validator_server.jobs import running_jobs
    survives = []
    try:
        from idtrackerai_validator_server.gpu_queue import read_state
        for entry in read_state().get("entries", []):
            if entry.get("status") == "running":
                survives.append("GPU queue: {} ({})".format(entry.get("what") or entry.get("kind"), entry.get("kind")))
    except Exception:   # noqa: BLE001 — the queue is optional here
        pass
    return {"jobs": running_jobs(), "survives": survives}


# ---- admin ---------------------------------------------------------------------------------------------

def _loopback(address):
    try:
        return ipaddress.ip_address((address or "").split("%")[0]).is_loopback
    except ValueError:
        return False


def is_admin(request, session):
    """May this request apply an update? From this computer, or a session unlocked with the admin password."""
    if os.environ.get("VALIDATOR_ADMIN_LOCALHOST", "1") != "0" and _loopback(request.remote_addr):
        return True
    return bool(os.environ.get("VALIDATOR_ADMIN_PASSWORD")) and session.get("update_admin") is True


# ---- status of the update in progress / last update ----------------------------------------------------

def read_status():
    try:
        with open(STATUS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _pid_alive(pid):
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).status() != psutil.STATUS_ZOMBIE
    except Exception:   # noqa: BLE001
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False


def update_running():
    status = read_status()
    return bool(status and status.get("state") == "running" and _pid_alive(status.get("helper_pid"))) \
        or bool(status and status.get("state") == "running" and time.time() - status.get("updated", 0) < 60)


class Status:
    """status.json (steps, last log lines) and update.log, written by the helper."""

    def __init__(self, plan):
        os.makedirs(UPDATE_DIR, exist_ok=True)
        self.data = {"id": plan["id"], "kind": plan["kind"], "state": "running", "from": plan["from"],
                     "to": plan["to"], "steps": [], "log": [], "message": None, "started": time.time(),
                     "ended": None, "updated": time.time(), "helper_pid": os.getpid(), "rolled_back": False}
        self._lock = threading.Lock()
        self.save()

    def save(self):
        with self._lock:
            self.data["updated"] = time.time()
            tmp = STATUS_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=1)
            os.replace(tmp, STATUS_FILE)

    def log(self, text):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            for line in str(text).splitlines() or [""]:
                f.write("{} [{}] {}\n".format(stamp, self.data["id"], line))
        with self._lock:
            self.data["log"].extend(str(text).splitlines() or [""])
            del self.data["log"][:-400]
        self.save()

    def step(self, name, state="running", detail=None):
        with self._lock:
            for step in self.data["steps"]:
                if step["name"] == name:
                    step.update(state=state, detail=detail if detail is not None else step.get("detail"))
                    break
            else:
                self.data["steps"].append({"name": name, "state": state, "detail": detail})
        self.log("== {}: {}{}".format(name, state, " ({})".format(detail) if detail else ""))

    def end(self, state, message):
        self.data.update(state=state, message=message, ended=time.time())
        self.log("== {}: {}".format(state, message))


def _run(status, cmd, cwd=None, timeout=1800, env=None):
    """Run a command, its output in the log; CalledProcessError when it fails."""
    status.log("$ " + " ".join(cmd) + ("   (in {})".format(cwd) if cwd else ""))
    process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace")
    deadline = time.time() + timeout
    lines = []
    for line in process.stdout:
        line = line.rstrip()
        lines.append(line)
        status.log("  " + line)
        if time.time() > deadline:
            process.kill()
            break
    code = process.wait()
    if time.time() > deadline:
        raise subprocess.TimeoutExpired(cmd, timeout)
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd, output="\n".join(lines[-20:]))
    return "\n".join(lines)


def _download(status, asset, folder):
    """Download a release asset, checking its size and sha256 digest when GitHub gives them."""
    import hashlib
    import requests
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, os.path.basename(asset["name"]))
    status.log("Downloading {} ({})".format(asset["url"], _fmt_size(asset.get("size"))))
    digest = hashlib.sha256()
    size = 0
    try:
        with _session().get(asset["url"], stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(path + ".part", "wb") as f:
                for block in response.iter_content(1 << 20):
                    f.write(block)
                    digest.update(block)
                    size += len(block)
    except requests.RequestException as error:
        raise UpdateError("Download of {} failed: {}".format(asset["name"], error))
    if asset.get("size") and size != asset["size"]:
        raise UpdateError("{}: {} bytes downloaded, GitHub lists {}".format(asset["name"], size, asset["size"]))
    if asset.get("digest"):
        algorithm, _, expected = asset["digest"].partition(":")
        if algorithm == "sha256" and digest.hexdigest() != expected:
            raise UpdateError("{}: sha256 {} differs from GitHub's {}".format(asset["name"], digest.hexdigest(), expected))
        status.log("sha256 checked: {}".format(expected))
    else:
        status.log("GitHub gives no digest of {}: only the size was checked".format(asset["name"]))
    os.replace(path + ".part", path)
    return path


def _fmt_size(n):
    return "{:.1f} MB".format(n / 1e6) if n else "size unknown"


def _safe_extract(archive, dest):
    """Extract a .zip / .tar.gz, refusing members outside dest."""
    dest = os.path.realpath(dest)
    os.makedirs(dest, exist_ok=True)

    def check(name):
        target = os.path.realpath(os.path.join(dest, name))
        if target != dest and not target.startswith(dest + os.sep):
            raise UpdateError("{} holds a path outside its folder: {}".format(os.path.basename(archive), name))

    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                check(name)
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                check(member.name)
                if member.issym() or member.islnk() or member.isdev():
                    raise UpdateError("{} holds a link or device: {}".format(os.path.basename(archive), member.name))
            t.extractall(dest)


def health(base, timeout=HEALTH_TIMEOUT_S, version=None, commit=None, status=None, not_pid=None):
    """Wait for /api/health at base to answer with that version / commit (and another pid than not_pid).
    Returns its answer, or raises UpdateError."""
    import requests
    import urllib3
    urllib3.disable_warnings()
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            response = requests.get(base + "/api/health", timeout=5, verify=False)
            if response.ok:
                data = response.json()
                last = data
                same_version = version is None or (data.get("version") and compare_versions(data["version"], version) == 0)
                same_commit = commit is None or (data.get("commit") and (
                    commit.startswith(data["commit"]) or data["commit"].startswith(commit)))
                other = not_pid is None or data.get("pid") != not_pid
                if same_version and same_commit and other:
                    return data
        except (requests.RequestException, ValueError) as error:
            last = error.__class__.__name__
        time.sleep(2)
    raise UpdateError("The server at {} did not answer with version {} / commit {} within {:.0f} s (last: {})".format(
        base, version, (commit or "")[:12], timeout, last))


# ---- the helper ----------------------------------------------------------------------------------------

class _Update:
    def __init__(self, plan):
        self.plan = plan
        self.status = Status(plan)
        self.server = plan["server"]

    def base_url(self):
        host = self.server.get("host") or "127.0.0.1"
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        if ":" in host and not host.startswith("["):
            host = "[{}]".format(host)
        return "{}://{}:{}".format(self.server.get("scheme", "http"), host, self.server["port"])

    def check_health(self, version, commit, not_pid=None):
        if self.plan.get("test_fail_health") and not self.status.data["rolled_back"]:
            # tests: the new server answers, but the check fails as if it did not (VALIDATOR_UPDATE_TEST_FAIL_HEALTH)
            answer = health(self.base_url(), version=version, commit=commit, not_pid=not_pid)
            raise UpdateError("health check failed on purpose (VALIDATOR_UPDATE_TEST_FAIL_HEALTH); the server answered {}".format(answer))
        return health(self.base_url(), version=version, commit=commit, not_pid=not_pid)


class GitUpdate(_Update):
    """The lab server's update: an editable install of a git checkout (see the module docstring)."""

    def __init__(self, plan):
        super().__init__(plan)
        self.root = plan["install"]["root"]
        self.python = plan["install"]["python"]
        self.tag = plan["to"]["tag"]
        self.record = {"moved": [], "reinstalled": [], "frontend_backup": None, "viewer_moved": False}

    def pip(self, *args):
        return _run(self.status, [self.python, "-m", "pip", "--disable-pip-version-check", *args], timeout=3600)

    def fetch(self, path):
        try:
            _run(self.status, ["git", "-C", path, "fetch", "--quiet", "origin"], timeout=600)
        except (subprocess.SubprocessError, OSError) as error:
            self.status.log("git fetch in {} failed: {}".format(path, error))

    # -- 1. checks, no change made --
    def preflight(self):
        st = self.status
        st.step("Check the checkout")
        state = checkout_state(self.root)
        blockers = git_blockers(self.root, state)
        if blockers:
            raise UpdateError(" ".join(blockers))
        self.record.update(head=state["head"], branch=state["branch"])
        service = self.plan["install"].get("service")
        if service and not service.get("user"):
            raise UpdateError("The server runs as the system unit {}: it cannot restart itself. Update it by hand "
                              "(git pull, pip install --no-deps -e ., sudo systemctl restart {}).".format(
                                  service["unit"], service["unit"]))
        st.step("Check the checkout", "done", "{} at {}".format(state["branch"], state["head"][:12]))

        st.step("Fetch " + self.tag)
        try:
            _run(st, ["git", "-C", self.root, "fetch", "--quiet", "origin",
                      "+refs/tags/{0}:refs/tags/{0}".format(self.tag)], timeout=600)
        except (subprocess.SubprocessError, OSError) as error:
            raise UpdateError("git fetch of {} from origin failed: {}".format(self.tag, getattr(error, "output", None) or error))
        self.target = git(self.root, "rev-parse", self.tag + "^{commit}")
        if self.target == state["head"]:
            raise UpdateError("The checkout is at {} already: restart the server to run it.".format(self.tag))
        if not _is_ancestor(self.root, state["head"], self.target):
            raise UpdateError("{} ({}) has commits that are not in {}: it cannot be fast-forwarded. Merge by hand."
                              .format(MAIN_BRANCH, state["head"][:12], self.tag))
        self.version = source_version_at(self.root, self.tag) or self.plan["to"]["version"]
        st.step("Fetch " + self.tag, "done", self.target[:12])

        st.step("Check the lab's packages")
        self.old_pins = read_pins(self.root, "HEAD")
        self.new_pins = read_pins(self.root, self.target)
        deps = lab_deps()
        self.deps = []
        for name, dep in sorted(deps.items()):
            pin = dep_pin(name, self.new_pins, deps)
            if dep["editable"]:
                plan = plan_dep(dep, pin, fetch=self.fetch)
            else:
                old = dep_pin(name, self.old_pins, deps)
                plan = {"name": name, "path": dep.get("path"), "pin": pin, "old_pin": old, "action": "keep",
                        "reason": "not editable, pin unchanged"}
                if pin and old and pin != old and name in ("flyhostel", "imgstore-shaliulab"):
                    plan.update(action="reinstall", reason="not editable: reinstalled from {} at {}".format(
                        LAB_DEPS[name], pin[:12]))
            st.log("  {}: {} ({})".format(name, plan["action"], plan["reason"]))
            self.deps.append(plan)
        refused = [d for d in self.deps if d["action"] == "refuse"]
        if refused:
            raise UpdateError("Not updated: " + "; ".join(d["reason"] for d in refused))
        st.step("Check the lab's packages", "done", ", ".join("{} {}".format(d["name"], d["action"]) for d in self.deps) or "none installed")

    # -- 2. changes --
    def apply(self):
        st = self.status
        st.step("Move to " + self.tag)
        self.record["viewer_moved"] = True
        _run(st, ["git", "-C", self.root, "merge", "--ff-only", "--quiet", self.target])
        _run(st, ["git", "-C", self.root, "submodule", "update"], timeout=1800)   # initialized ones, not recursive
        for dep in self.deps:
            if dep["action"] == "ff":
                self.record["moved"].append(dep)
                if dep.get("branch"):
                    _run(st, ["git", "-C", dep["path"], "merge", "--ff-only", "--quiet", dep["pin"]])
                else:
                    _run(st, ["git", "-C", dep["path"], "checkout", "--quiet", dep["pin"]])
        st.step("Move to " + self.tag, "done", self.target[:12])

        st.step("Install")
        self.pip("install", "--no-deps", "-e", self.root)
        for dep in self.record["moved"]:
            self.pip("install", "--no-deps", "-e", dep["path"])
        for dep in self.deps:
            if dep["action"] == "reinstall":
                self.record["reinstalled"].append(dep)
                self.pip("install", "--no-deps", "{}/{}/archive/{}.tar.gz".format(lab_paths.LAB_GITHUB, LAB_DEPS[dep["name"]], dep["pin"]))
        missing = plain_requirements(self.python, [PACKAGE] + [d["name"] for d in self.record["moved"]])
        if missing:
            self.pip("install", *missing)
        else:
            st.log("No new requirement to install")
        st.step("Install", "done")

        st.step("Web interface")
        self.frontend()

    def frontend(self):
        st = self.status
        dest = os.path.join(self.root, PACKAGE, "frontend")
        changed = self.old_pins.get(CLIENT) != self.new_pins.get(CLIENT)
        if not changed and os.path.isfile(os.path.join(dest, "index.html")):
            st.step("Web interface", "done", "unchanged")
            return
        backup = os.path.join(UPDATE_DIR, "frontend-backup")
        if os.path.isdir(backup):
            shutil.rmtree(backup)
        if os.path.isdir(dest):
            shutil.copytree(dest, backup)
            self.record["frontend_backup"] = backup
        else:
            self.record["frontend_backup"] = "none"
        asset = self.plan.get("frontend_asset")
        if asset:
            archive = _download(st, asset, os.path.join(UPDATE_DIR, "downloads"))
            tmp = tempfile.mkdtemp(prefix="frontend_", dir=UPDATE_DIR)
            _safe_extract(archive, tmp)
            source = tmp if os.path.isfile(os.path.join(tmp, "index.html")) else os.path.join(tmp, "frontend")
            if not os.path.isfile(os.path.join(source, "index.html")):
                raise UpdateError("{} has no index.html".format(asset["name"]))
            how = "from " + asset["name"]
        else:
            client = os.path.join(self.root, CLIENT)
            _run(st, ["git", "-C", self.root, "submodule", "update", "--init", CLIENT], timeout=1800)
            npm = lab_paths.NPM
            env = dict(os.environ)
            env["PATH"] = os.path.dirname(npm) + os.pathsep + env.get("PATH", "") if os.path.isabs(npm) else env.get("PATH", "")
            _run(st, [npm, "ci", "--no-audit", "--no-fund"], cwd=client, timeout=1800, env=env)
            _run(st, [npm, "run", "build"], cwd=client, timeout=1800, env=env)
            source = os.path.join(client, "build")
            if not os.path.isfile(os.path.join(source, "index.html")):
                raise UpdateError("npm run build made no index.html")
            how = "built with npm"
            tmp = None
        new = dest + ".new"
        if os.path.isdir(new):
            shutil.rmtree(new)
        shutil.copytree(source, new)
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        os.replace(new, dest)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        st.step("Web interface", "done", how)

    # -- 3. restart --
    def restart(self):
        st = self.status
        service = self.plan["install"].get("service")
        if service:
            _run(st, [lab_paths.SYSTEMCTL, "--user", "restart", service["unit"]], timeout=300)
            return
        pid = self.server.get("pid")
        _stop_pid(st, pid)
        log = open(os.path.join(UPDATE_DIR, "server-{}.log".format(time.strftime("%Y%m%d-%H%M%S"))), "ab")
        st.log("Starting {} (in {})".format(" ".join(self.server["argv"]), self.server["cwd"]))
        process = subprocess.Popen(self.server["argv"], cwd=self.server["cwd"], stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, **new_session())
        self.server["pid"] = process.pid
        st.data["server_pid"] = process.pid
        st.save()

    def rollback(self, reason):
        st = self.status
        st.data["rolled_back"] = True
        st.step("Roll back", "running", reason)
        try:
            if self.record.get("viewer_moved"):
                _run(st, ["git", "-C", self.root, "checkout", "--quiet", self.record["branch"]])
                _run(st, ["git", "-C", self.root, "reset", "--quiet", "--hard", self.record["head"]])
                _run(st, ["git", "-C", self.root, "submodule", "update"], timeout=1800)
            for dep in self.record["moved"]:
                if dep.get("branch"):
                    _run(st, ["git", "-C", dep["path"], "checkout", "--quiet", dep["branch"]])
                    _run(st, ["git", "-C", dep["path"], "reset", "--quiet", "--hard", dep["head"]])
                else:
                    _run(st, ["git", "-C", dep["path"], "checkout", "--quiet", dep["head"]])
                self.pip("install", "--no-deps", "-e", dep["path"])
            for dep in self.record["reinstalled"]:
                self.pip("install", "--no-deps", "{}/{}/archive/{}.tar.gz".format(
                    lab_paths.LAB_GITHUB, LAB_DEPS[dep["name"]], dep["old_pin"]))
            if self.record.get("viewer_moved"):
                self.pip("install", "--no-deps", "-e", self.root)
            backup = self.record.get("frontend_backup")
            dest = os.path.join(self.root, PACKAGE, "frontend")
            if backup and backup != "none" and os.path.isdir(backup):
                if os.path.isdir(dest):
                    shutil.rmtree(dest)
                shutil.copytree(backup, dest)
            st.step("Roll back", "done", "back at {}".format(self.record["head"][:12]))
            return True
        except Exception as error:   # noqa: BLE001
            st.step("Roll back", "failed", str(error))
            return False

    def run(self):
        st = self.status
        old = self.plan["from"]
        try:
            self.preflight()
        except Exception as error:   # noqa: BLE001 — nothing changed
            st.end("failed", "{} Nothing was changed.".format(error if isinstance(error, UpdateError) else repr(error)))
            return 1
        try:
            self.apply()
        except Exception as error:   # noqa: BLE001
            message = _describe(error)
            ok = self.rollback(message)
            st.end("rolled_back" if ok else "rollback_failed",
                   "Update failed: {}. {}".format(message, "Rolled back to {}; the server was not restarted.".format(
                       old["version"]) if ok else "Rolling back failed too: see {}".format(LOG_FILE)))
            return 1
        st.step("Restart")
        old_pid = self.server.get("pid")
        try:
            self.restart()
            st.step("Restart", "done")
            st.step("Health check")
            answer = self.check_health(self.version, self.target, not_pid=old_pid)
            st.step("Health check", "done", "version {} commit {}".format(answer.get("version"), answer.get("commit")))
            st.end("done", "Updated from {} to {}.".format(old["version"], self.version))
            return 0
        except Exception as error:   # noqa: BLE001
            message = _describe(error)
            st.step("Health check", "failed", message)
            ok = self.rollback(message)
            try:
                bad_pid = self.server.get("pid")
                self.restart()
                answer = health(self.base_url(), version=None, commit=self.record["head"], not_pid=bad_pid)
                st.log("The old version answers again: {}".format(answer))
            except Exception as again:   # noqa: BLE001
                st.end("rollback_failed", "Update failed ({}) and the old version did not start again: {}. See {}".format(
                    message, _describe(again), LOG_FILE))
                return 1
            st.end("rolled_back" if ok else "rollback_failed",
                   "Update failed ({}). {}".format(message, "Rolled back to {} and restarted.".format(old["version"])
                                                   if ok else "Rolling back failed: see {}".format(LOG_FILE)))
            return 1


def source_version_at(root, rev):
    try:
        match = re.search(r"""\bversion\s*=\s*['"]([^'"]+)['"]""", git(root, "show", "{}:setup.py".format(rev)))
        return match.group(1) if match else None
    except (OSError, subprocess.SubprocessError):
        return None


def _describe(error):
    if isinstance(error, subprocess.CalledProcessError):
        tail = (error.output or "").strip().splitlines()[-3:]
        return "{} exited with {}{}".format(" ".join(error.cmd[:4]), error.returncode, ": " + " / ".join(tail) if tail else "")
    return str(error) if isinstance(error, UpdateError) else repr(error)


def _stop_pid(status, pid, wait=30):
    if not pid:
        return
    status.log("Stopping the server (pid {})".format(pid))
    try:
        os.kill(int(pid), signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + wait
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.5)
    if _pid_alive(pid):
        try:
            os.kill(int(pid), SIGKILL)
        except OSError:
            pass
        time.sleep(1)


_REQUIRES = (
    "import importlib.metadata as m, json, sys\n"
    "out = []\n"
    "for name in sys.argv[1:]:\n"
    "    try:\n"
    "        out += m.requires(name) or []\n"
    "    except m.PackageNotFoundError:\n"
    "        pass\n"
    "print(json.dumps(out))\n"
)


def filter_requirements(requirements, installed):
    """Requirement strings pip may install without touching the lab's packages: no direct references
    (`name @ url`), no extras-only ones, none of LAB_DISTS, markers true here; and only those not satisfied
    (`installed`: canonical name -> version)."""
    try:
        from packaging.requirements import Requirement
    except ImportError:   # pragma: no cover
        from pip._vendor.packaging.requirements import Requirement
    out = []
    for text in requirements:
        try:
            req = Requirement(text)
        except Exception:   # noqa: BLE001
            continue
        if req.url or _canon(req.name) in LAB_DISTS:
            continue
        if req.marker is not None:
            if "extra" in str(req.marker):
                continue
            if not req.marker.evaluate():
                continue
        version = installed.get(_canon(req.name))
        if version is not None and (not req.specifier or req.specifier.contains(version, prereleases=True)):
            continue
        req.marker = None
        if str(req) not in out:
            out.append(str(req))
    return out


def plain_requirements(python, dists):
    """Requirements of these distributions that the server's python is missing (read in a new process:
    the metadata just written by pip)."""
    out = subprocess.run([python, "-c", _REQUIRES, *dists], capture_output=True, text=True, timeout=120, check=True)
    requirements = json.loads(out.stdout)
    script = "import importlib.metadata as m, json; print(json.dumps({d.metadata['Name']: d.version for d in m.distributions() if d.metadata['Name']}))"
    installed = json.loads(subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=120,
                                          check=True).stdout)
    return filter_requirements(requirements, {_canon(k): v for k, v in installed.items()})


class WindowsUpdate(_Update):
    """Bundle install of scripts/windows/install.ps1 (not testable on Linux: scripts/windows/TESTING.md)."""

    def __init__(self, plan):
        super().__init__(plan)
        self.install_dir = plan["install"]["install_dir"]
        self.python = plan["install"]["python"]

    def bundle(self, asset):
        path = _download(self.status, asset, os.path.join(UPDATE_DIR, "downloads"))
        dest = os.path.join(UPDATE_DIR, "bundles", os.path.splitext(os.path.basename(path))[0])
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        _safe_extract(path, dest)
        if not os.path.isfile(os.path.join(dest, "install.ps1")):
            raise UpdateError("{} has no install.ps1".format(asset["name"]))
        return dest

    def stop(self):
        """The server, and the console window of start-validator.bat it runs in."""
        import psutil
        pid = self.server.get("pid")
        try:
            process = psutil.Process(int(pid))
        except (psutil.Error, TypeError, ValueError):
            return
        victims = [process]
        parent = process.parent()
        while parent is not None and parent.name().lower() in ("powershell.exe", "pwsh.exe", "cmd.exe"):
            try:
                if "start-validator" not in " ".join(parent.cmdline()).lower():
                    break
            except psutil.Error:
                break
            victims.append(parent)
            parent = parent.parent()
        self.status.log("Stopping the server (pid {}) and its console".format(pid))
        for victim in victims:
            try:
                for child in victim.children(recursive=True):
                    child.kill()
                victim.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(victims, timeout=30)

    def install(self, folder):
        _run(self.status, ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                           os.path.join(folder, "install.ps1"), "-Update", "-InstallDir", self.install_dir], timeout=3600)

    def start(self):
        launcher = os.path.join(self.install_dir, "start-validator.bat")
        env = dict(os.environ, VALIDATOR_NO_BROWSER="1")
        self.status.log("Starting {}".format(launcher))
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen(["cmd.exe", "/c", "start", "FlyHostel validator", launcher], cwd=self.install_dir,
                         env=env, creationflags=flags)

    def run(self):
        st = self.status
        old, new = self.plan["from"], self.plan["to"]
        try:
            st.step("Download " + new["version"])
            folder = self.bundle(self.plan["windows_asset"])
            st.step("Download " + new["version"], "done", folder)
            previous = None
            st.step("Keep the running version for a rollback")
            if self.plan.get("previous_windows_asset"):
                try:
                    previous = self.bundle(self.plan["previous_windows_asset"])
                    st.step("Keep the running version for a rollback", "done", previous)
                except Exception as error:   # noqa: BLE001
                    st.step("Keep the running version for a rollback", "failed", "{}: no automatic rollback".format(_describe(error)))
            else:
                st.step("Keep the running version for a rollback", "failed",
                        "release {} has no windows zip: no automatic rollback".format(old.get("tag")))
            with open(os.path.join(UPDATE_DIR, "freeze-{}.txt".format(time.strftime("%Y%m%d-%H%M%S"))), "w") as f:
                f.write(subprocess.run([self.python, "-m", "pip", "freeze"], capture_output=True, text=True,
                                       timeout=300).stdout)
        except Exception as error:   # noqa: BLE001
            st.end("failed", "{}. Nothing was changed.".format(_describe(error)))
            return 1
        st.step("Stop the server")
        old_pid = self.server.get("pid")
        self.stop()
        st.step("Stop the server", "done")
        try:
            st.step("Install " + new["version"])
            self.install(folder)
            st.step("Install " + new["version"], "done")
            st.step("Start")
            self.start()
            st.step("Start", "done")
            st.step("Health check")
            answer = self.check_health(new["version"], self.plan.get("to_commit"), not_pid=old_pid)
            st.step("Health check", "done", "version {}".format(answer.get("version")))
            st.end("done", "Updated from {} to {}.".format(old["version"], new["version"]))
            return 0
        except Exception as error:   # noqa: BLE001
            message = _describe(error)
            st.log("Update failed: " + message)
            if previous is None:
                st.end("rollback_failed", "Update failed ({}) and there is no previous bundle to go back to: run "
                       "install.bat of the {} zip from {}".format(message, old["version"], self.plan.get("releases_url")))
                return 1
            st.data["rolled_back"] = True
            try:
                st.step("Roll back", "running", message)
                try:
                    self.server["pid"] = health(self.base_url(), timeout=5).get("pid")
                    self.stop()
                except UpdateError:
                    pass
                self.install(previous)
                self.start()
                health(self.base_url(), version=old["version"])
                st.step("Roll back", "done")
                st.end("rolled_back", "Update failed ({}). Rolled back to {} and restarted.".format(message, old["version"]))
            except Exception as again:   # noqa: BLE001
                st.end("rollback_failed", "Update failed ({}) and so did the rollback ({}). See {}".format(
                    message, _describe(again), LOG_FILE))
            return 1


def run_plan(path):
    with open(path, encoding="utf-8") as f:
        plan = json.load(f)
    kind = plan["kind"]
    update = {"git": GitUpdate, "windows": WindowsUpdate}[kind](plan)
    return update.run()


# ---- server side -----------------------------------------------------------------------------------------

def commands(install, release):
    """What to run by hand for an install the server does not update itself."""
    tag = release["tag"] if release else "<tag>"
    if install["kind"] == "docker":
        return ["git fetch --tags && git checkout {}".format(tag),
                "docker compose build && docker compose up -d"]
    if install["kind"] == "pip":
        return ['"{}" -m pip install --upgrade "idtrackerai_validator_server @ https://github.com/{}/archive/refs/tags/{}.tar.gz"'
                .format(install["python"], lab_paths.UPDATE_REPO, tag), "then restart the server"]
    if install["kind"] == "git":
        root = install["root"]
        return ["git -C {} checkout {} && git -C {} pull --ff-only".format(root, MAIN_BRANCH, root),
                "git -C {} submodule update".format(root),
                '"{}" -m pip install --no-deps -e {}'.format(install["python"], root),
                "cd {} && ./build_frontend.sh   # when the client changed".format(root),
                "then restart the server" + (" (systemctl --user restart {})".format(install["service"]["unit"])
                                             if install.get("service") else "")]
    if install["kind"] == "windows":
        return ["download {} from {}".format("the windows zip", release["url"] if release else "the releases page"),
                "unzip it and double click install.bat"]
    return []


def _asset(release, regex, version=None):
    for asset in (release or {}).get("assets", []):
        match = regex.match(asset["name"])
        if match and (version is None or match.group("version") == version):
            return asset
    return None


def register_updater(app, get_commit, get_version):
    """/api/health (public), /api/update/check, /api/update/admin, /api/update/apply, /api/update/status.
    get_commit / get_version: what the server read when it started (the checkout may have moved since)."""
    from flask import jsonify, request, session
    from idtrackerai_validator_server import auth

    auth.PUBLIC_PATHS.add("/api/health")
    last_check = {}

    def running():
        install = detect_install()
        commit = get_commit() or {}
        return install, {"version": get_version(), "commit": commit.get("commit"), "date": commit.get("date"),
                         "dirty": commit.get("dirty", False)}

    @app.route("/api/health")
    def update_health():
        return jsonify({"ok": True, "version": get_version(), "commit": (get_commit() or {}).get("commit"),
                        "pid": os.getpid()})

    def do_check():
        install, current = running()
        out = {"install": {k: install.get(k) for k in ("kind", "root", "python", "reason", "service", "install_dir")},
               "current": current, "admin": is_admin(request, session),
               "admin_password": bool(os.environ.get("VALIDATOR_ADMIN_PASSWORD")),
               "checked_at": time.time(), "last_update": read_status()}
        if out["last_update"]:
            out["last_update"].pop("log", None)
        release = latest_release()
        out["latest"] = {k: release[k] for k in ("tag", "version", "name", "url", "published_at", "source")}
        out["latest"]["summary"] = summarize(release["notes"])
        cmp = compare_versions(release["version"], current["version"]) if current["version"] else 1
        out["state"] = "update_available" if cmp > 0 else ("up_to_date" if cmp == 0 else "ahead")
        blockers = []
        if install["kind"] == "git":
            try:
                blockers += git_blockers(install["root"])
            except (OSError, subprocess.SubprocessError) as error:
                blockers.append("git does not work in {}: {}".format(install["root"], error))
            service = install.get("service")
            if service and not service.get("user"):
                blockers.append("Runs as the system unit {}: it cannot restart itself.".format(service["unit"]))
            if service and not shutil.which(lab_paths.SYSTEMD_RUN):
                blockers.append("{} not found: needed to update a systemd unit.".format(lab_paths.SYSTEMD_RUN))
        elif install["kind"] == "windows":
            if not _asset(release, WINDOWS_ASSET_RE, release["version"]):
                blockers.append("Release {} has no windows zip.".format(release["tag"]))
        else:
            blockers.append("This {} install is not updated from the app: run the commands below.".format(install["kind"]))
        if update_running():
            blockers.append("An update is running already.")
        out["blockers"] = blockers
        out["can_apply"] = out["state"] == "update_available" and not blockers
        out["commands"] = commands(install, release)
        out["busy"] = busy()
        out["frontend_asset"] = bool(_asset(release, FRONTEND_ASSET_RE, release["version"]))
        last_check.update(time=time.time(), release=release, install=install, current=current)
        return out

    @app.route("/api/update/check")
    def update_check():
        try:
            return jsonify(do_check())
        except UpdateError as error:
            install, current = running()
            return jsonify({"error": str(error), "current": current,
                            "install": {k: install.get(k) for k in ("kind", "root", "reason")},
                            "admin": is_admin(request, session)}), 502

    @app.route("/api/update/admin", methods=["POST"])
    def update_admin():
        password = os.environ.get("VALIDATOR_ADMIN_PASSWORD")
        given = (request.get_json(silent=True) or {}).get("password") or ""
        if not password:
            return jsonify({"error": "no admin password is set on the server (VALIDATOR_ADMIN_PASSWORD)"}), 400
        if not hmac.compare_digest(given.encode(), password.encode()):
            time.sleep(1)
            return jsonify({"error": "wrong admin password"}), 403
        session["update_admin"] = True
        return jsonify({"admin": True})

    @app.route("/api/update/status")
    def update_status():
        status = read_status()
        if status is not None:
            status["alive"] = _pid_alive(status.get("helper_pid"))
        _, current = running()
        return jsonify({"status": status, "current": current})

    @app.route("/api/update/apply", methods=["POST"])
    def update_apply():
        if not is_admin(request, session):
            return jsonify({"error": "Only an admin can update the server: it restarts it for everyone. "
                                     "Ask your admin."}), 403
        body = request.get_json(silent=True) or {}
        if update_running():
            return jsonify({"error": "An update is running already."}), 409
        if not last_check or time.time() - last_check["time"] > CHECK_MAX_AGE_S:
            try:
                do_check()
            except UpdateError as error:
                return jsonify({"error": str(error)}), 502
        release, install, current = last_check["release"], last_check["install"], last_check["current"]
        if body.get("version") and body["version"] != release["version"]:
            return jsonify({"error": "The latest release is {} now, not {}: check again.".format(
                release["version"], body["version"])}), 409
        if compare_versions(release["version"], current["version"] or "0") <= 0:
            return jsonify({"error": "Already at {} (latest {}).".format(current["version"], release["version"])}), 400
        if install["kind"] not in ("git", "windows"):
            return jsonify({"error": "This {} install is not updated from the app.".format(install["kind"]),
                            "commands": commands(install, release)}), 400
        if install["kind"] == "git":
            blockers = git_blockers(install["root"])
            if blockers:
                return jsonify({"error": " ".join(blockers)}), 409
        jobs = busy()["jobs"]
        if jobs and not body.get("force"):
            return jsonify({"error": "{} job(s) running would be stopped by the restart: {}".format(
                len(jobs), ", ".join(j["title"] for j in jobs)), "jobs": jobs}), 409
        try:
            plan = make_plan(install, current, release, app)
        except UpdateError as error:
            return jsonify({"error": str(error)}), 400
        start_helper(plan, install)
        return jsonify({"started": True, "id": plan["id"], "to": plan["to"]}), 202


def make_plan(install, current, release, app):
    plan = {
        "id": time.strftime("%Y%m%d-%H%M%S"),
        "kind": install["kind"],
        "install": install,
        "from": {"version": current["version"], "commit": current["commit"], "tag": "v" + (current["version"] or "")},
        "to": {"version": release["version"], "tag": release["tag"]},
        "server": {"pid": os.getpid(), "argv": list(getattr(sys, "orig_argv", None) or [sys.executable] + sys.argv),
                   "cwd": os.getcwd(), "port": app.config.get("VALIDATOR_PORT") or int(os.environ.get("BACKEND_PORT", 5000)),
                   "host": app.config.get("VALIDATOR_HOST") or "127.0.0.1",
                   "scheme": "https" if app.config.get("SESSION_COOKIE_SECURE") else "http"},
        "frontend_asset": _asset(release, FRONTEND_ASSET_RE, release["version"]),
        "test_fail_health": os.environ.get("VALIDATOR_UPDATE_TEST_FAIL_HEALTH") == "1",
        "releases_url": "{}/{}/releases".format(lab_paths.UPDATE_WEB, lab_paths.UPDATE_REPO),
    }
    if install["kind"] == "windows":
        asset = _asset(release, WINDOWS_ASSET_RE, release["version"])
        if asset is None:
            raise UpdateError("Release {} has no windows zip".format(release["tag"]))
        plan["windows_asset"] = asset
        plan["to_commit"] = WINDOWS_ASSET_RE.match(asset["name"]).group("commit")
        try:
            previous = latest_release(tag=plan["from"]["tag"])
            plan["previous_windows_asset"] = _asset(previous, WINDOWS_ASSET_RE, current["version"])
        except UpdateError as error:
            logger.warning("No previous bundle for a rollback: %s", error)
            plan["previous_windows_asset"] = None
    return plan


def start_helper(plan, install):
    """Start the helper outside the server: in a transient user unit when the server is a user unit
    (systemctl restart stops every process of the unit), else as a detached process."""
    os.makedirs(UPDATE_DIR, exist_ok=True)
    path = os.path.join(UPDATE_DIR, "plan-{}.json".format(plan["id"]))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=1)
    # status.json at once: the client sees "running" before the helper has started
    with open(STATUS_FILE + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"id": plan["id"], "kind": plan["kind"], "state": "running", "from": plan["from"], "to": plan["to"],
                   "steps": [], "log": ["Starting the update helper"], "message": None, "started": time.time(),
                   "updated": time.time(), "helper_pid": None}, f)
    os.replace(STATUS_FILE + ".tmp", STATUS_FILE)
    cmd = [sys.executable, "-m", "idtrackerai_validator_server.updater", "run", path]
    service = install.get("service")
    log = open(LOG_FILE, "ab")
    if service and service.get("user"):
        keep = ("PATH", "HOME", "LANG", "LC_ALL", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "NPM", "SYSTEMCTL",
                "SYSTEMD_RUN", "GITHUB_TOKEN", "PYTHONPATH")
        setenv = ["--setenv={}={}".format(k, v) for k, v in os.environ.items()
                  if k in keep or k.startswith("VALIDATOR_UPDATE") or k.startswith("VALIDATOR_LAB")]
        cmd = [lab_paths.SYSTEMD_RUN, "--user", "--collect", "--quiet", "--unit",
               "idtrackerai-validator-update-{}".format(plan["id"]), "--working-directory", UPDATE_DIR,
               *setenv, *cmd]
        subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
        return
    kwargs = new_session()
    if IS_WINDOWS:
        kwargs["creationflags"] |= getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    try:
        subprocess.Popen(cmd, cwd=UPDATE_DIR, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **kwargs)
    except OSError:
        if not IS_WINDOWS:
            raise
        kwargs["creationflags"] &= ~getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)   # the job forbids it
        subprocess.Popen(cmd, cwd=UPDATE_DIR, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **kwargs)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "run":
        logging.basicConfig(level=logging.INFO)
        return run_plan(argv[1])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
