"""updater.py: version comparison, release pages, install detection, the refused updates.

    python -m unittest tests.test_updater
"""
import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from flask import Flask

from idtrackerai_validator_server import updater


def sh(cwd, *args):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def make_repo(path, branch="master"):
    os.makedirs(path, exist_ok=True)
    sh(path, "git", "init", "-q", "-b", branch)
    sh(path, "git", "config", "user.email", "t@example.org")
    sh(path, "git", "config", "user.name", "t")
    return commit(path, "a.txt", "1")


def commit(path, name, text):
    with open(os.path.join(path, name), "w") as f:
        f.write(text)
    sh(path, "git", "add", name)
    sh(path, "git", "commit", "-q", "-m", text)
    return sh(path, "git", "rev-parse", "HEAD")


class Versions(unittest.TestCase):
    def test_order(self):
        c = updater.compare_versions
        self.assertEqual(c("v1.5.0", "1.5.0"), 0)
        self.assertEqual(c("1.5", "1.5.0"), 0)
        self.assertEqual(c("1.10.0", "1.9.9"), 1)
        self.assertEqual(c("1.4.1", "1.5.0"), -1)
        self.assertEqual(c("2.0.0", "1.99.99"), 1)
        self.assertEqual(c("1.6.0rc1", "1.6.0"), -1)      # a pre-release before its release
        self.assertEqual(c("1.6.0-beta.2", "1.5.9"), 1)
        self.assertEqual(c("1.5.0+local", "1.5.0"), 0)    # build metadata
        with self.assertRaises(ValueError):
            updater.parse_version("latest")

    def test_summary(self):
        notes = "\n".join("line {}".format(i) for i in range(500))
        short = updater.summarize(notes, limit=100)
        self.assertLessEqual(len(short), 102)
        self.assertTrue(short.endswith("…"))
        self.assertEqual(updater.summarize("short"), "short")

    def test_html_to_text(self):
        text = updater.html_to_text("<h2>What&#39;s new</h2><ul><li><p><strong>A</strong> b</p></li><li>c</li></ul>")
        self.assertIn("## What's new", text)
        self.assertIn("- A b", text)
        self.assertIn("- c", text)


class Requirements(unittest.TestCase):
    def test_filter(self):
        reqs = ["flask>=2.2.5", "numpy>=1.21.6,<2", "flyhostel[idtrackerai] @ https://github.com/x/flyhostel/archive/a.tar.gz",
                "imgstore-shaliulab @ file:///home/x/imgstore", "sleap-io==0.9.5; extra == \"sleap\"",
                "tomli; python_version < '3.0'", "scipy", "newpackage>=1", "idtrackerai-shaliulab>=4"]
        installed = {"flask": "3.0.0", "numpy": "2.1.0", "scipy": "1.11"}
        self.assertEqual(updater.filter_requirements(reqs, installed), ["numpy<2,>=1.21.6", "newpackage>=1"])


class Pins(unittest.TestCase):
    def test_read_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_repo(tmp)
            commit(tmp, "setup.py", 'PINS = {\n    "idtrackerai-validator-client": "' + "a" * 40 + '",\n'
                   '    "flyhostel": "' + "b" * 40 + '",\n    "flyhostel/libraries/imgstore": "' + "c" * 40 + '",\n}\n'
                   "setup(version='1.6.0')\n")
            pins = updater.read_pins(tmp, "HEAD")
            self.assertEqual(pins["flyhostel"], "b" * 40)
            self.assertEqual(pins["flyhostel/libraries/imgstore"], "c" * 40)
            self.assertEqual(updater.source_version_at(tmp, "HEAD"), "1.6.0")
            self.assertEqual(updater.dep_pin("imgstore-shaliulab", pins, {}), "c" * 40)


class Checkout(unittest.TestCase):
    def test_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_repo(tmp)
            self.assertEqual(updater.git_blockers(tmp), [])
            with open(os.path.join(tmp, "untracked.txt"), "w") as f:
                f.write("x")
            self.assertEqual(updater.git_blockers(tmp), [], "untracked files do not block")
            with open(os.path.join(tmp, "a.txt"), "w") as f:
                f.write("changed")
            blockers = updater.git_blockers(tmp)
            self.assertEqual(len(blockers), 1)
            self.assertIn("uncommitted changes (a.txt)", blockers[0])
            sh(tmp, "git", "checkout", "-q", "a.txt")
            sh(tmp, "git", "checkout", "-q", "-b", "feature")
            self.assertIn("on branch feature", updater.git_blockers(tmp)[0])
            sh(tmp, "git", "checkout", "-q", "--detach")
            self.assertIn("detached", updater.git_blockers(tmp)[0])

    def test_plan_dep(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = make_repo(tmp)
            second = commit(tmp, "a.txt", "2")
            third = commit(tmp, "a.txt", "3")
            dep = {"name": "flyhostel", "editable": True, "path": tmp}
            plan = updater.plan_dep
            self.assertEqual(plan(dep, second)["action"], "keep")            # ahead of the pin
            self.assertEqual(plan(dep, third)["action"], "keep")             # at it
            self.assertEqual(plan(dep, None)["action"], "keep")              # no pin
            self.assertEqual(plan(dict(dep, editable=False), third)["action"], "keep")
            sh(tmp, "git", "reset", "-q", "--hard", first)
            self.assertEqual(plan(dep, third)["action"], "ff")               # behind, clean
            with open(os.path.join(tmp, "a.txt"), "w") as f:
                f.write("dirty")
            self.assertEqual(plan(dep, third)["action"], "refuse")           # behind, dirty
            sh(tmp, "git", "checkout", "-q", "a.txt")
            commit(tmp, "b.txt", "diverged")
            out = plan(dep, third)
            self.assertEqual(out["action"], "refuse")                        # diverged
            self.assertIn("neither contains nor is behind", out["reason"])
            self.assertEqual(plan(dep, "f" * 40)["action"], "refuse")        # unknown commit


class Install(unittest.TestCase):
    def test_kinds(self):
        root = updater.package_root()
        dist = mock.Mock()
        dist.read_text.return_value = json.dumps({"url": "file://" + root, "dir_info": {"editable": True}})
        with mock.patch.object(updater.importlib.metadata, "distribution", return_value=dist), \
                mock.patch.object(updater.os.path, "exists", side_effect=lambda p: False if p == "/.dockerenv" else os.path.lexists(p)), \
                mock.patch.object(updater, "_git_ok", return_value=True), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VALIDATOR_INSTALL", None)
            self.assertEqual(updater.detect_install()["kind"], "git")
            dist.read_text.return_value = json.dumps({"url": "file:///somewhere/else", "dir_info": {}})
            self.assertEqual(updater.detect_install()["kind"], "pip")
            dist.read_text.return_value = json.dumps({"url": "file://" + root, "dir_info": {"editable": True}})
            with mock.patch.object(updater, "_git_ok", return_value=False):
                self.assertEqual(updater.detect_install()["kind"], "pip")    # editable, no git checkout
        with mock.patch.object(updater.os.path, "exists", side_effect=lambda p: p == "/.dockerenv" or os.path.lexists(p)):
            os.environ.pop("VALIDATOR_INSTALL", None)
            self.assertEqual(updater.detect_install()["kind"], "docker")
        with mock.patch.dict(os.environ, {"VALIDATOR_INSTALL": "windows"}):
            self.assertEqual(updater.detect_install()["kind"], "windows")

    def test_systemd_unit(self):
        cgroup = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/idtrackerai-viewer.service\n"
        with mock.patch.dict(os.environ, {"INVOCATION_ID": "x"}), \
                mock.patch("builtins.open", mock.mock_open(read_data=cgroup)):
            self.assertEqual(updater.systemd_unit(), {"unit": "idtrackerai-viewer.service", "user": True})
        with mock.patch.dict(os.environ, {"INVOCATION_ID": "x"}), \
                mock.patch("builtins.open", mock.mock_open(read_data="0::/system.slice/viewer.service\n")):
            self.assertEqual(updater.systemd_unit(), {"unit": "viewer.service", "user": False})
        env = dict(os.environ)
        env.pop("INVOCATION_ID", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(updater.systemd_unit())


RELEASE = {"tag": "v9.0.0", "version": "9.0.0", "name": "v9.0.0", "notes": "## New\n- things", "url": "https://x/v9.0.0",
           "published_at": None, "assets": [], "source": "api"}


class Endpoints(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        make_repo(self.root)
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "test"
        updater.register_updater(self.app, get_commit=lambda: {"commit": "abc1234", "date": None, "dirty": False},
                                 get_version=lambda: "1.5.0")
        install = {"kind": "git", "root": self.root, "python": "python", "reason": "test", "service": None}
        self.patches = [mock.patch.object(updater, "latest_release", return_value=RELEASE),
                        mock.patch.object(updater, "detect_install", return_value=install),
                        mock.patch.object(updater, "start_helper"),
                        mock.patch.object(updater, "STATUS_FILE", os.path.join(self.root, "status.json")),
                        mock.patch("idtrackerai_validator_server.jobs.running_jobs", return_value=[]),
                        mock.patch.object(updater, "cgroup_processes", return_value=[])]
        for p in self.patches:
            p.start()
        self.client = self.app.test_client()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def remote(self):
        return {"REMOTE_ADDR": "10.1.2.3"}

    def test_check(self):
        data = self.client.get("/api/update/check").get_json()
        self.assertEqual(data["state"], "update_available")
        self.assertTrue(data["can_apply"])
        self.assertTrue(data["admin"])
        self.assertEqual(self.client.get("/api/update/check", environ_base=self.remote()).get_json()["admin"], False)
        health = self.client.get("/api/health").get_json()
        self.assertEqual((health["version"], health["commit"]), ("1.5.0", "abc1234"))

    def test_not_admin(self):
        response = self.client.post("/api/update/apply", json={}, environ_base=self.remote())
        self.assertEqual(response.status_code, 403)
        self.assertIn("ask your admin", response.get_json()["error"])
        updater.start_helper.assert_not_called()

    def test_loopback_only(self):
        # no password path: a remote browser can never apply, whatever the environment says
        with mock.patch.dict(os.environ, {"VALIDATOR_ADMIN_PASSWORD": "s3cret"}):
            self.assertEqual(self.client.post("/api/update/admin", json={"password": "s3cret"},
                                              environ_base=self.remote()).status_code, 404)
            self.assertEqual(self.client.post("/api/update/apply", json={}, environ_base=self.remote()).status_code, 403)
            self.assertNotIn("admin_password", self.client.get("/api/update/check").get_json())
        self.assertEqual(self.client.post("/api/update/apply", json={}, environ_base={"REMOTE_ADDR": "::1"}).status_code, 202)
        with mock.patch.dict(os.environ, {"VALIDATOR_ADMIN_LOCALHOST": "0"}):
            self.assertEqual(self.client.post("/api/update/apply", json={}).status_code, 403)

    def test_cgroup_process_blocks(self):
        proc = {"pid": 4242, "name": "preprocess-sample", "command": "/env/bin/python3.7 /env/bin/preprocess-sample --store-path x"}
        with mock.patch.object(updater, "cgroup_processes", return_value=[proc]):
            check = self.client.get("/api/update/check").get_json()
            self.assertEqual(check["busy"]["jobs"][0]["title"], "preprocess-sample (pid 4242)")
            response = self.client.post("/api/update/apply", json={})
            self.assertEqual(response.status_code, 409)
            self.assertIn("preprocess-sample (pid 4242)", response.get_json()["error"])
            updater.start_helper.assert_not_called()
            self.assertEqual(self.client.post("/api/update/apply", json={"force": True}).status_code, 202)

    def test_dirty(self):
        with open(os.path.join(self.root, "a.txt"), "w") as f:
            f.write("changed")
        self.assertFalse(self.client.get("/api/update/check").get_json()["can_apply"])
        response = self.client.post("/api/update/apply", json={})
        self.assertEqual(response.status_code, 409)
        self.assertIn("uncommitted changes", response.get_json()["error"])
        updater.start_helper.assert_not_called()

    def test_jobs_running(self):
        with mock.patch("idtrackerai_validator_server.jobs.running_jobs",
                        return_value=[{"id": "3", "title": "Pre-download FlyHostel1_16X"}]):
            response = self.client.post("/api/update/apply", json={})
            self.assertEqual(response.status_code, 409)
            self.assertIn("Pre-download FlyHostel1_16X", response.get_json()["error"])
            updater.start_helper.assert_not_called()
            self.assertEqual(self.client.post("/api/update/apply", json={"force": True}).status_code, 202)
            updater.start_helper.assert_called_once()

    def test_stale_version_and_up_to_date(self):
        self.assertEqual(self.client.post("/api/update/apply", json={"version": "8.0.0"}).status_code, 409)
        with mock.patch.object(updater, "latest_release", return_value=dict(RELEASE, tag="v1.5.0", version="1.5.0")):
            self.client.get("/api/update/check")
            self.assertEqual(self.client.post("/api/update/apply", json={}).status_code, 400)

    def test_pip_install(self):
        install = {"kind": "pip", "root": "/x", "python": "/env/bin/python", "reason": "pip", "service": None}
        with mock.patch.object(updater, "detect_install", return_value=install):
            data = self.client.get("/api/update/check").get_json()
            self.assertFalse(data["can_apply"])
            self.assertIn("pip install --upgrade", data["commands"][0])
            self.assertIn("v9.0.0", data["commands"][0])
            self.assertEqual(self.client.post("/api/update/apply", json={}).status_code, 400)

    def test_offline(self):
        with mock.patch.object(updater, "latest_release", side_effect=updater.UpdateError("Could not reach GitHub")):
            response = self.client.get("/api/update/check")
            self.assertEqual(response.status_code, 502)
            self.assertIn("Could not reach GitHub", response.get_json()["error"])


class Cgroup(unittest.TestCase):
    def test_processes_of_the_unit(self):
        """Every process of the unit's cgroup (and below) but the server and multiprocessing's resource_tracker."""
        import sys
        children = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "job"]),
                    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)  # multiprocessing.resource_tracker"])]
        try:
            with tempfile.TemporaryDirectory() as root:
                unit = os.path.join(root, "user.slice", "app.slice", "viewer.service")
                os.makedirs(os.path.join(unit, "sub"))
                with open(os.path.join(unit, "cgroup.procs"), "w") as f:
                    f.write("{}\n{}\n".format(os.getpid(), children[0].pid))
                with open(os.path.join(unit, "sub", "cgroup.procs"), "w") as f:
                    f.write("{}\n999999999\n".format(children[1].pid))     # a pid already gone is skipped
                found = updater.cgroup_processes(cgroup="/user.slice/app.slice/viewer.service", root=root,
                                                 self_pid=os.getpid())
            self.assertEqual([p["pid"] for p in found], [children[0].pid])
            self.assertIn("sleep(60)", found[0]["command"])
        finally:
            for child in children:
                child.kill()
                child.wait()

    def test_not_under_systemd(self):
        with mock.patch.object(updater, "systemd_unit", return_value=None):
            self.assertEqual(updater.cgroup_processes(), [])

    def test_own_cgroup(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("0::/user.slice/user-1000.slice/user@1000.service/app.slice/idtrackerai-viewer.service\n")
        try:
            self.assertEqual(updater.own_cgroup(f.name),
                             "/user.slice/user-1000.slice/user@1000.service/app.slice/idtrackerai-viewer.service")
        finally:
            os.unlink(f.name)


class WebFallback(unittest.TestCase):
    def test_rate_limited_api_reads_the_pages(self):
        class Response:
            def __init__(self, status=200, text="", headers=None, data=None):
                self.status_code, self.text, self.headers, self._data = status, text, headers or {}, data
                self.ok = status < 400

            def json(self):
                return self._data

            def raise_for_status(self):
                if not self.ok:
                    import requests
                    raise requests.HTTPError(str(self.status_code))

        atom = ('<feed><entry><link rel="alternate" href="https://github.com/r/releases/tag/v1.5.0"/>'
                '<updated>2026-11-24T18:38:02Z</updated><title>v1.5.0: Annotate pre-download</title>'
                '<content type="html">&lt;h2&gt;New&lt;/h2&gt;&lt;p&gt;text&lt;/p&gt;</content></entry></feed>')
        assets = ('<ul><li class="Box-row"><a href="/r/releases/download/v1.5.0/idtrackerai-validator-windows-1.5.0-6982294.zip">'
                  'x</a><span>sha256:' + "ab" * 32 + '</span></li></ul>')

        def get(url, **kwargs):
            if "api." in url:
                return Response(403, "API rate limit exceeded", {"X-RateLimit-Remaining": "0"})
            if url.endswith("/releases/latest"):
                return Response(302, headers={"Location": "https://github.com/r/releases/tag/v1.5.0"})
            if url.endswith(".atom"):
                return Response(200, atom)
            if "expanded_assets" in url:
                return Response(200, assets)
            return Response(404)

        session = mock.Mock()
        session.get.side_effect = get
        with mock.patch.object(updater, "_session", return_value=session):
            release = updater.latest_release()
        self.assertEqual((release["tag"], release["version"], release["source"]), ("v1.5.0", "1.5.0", "web"))
        self.assertEqual(release["name"], "v1.5.0: Annotate pre-download")
        self.assertIn("## New", release["notes"])
        self.assertEqual(release["assets"][0]["digest"], "sha256:" + "ab" * 32)
        self.assertTrue(updater.WINDOWS_ASSET_RE.match(release["assets"][0]["name"]))


if __name__ == "__main__":
    unittest.main()
