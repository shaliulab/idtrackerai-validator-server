"""pipeline_csv.py on a copy of idtrackerai.csv: untouched lines keep their bytes, duplicates refused,
stale previews refused, lock honoured, backups / audit log, permissions.

    python -m unittest tests.test_pipeline_csv

Uses the real file read-only (copied to a temporary folder) when it exists, else a sample with the same quirks.
"""
import fcntl
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

from flask import Flask

from idtrackerai_validator_server import lab_paths, pipeline_csv

REAL = os.path.join(lab_paths.NEXTFLOW_PIPELINES, "idtrackerai_v3", "idtrackerai.csv")
SAMPLE = (
    "/V/FlyHostel1/3X/2025-06-21_15-00-00,2025-07-04,OK\n"
    "/V/FlyHostel3/1X/2026-05-18_14-00-00,cv3\n"
    "\n"
    "/V/FlyHostel1/4X/2026-10-13_17-00-00/arenas/1,2026-10-17,OK\n"
    "/V/FlyHostel1/1X/2026-11-23_17-00-00,NEXT,OK\n"
)
EXP = "FlyHostel2/6X/2026-11-24_17-00-00"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.root = os.path.join(self.tmp, "videos")
        self.csv = os.path.join(self.tmp, "idtrackerai.csv")
        if os.path.exists(REAL):
            shutil.copyfile(REAL, self.csv)
        else:
            with open(self.csv, "w") as fh:
                fh.write(SAMPLE.replace("/V", self.root))
        with open(self.csv, "rb") as fh:
            self.original = fh.read()
        os.chmod(self.csv, 0o640)
        for name, value in {"IDTRACKERAI_CSV": self.csv,
                            "IDTRACKERAI_CSV_BACKUPS": os.path.join(self.tmp, "backups"),
                            "IDTRACKERAI_CSV_LOG": os.path.join(self.tmp, "log.jsonl"),
                            "PIPELINE_USERS": None}.items():
            patcher = mock.patch.object(lab_paths, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {"FLYHOSTEL_VIDEOS": self.root, "VALIDATOR_USERNAME": "owner"})
        env.start()
        self.addCleanup(env.stop)
        self.folder = self.make_experiment(EXP)

    def make_experiment(self, experiment, complete=True):
        folder = os.path.join(self.root, experiment)
        os.makedirs(folder)
        date = experiment.split("/")[2]
        files = ["metadata.yaml", "index.db", "landmarks.toml", "a2A3840-45umPRO.pfs", date + ".toml", date + ".conf"]
        for name in files if complete else files[:2]:
            open(os.path.join(folder, name), "w").close()
        time.sleep(0.01)
        with open(os.path.join(folder, "preprocess_sample.log"), "w") as fh:
            fh.write("...\nTotal problematic frames = 1\nSaved to x\n")
        with open(os.path.join(folder, "preprocess_sample.csv"), "w") as fh:
            fh.write("frame_number,chunk,video,frame_idx,n,x,y\n100,5,000005.mp4,0,2,1,1\n")
        for c in range(10):
            open(os.path.join(folder, f"{c:06d}.npz"), "w").close()
        return folder

    def read(self):
        with open(self.csv, "rb") as fh:
            return fh.read()

    def sig(self):
        return pipeline_csv.sig_of(self.read())


class Edits(Base):
    def test_add_appends_one_line_and_keeps_every_byte(self):
        result = pipeline_csv.write({"op": "add", "experiment": EXP, "mark": "1", "note": "OK"}, self.sig())
        data = self.read()
        self.assertTrue(data.startswith(self.original if self.original.endswith(b"\n") else self.original + b"\n"))
        self.assertEqual(data[len(data) - len(f"{self.root}/{EXP},1,OK\n"):], f"{self.root}/{EXP},1,OK\n".encode())
        self.assertEqual(result["diff"][0]["old"], None)
        self.assertEqual(os.stat(self.csv).st_mode & 0o777, 0o640)
        with open(result["backup"], "rb") as fh:
            self.assertEqual(fh.read(), self.original)
        with open(lab_paths.IDTRACKERAI_CSV_LOG) as fh:
            entry = json.loads(fh.readline())
        self.assertEqual(entry["op"], "add")
        self.assertEqual(entry["diff"][0]["new"], f"{self.root}/{EXP},1,OK")

    def test_add_then_remove_round_trips(self):
        pipeline_csv.write({"op": "add", "experiment": EXP, "mark": "1", "note": "OK"}, self.sig())
        row = pipeline_csv.rows_of(self.read().decode(), f"{self.root}/{EXP}")[0]
        pipeline_csv.write({"op": "remove", "experiment": EXP, "line": row["line"], "expect": row["text"]}, self.sig())
        expected = self.original if self.original.endswith(b"\n") else self.original + b"\n"
        self.assertEqual(self.read(), expected)

    def test_duplicate_refused_trailing_slash_too(self):
        with open(self.csv, "ab") as fh:
            fh.write(f"{self.root}/{EXP}/ ,NEXT,OK\n".encode())
        with self.assertRaises(pipeline_csv.Conflict):
            pipeline_csv.write({"op": "add", "experiment": EXP, "mark": "1"}, self.sig())

    def test_update_changes_only_that_line(self):
        with open(self.csv, "ab") as fh:
            fh.write(f"{self.root}/{EXP},NEXT,OK\n".encode())
        before = self.read()
        row = pipeline_csv.rows_of(before.decode(), f"{self.root}/{EXP}")[0]
        pipeline_csv.write({"op": "update", "experiment": EXP, "line": row["line"], "expect": row["text"],
                            "mark": "1", "note": "OK"}, self.sig())
        after = self.read()
        self.assertEqual(after, before.replace(f"{EXP},NEXT,OK".encode(), f"{EXP},1,OK".encode()))

    def test_stale_sig_and_stale_line_refused(self):
        sig = self.sig()
        with open(self.csv, "ab") as fh:
            fh.write(b"/x,LATER\n")
        with self.assertRaises(pipeline_csv.Conflict):
            pipeline_csv.write({"op": "add", "experiment": EXP, "mark": "1"}, sig)
        with self.assertRaises(pipeline_csv.Conflict):
            pipeline_csv.write({"op": "remove", "experiment": EXP, "line": 1, "expect": "nope"}, self.sig())
        self.assertTrue(self.read().endswith(b"/x,LATER\n"))

    def test_bad_values(self):
        for mark in ("", "1,2", "a\nb"):
            with self.assertRaises(ValueError):
                pipeline_csv.write({"op": "add", "experiment": EXP, "mark": mark}, self.sig())
        with self.assertRaises(ValueError):
            pipeline_csv.write({"op": "add", "experiment": "../etc", "mark": "1"}, self.sig())

    def test_no_trailing_newline(self):
        with open(self.csv, "wb") as fh:
            fh.write(b"/a,1,OK")
        pipeline_csv.write({"op": "add", "experiment": EXP, "mark": "NEXT", "note": ""}, self.sig())
        self.assertEqual(self.read(), f"/a,1,OK\n{self.root}/{EXP},NEXT\n".encode())

    def test_waits_for_the_lock(self):
        holder = open(self.csv, "a")
        fcntl.flock(holder, fcntl.LOCK_EX)
        sig = self.sig()
        done = []
        thread = threading.Thread(target=lambda: done.append(
            pipeline_csv.write({"op": "add", "experiment": EXP, "mark": "1"}, sig)))
        thread.start()
        time.sleep(0.3)
        self.assertFalse(done)                      # blocked while someone else holds it
        self.assertEqual(self.read(), self.original)
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()
        thread.join(5)
        self.assertTrue(done)

    def test_checks(self):
        found, _ = pipeline_csv.checks(EXP, "1", root=self.root, text="")
        self.assertFalse([c for c in found if c["level"] == "error"], found)
        bad = "FlyHostel3/2X/2026-11-24_17-00-00"
        self.make_experiment(bad, complete=False)
        found, _ = pipeline_csv.checks(bad, "1", root=self.root, text="")
        self.assertTrue([c for c in found if c["level"] == "error"])
        found, _ = pipeline_csv.checks(bad, "NEXT", root=self.root, text="")
        self.assertFalse([c for c in found if c["level"] == "error"])    # not taken by a run: warnings only


class Routes(Base):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.config["SECRET_KEY"] = "x"
        pipeline_csv.register_pipeline_csv(app)
        self.client = app.test_client()

    def post(self, path, body, addr="10.0.0.5", user=None):
        if user:
            with self.client.session_transaction() as s:
                s["user"] = user
        return self.client.post(path, json=body, environ_base={"REMOTE_ADDR": addr})

    def test_preview_apply_flow_and_permissions(self):
        info = self.client.get("/api/collector/idtrackerai_csv", query_string={"experiment": EXP}).get_json()
        self.assertEqual(info["rows"], [])
        body = {"op": "add", "experiment": EXP, "mark": "1", "note": "OK"}
        preview = self.post("/api/collector/idtrackerai_csv/preview", body).get_json()
        self.assertEqual(preview["diff"][0]["new"], f"{self.root}/{EXP},1,OK")
        self.assertEqual(self.read(), self.original)
        r = self.post("/api/collector/idtrackerai_csv/apply", {**body, "sig": preview["sig"]})
        self.assertEqual(r.status_code, 403)                                   # LAN, no login
        r = self.post("/api/collector/idtrackerai_csv/apply", {**body, "sig": preview["sig"]}, user="guest")
        self.assertEqual(r.status_code, 403)
        r = self.post("/api/collector/idtrackerai_csv/apply", {**body, "sig": preview["sig"]}, user="owner")
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertEqual(len(r.get_json()["rows"]), 1)
        r = self.post("/api/collector/idtrackerai_csv/apply", {**body, "sig": r.get_json()["sig"]}, addr="127.0.0.1")
        self.assertEqual(r.status_code, 409)                                   # already listed

    def test_blocking_check(self):
        bad = "FlyHostel3/2X/2026-11-24_17-00-00"
        self.make_experiment(bad, complete=False)
        body = {"op": "add", "experiment": bad, "mark": "1", "sig": self.sig()}
        r = self.post("/api/collector/idtrackerai_csv/apply", body, addr="127.0.0.1")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.read(), self.original)


if __name__ == "__main__":
    unittest.main()
