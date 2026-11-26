"""The Annotate tab's options (continuity version, easy-scene method) and "Accept all easy scenes" with its undo, on a
synthetic experiment in a temporary folder: python -m unittest tests.test_accept_easy"""
import json
import os
import sqlite3
import tempfile
import time
import unittest

from flask import Flask

from idtrackerai_validator_server import box_annotations as ba, continuity, scene_scorer
from idtrackerai_validator_server.jobs import _jobs
from tests.test_continuity_v2 import make_db

EXPERIMENT = "FlyHostel9/2X/2026-01-01_12-00-00"
FLAT = "FlyHostel9_2X_2026-01-01_12-00-00"
SCENE_QC = ("index,all_valid_ids,min_distance,max_velocity,gap_n_frames,gap_distance,maintains_id,between_chunks,"
            "broken,length,n_failed_fragments,pass,scene_start\n"
            "0,False,0,0,0,0,True,0,0,10,0,False,30\n"
            "1,True,0,0,0,0,True,0,0,5,0,True,60\n")


def wait(job_id, timeout=30):
    t = time.time()
    while _jobs[str(job_id)].status == "running":
        if time.time() - t > timeout:
            raise TimeoutError(job_id)
        time.sleep(0.05)
    return _jobs[str(job_id)]


class AcceptEasyTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.basedir = os.path.join(self.tmp.name, EXPERIMENT)
        os.makedirs(os.path.join(self.basedir, "flyhostel", "validation"))
        self.db = os.path.join(self.basedir, FLAT + ".db")
        frames = {fn: [(100 + fn, 100, 1, 1), (500, 500, 2, 2)] for fn in range(30)}
        frames.update({fn: [(130, 100, 0, 3)] for fn in range(30, 40)})    # fly 2 not detected
        frames.update({fn: [(130, 100, 1, 3), (500, 500, 2, 4)] for fn in range(40, 80)})
        make_db(self.db, frames)
        with open(os.path.join(self.basedir, "flyhostel", "validation", "scene_qc.csv"), "w") as handle:
            handle.write(SCENE_QC)
        self.state = {"experiment": EXPERIMENT, "basedir": self.basedir, "dbfile": self.db, "chunksize": 1000,
                      "n_animals": 2, "contours": lambda fn: [], "size": lambda: (2100, 2100)}
        self.loaded = self.state
        continuity._runs.clear()
        continuity._notches.clear()
        continuity._anchors.clear()
        ba._easy_cache.clear()
        ba._scene_cache.clear()
        self._dirs = (ba.EASY_CACHE_DIR, scene_scorer.MODEL_PATH)
        ba.EASY_CACHE_DIR = os.path.join(self.tmp.name, "easy")
        scene_scorer.MODEL_PATH = os.path.join(self.tmp.name, "scorer", "model.pkl")
        app = Flask(__name__)
        app.secret_key = "test"
        ba.register_box_annotations(app, get_state=lambda: self.loaded)
        self.client = app.test_client()
        with self.client.session_transaction() as sess:
            sess["user"] = "tester"

    def tearDown(self):
        ba.EASY_CACHE_DIR, scene_scorer.MODEL_PATH = self._dirs
        self.tmp.cleanup()

    def sidecar(self):
        return sqlite3.connect(ba.sidecar_path(self.basedir))

    def test_continuity_version_per_request(self):
        v1 = self.client.get("/api/annotate/frame/35?continuity=1").get_json()
        v2 = self.client.get("/api/annotate/frame/35?continuity=2").get_json()
        self.assertEqual(v1["source"], "continuity")
        self.assertEqual(v2["source"], "continuity2")
        self.assertEqual(sorted(b["label"] for b in v1["boxes"]), ["1"])
        self.assertEqual(sorted(b["label"] for b in v2["boxes"]), ["1", "2"])   # fly 2 kept where last seen
        self.assertEqual(self.client.get("/api/annotate/frame/35?proposal=continuity2").get_json()["source"], "continuity2")

    def test_easy_methods(self):
        info = self.client.get("/api/annotate/scenes?easy=rules").get_json()
        self.assertEqual(info["options"]["continuity_default"], 1)
        self.assertFalse(info["options"]["learned_available"])
        self.assertEqual(info["easy"]["method"], "rules")
        wait(info["easy"]["pending"]["id"])
        info = self.client.get("/api/annotate/scenes?easy=rules").get_json()
        self.assertIn("scenes", info["easy"])
        learned = self.client.get("/api/annotate/scenes?easy=learned").get_json()["easy"]
        self.assertEqual(learned["method"], "learned")
        self.assertIn("no scene scorer", learned["error"])

    def test_accept_and_undo(self):
        # frame 32 saved by someone before: never overwritten
        before = [{"x0": 100, "y0": 80, "x1": 160, "y1": 120, "label": "1"},
                  {"x0": 470, "y0": 470, "x1": 530, "y1": 530, "label": "2"}]
        self.assertEqual(self.client.post("/api/annotate/frame/32", json={"boxes": before}).status_code, 200)
        r = self.client.post("/api/annotate/accept_easy", json={"experiment": "other", "scenes": [{"start": 30, "length": 10}]})
        self.assertEqual(r.status_code, 409)
        r = self.client.post("/api/annotate/accept_easy", json={
            "experiment": FLAT, "continuity": 2, "method": "learned",
            "scenes": [{"start": 30, "length": 10, "score": 0.97}]})
        self.assertEqual(r.status_code, 202)
        job = wait(r.get_json()["job"]["id"])
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(job.result["frames"], 9)
        con = self.sidecar()
        rows = dict(con.execute("SELECT frame_number, author FROM frames"))
        self.assertEqual(sorted(rows), list(range(30, 40)))
        self.assertEqual(set(rows.values()), {"tester"})
        logs = [a for (a,) in con.execute("SELECT author FROM log WHERE frame_number = 35")]
        self.assertEqual(logs, [f"tester (accepted easy scene, continuity2, learned 0.97, batch {job.result['batch']})"])
        boxes = json.loads(con.execute("SELECT boxes FROM log WHERE frame_number = 35").fetchone()[0])
        self.assertEqual(sorted(b["label"] for b in boxes), ["1", "2"])
        self.assertEqual(json.loads(con.execute("SELECT boxes FROM log WHERE frame_number = 32").fetchone()[0]), before)
        con.close()
        info = self.client.get("/api/annotate/scenes").get_json()
        batch = info["accept_batches"][0]
        self.assertEqual((batch["frames"], batch["kept"], batch["continuity"], batch["method"]), (9, 9, 2, "learned"))
        # frame 35 corrected after the accept: kept by the undo
        self.client.post("/api/annotate/frame/35", json={"boxes": before})
        r = self.client.post(f"/api/annotate/accept_easy/{batch['id']}/undo", json={"experiment": FLAT}).get_json()
        self.assertEqual((r["removed"], r["kept"]), (8, 1))
        con = self.sidecar()
        self.assertEqual(sorted(fn for (fn,) in con.execute("SELECT frame_number FROM frames")), [32, 35])
        con.close()
        self.assertIsNotNone(self.client.get("/api/annotate/scenes").get_json()["accept_batches"][0]["undone"])
        self.assertEqual(self.client.post("/api/annotate/accept_easy/999/undo", json={"experiment": FLAT}).status_code, 404)

    def test_accept_stops_when_another_experiment_is_loaded(self):
        calls = []

        def switching():
            calls.append(1)
            return self.state if len(calls) < 2 else {**self.state, "experiment": "FlyHostel9/2X/other"}

        result = ba.accept_scenes(self.state, [{"start": 30, "length": 5}, {"start": 40, "length": 5}], 1, "rules",
                                  "tester", get_state=switching)
        self.assertTrue(result["stopped"])
        con = self.sidecar()
        self.assertEqual(con.execute("SELECT COUNT(*) FROM frames").fetchone()[0], 0)
        con.close()


if __name__ == "__main__":
    unittest.main()
