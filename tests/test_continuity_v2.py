"""Tracker v2 of the continuity proposal (continuity.RunV2) and the scene scorer (scene_scorer.py) on synthetic data:
python -m unittest tests.test_continuity_v2"""
import os
import sqlite3
import tempfile
import unittest

import numpy as np

from idtrackerai_validator_server import continuity, scene_scorer

NOTCHES = [(1000.0, 50.0)]


def rows(*flies):
    """(x, y, lid, fragment, area) per detection"""
    return [(float(x), float(y), lid, frag, 100.0) for x, y, lid, frag in flies]


def start(*positions):
    return {lab: (x, y, True) for lab, (x, y) in enumerate(positions, 1)}


class RunV2Test(unittest.TestCase):

    def test_undetected_fly_kept_away_from_notches(self):
        run1 = continuity.Run("x.db", 0, "saved", start((100, 100), (500, 500)), NOTCHES, 2)
        run2 = continuity.RunV2("x.db", 0, "saved", start((100, 100), (500, 500)), NOTCHES, 2)
        r = rows((101, 100, 1, 7))
        res1 = run1.step(1, [x[:3] for x in r])
        res2 = run2.step(1, r)
        self.assertEqual(res1["ghosts"], [])
        self.assertTrue(res1["flags"])
        self.assertEqual([g[0] for g in res2["ghosts"]], [2])
        self.assertEqual(res2["ghosts"][0][1:3], (500, 500))
        self.assertEqual(res2["flags"], [])

    def test_fly_reappearing_away_is_paired(self):
        run = continuity.RunV2("x.db", 0, "saved", start((100, 100), (500, 500)), NOTCHES, 2)
        run.step(1, rows((100, 100, 1, 1)))
        run.step(2, rows((100, 100, 1, 1)))
        # fly 2 comes back 150 px away, with idtrackerai's identity of fly 1's (wrong) lid
        res = run.step(3, rows((100, 100, 1, 1), (650, 500, 1, 9)))
        self.assertEqual([lab for lab, _, _ in res["rows"]], [1, 2])
        self.assertIn("reappeared", res["rows"][1][1])
        self.assertEqual(res["ghosts"], [])

    def test_gate_100(self):
        run1 = continuity.Run("x.db", 0, "saved", start((100, 100), (900, 900)), NOTCHES, 2)
        run2 = continuity.RunV2("x.db", 0, "saved", start((100, 100), (900, 900)), NOTCHES, 2)
        r = rows((180, 100, 0, 3), (900, 900, 2, 4))
        self.assertIsNone(run1.step(1, [x[:3] for x in r])["rows"][0][0])
        self.assertEqual(run2.step(1, r)["rows"][0][0], 1)

    def test_fragment_continuity_beats_nearest(self):
        # two flies side by side; in the next frame fly 1's detection (same fragment) is nearer fly 2's last place
        run = continuity.RunV2("x.db", 0, "saved", start((100, 100), (130, 100)), NOTCHES, 2, {1: 11, 2: 12})
        r = rows((128, 100, 0, 11), (100, 100, 0, 12))
        self.assertEqual([lab for lab, _, _ in run.step(1, r)["rows"]], [1, 2])
        plain = continuity.Run("x.db", 0, "saved", start((100, 100), (130, 100)), NOTCHES, 2)
        self.assertEqual([lab for lab, _, _ in plain.step(1, [x[:3] for x in r])["rows"]], [2, 1])

    def test_box_fragments(self):
        boxes = [{"x0": 90, "y0": 90, "x1": 110, "y1": 110, "label": "2"},
                 {"x0": 0, "y0": 0, "x1": 5, "y1": 5, "label": "1"},
                 {"x0": 90, "y0": 90, "x1": 110, "y1": 110, "label": "CROSSING"}]
        self.assertEqual(continuity._box_fragments(boxes, rows((100, 100, 1, 42))), {2: 42})


def make_db(path, frames):
    """frames: {frame: [(x, y, lid, fragment)]}"""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE ROI_0 (id INTEGER PRIMARY KEY, frame_number int, in_frame_index int, x real, y real, "
                "fragment int, area int, modified int, class_name char(10))")
    con.execute("CREATE TABLE IDENTITY (id INTEGER PRIMARY KEY, frame_number int, in_frame_index int, "
                "local_identity int, identity int)")
    con.execute("CREATE TABLE LANDMARKS (id INTEGER PRIMARY KEY, shape char(50), specification char(100))")
    con.execute("INSERT INTO LANDMARKS (shape, specification) VALUES ('notch', '[[990, 40], [1010, 60]]')")
    for fn, flies in frames.items():
        for i, (x, y, lid, frag) in enumerate(flies):
            con.execute("INSERT INTO ROI_0 (frame_number, in_frame_index, x, y, fragment, area) VALUES (?,?,?,?,?,100)",
                        (fn, i, x, y, frag))
            con.execute("INSERT INTO IDENTITY (frame_number, in_frame_index, local_identity, identity) VALUES (?,?,?,?)",
                        (fn, i, lid, lid))
    con.commit()
    con.close()


class TrackTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "exp.db")
        frames = {}
        for fn in range(0, 30):      # stable: both flies identified
            frames[fn] = [(100 + fn, 100, 1, 1), (500, 500, 2, 2)]
        for fn in range(30, 40):     # the scene: fly 2 not detected, fly 1 without identity
            frames[fn] = [(130, 100, 0, 3)]
        make_db(self.db, frames)
        self.scenes = [{"start": 30, "length": 10, "reason": "ids missing"}]
        continuity._runs.clear()
        continuity._notches.clear()
        continuity._anchors.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def test_version_2_from_idtrackerai_anchor(self):
        v1 = continuity.track(self.db, 35, 1000, 2, self.scenes, None)
        v2 = continuity.track(self.db, 35, 1000, 2, self.scenes, None, version=2)
        self.assertEqual(v1["anchor_kind"], "tracking")
        self.assertEqual(v1["ghosts"], [])
        self.assertEqual(v2["anchor"], v1["anchor"])
        self.assertEqual([lab for lab, _, _ in v2["rows"]], [1])
        self.assertEqual([g[0] for g in v2["ghosts"]], [2])

    def test_version_2_from_saved_frame(self):
        boxes = [{"x0": 120, "y0": 90, "x1": 140, "y1": 110, "label": "2"},
                 {"x0": 490, "y0": 490, "x1": 510, "y1": 510, "label": "1"}]
        v2 = continuity.track(self.db, 35, 1000, 2, self.scenes, (29, boxes, 1.0), version=2)
        self.assertEqual(v2["anchor_kind"], "saved")
        # the person swapped them in frame 29: the detection follows their fly 2
        self.assertEqual([lab for lab, _, _ in v2["rows"]], [2])
        self.assertEqual([g[0] for g in v2["ghosts"]], [1])

    def test_scorer_end_to_end(self):
        saved = {}

        def saved_before(first):
            return None

        runs = list(scene_scorer.stretch_runs(self.db, self.scenes, 1000, 2, saved_before, lambda first: None))
        self.assertEqual(len(runs), 1)
        first, last, members, run, results, rws = runs[0]
        f = scene_scorer.features(first, last, 2, members, run, results, rws)
        self.assertEqual(set(f), set(scene_scorer.FEATURES))
        self.assertEqual(f["f_ghost_ff"], 10)
        self.assertEqual(f["frames_short"], 10)
        self.assertEqual(f["r_ids_missing"], 1)
        # a model trained on two kinds of stretch: ghosts kept (wrong) vs none (right)
        X = np.array([[float(f[k]) for k in scene_scorer.FEATURES]] * 30
                     + [[0.0 if k.startswith("f_") or k.startswith("frames_") else float(f[k]) for k in scene_scorer.FEATURES]] * 30)
        y = np.array([0] * 30 + [1] * 30)
        with scene_scorer.limited():
            fitted = scene_scorer.new_model().fit(X, y)
        model = {"model": fitted, "features": scene_scorer.FEATURES, "threshold": 0.5}
        out = scene_scorer.score_scenes(self.db, self.scenes, 1000, 2, saved_before, lambda first: None, model=model)
        self.assertFalse(out[30]["easy"])
        self.assertLess(out[30]["p"], 0.5)
        del saved


class ScorerTest(unittest.TestCase):

    def test_frame_errors(self):
        result = {"rows": [(1, "", None), (2, "", None), (None, "", "low")], "ghosts": [(3, 300.0, 300.0, 5)]}
        r = rows((100, 100, 1, 1), (200, 200, 2, 2), (50, 50, 0, 3))
        boxes = [{"x0": 90, "y0": 90, "x1": 110, "y1": 110, "label": "2"},
                 {"x0": 190, "y0": 190, "x1": 210, "y1": 210, "label": "1"},
                 {"x0": 290, "y0": 290, "x1": 320, "y1": 320, "label": "3"}]
        kinds = sorted(k for k, _ in scene_scorer.frame_errors(boxes, r, result))
        self.assertEqual(kinds, ["wrong_label", "wrong_label"])
        boxes[2] = {"x0": 590, "y0": 590, "x1": 620, "y1": 620, "label": "3"}
        boxes[0]["label"], boxes[1]["label"] = "1", "2"
        self.assertEqual([k for k, _ in scene_scorer.frame_errors(boxes, r, result)], ["ghost_misplaced"])

    def test_precision_threshold(self):
        scores = np.linspace(1, 0, 100)
        y = np.array([1] * 60 + [0] + [1] * 9 + [0] * 30)
        self.assertAlmostEqual(scene_scorer.precision_threshold(scores, y, 0.99), scores[59])
        self.assertIsNone(scene_scorer.precision_threshold(scores[:10], [0] * 10, 0.99))


if __name__ == "__main__":
    unittest.main()
