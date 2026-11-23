"""Identity swap detector (swaps.py) on synthetic tracks: python -m unittest tests.test_swaps"""
import unittest

import numpy as np

from idtrackerai_validator_server import swaps

FPS = 47.0
BODY = 83.25        # 2.5 mm at 333 px/cm


def rows(tracks):
    """tracks: {key: [(frame, x, y) or None]} -> arrays of frame, key, x, y."""
    out = []
    for key, points in tracks.items():
        for p in points:
            if p is not None:
                out.append((p[0], key, p[1], p[2]))
    out.sort()
    a = np.array(out, dtype=float)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2], a[:, 3]


def walk(start, velocity, frames, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x0, y0 = start
    vx, vy = velocity
    return [(f, x0 + vx * f + rng.normal(0, jitter), y0 + vy * f + rng.normal(0, jitter)) for f in frames]


def run(tracks, params=None, n_frames=200):
    p = swaps.params_with(params)
    f, k, x, y = rows(tracks)
    cands, carry, jump = swaps.detect_block(f, k, x, y, 0, n_frames - 1, FPS, BODY, p)
    for c in cands:
        c["chunk"], c["jump"] = 0, jump
    return swaps.events_from_candidates(cands, FPS, BODY, p)


class SwapDetector(unittest.TestCase):

    def test_clean_exchange(self):
        # two flies 200 px apart walking slowly; at frame 100 their identities swap
        frames = range(200)
        a = walk((500, 500), (0.5, 0), frames, jitter=0.5, seed=1)
        b = walk((700, 500), (-0.5, 0), frames, jitter=0.5, seed=2)
        A = [p if p[0] < 100 else b[p[0]] for p in a]
        B = [p if p[0] < 100 else a[p[0]] for p in b]
        events = run({1: A, 2: B})
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual(e["type"], "exchange")
        self.assertEqual(e["frame"], 100)
        self.assertTrue(e["net_swap"])
        self.assertGreater(e["score"], 0.8)

    def test_takeover_of_a_missing_fly(self):
        # fly 2 is lost at frame 100; identity 1 jumps onto fly 2's track
        frames = range(200)
        a = walk((300, 300), (0, 0), frames, jitter=0.5, seed=1)
        b = walk((800, 300), (0.3, 0), frames, jitter=0.5, seed=2)
        A = [p if p[0] < 100 else b[p[0]] for p in a]
        B = [p if p[0] < 100 else None for p in b]
        events = run({1: A, 2: B})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "takeover")
        self.assertEqual((events[0]["a"], events[0]["b"]), (1, 2))

    def test_crossing_without_swap(self):
        # two flies cross: merged (one detection, identity 0 -> left out) for 10 frames, then go on
        frames = range(200)
        a = walk((400, 500), (2, 0), frames, jitter=0.5, seed=1)
        b = walk((800, 500), (-2, 0), frames, jitter=0.5, seed=2)
        A = [p if not 95 <= p[0] < 105 else None for p in a]
        B = [p if not 95 <= p[0] < 105 else None for p in b]
        self.assertEqual(run({1: A, 2: B}), [])

    def test_crossing_with_swap(self):
        # two flies meet (70 px apart), are not detected for 10 frames, and turn back with swapped
        # identities: each identity comes out where the other fly was
        frames = range(200)
        a = walk((400, 500), (1.7, 0), frames, jitter=0.5, seed=1)      # at ~560 by frame 94
        b = walk((800, 500), (-1.7, 0), frames, jitter=0.5, seed=2)     # at ~640
        a_end, b_end = a[94][1], b[94][1]
        A = [p if p[0] < 95 else (None if p[0] < 105 else (p[0], b_end + 1.7 * (p[0] - 105), 500)) for p in a]
        B = [p if p[0] < 95 else (None if p[0] < 105 else (p[0], a_end - 1.7 * (p[0] - 105), 500)) for p in b]
        events = run({1: A, 2: B})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "exchange")
        self.assertEqual(events[0]["gap"], 11)
        self.assertTrue(events[0]["encounter"])

    def test_stationary_touching_flies(self):
        # two flies sitting still, centroids 60 px apart, with centroid jitter: nothing
        frames = range(2000)
        a = walk((500, 500), (0, 0), frames, jitter=3, seed=1)
        b = walk((560, 500), (0, 0), frames, jitter=3, seed=2)
        self.assertEqual(run({1: a, 2: b}, n_frames=2000), [])

    def test_touching_flicker_is_one_ranked_down_event(self):
        # identities flip back and forth between two touching flies: one event, no net swap
        frames = range(200)
        a = walk((500, 500), (0, 0), frames, jitter=0.5, seed=1)
        b = walk((570, 500), (0, 0), frames, jitter=0.5, seed=2)
        flipped = set(range(100, 105)) | set(range(110, 115))
        A = [b[p[0]] if p[0] in flipped else p for p in a]
        B = [a[p[0]] if p[0] in flipped else p for p in b]
        events = run({1: A, 2: B})
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["net_swap"])
        self.assertTrue(events[0]["encounter"])
        self.assertLess(events[0]["score"], 0.6)

    def test_missing_fly_comes_back(self):
        # a fly is lost for 20 frames and comes back where it was: nothing
        frames = range(200)
        a = walk((500, 500), (0.2, 0), frames, jitter=0.5, seed=1)
        b = walk((900, 500), (0, 0.2), frames, jitter=0.5, seed=2)
        A = [p if not 80 <= p[0] < 100 else None for p in a]
        self.assertEqual(run({1: A, 2: b}), [])

    def test_carry_across_chunks(self):
        # the swap happens on the first frame of a block: found through the carried positions
        frames = range(100, 200)
        a = walk((500, 500), (0, 0), frames, seed=1)
        b = walk((800, 500), (0, 0), frames, seed=2)
        p = swaps.params_with()
        f, k, x, y = rows({1: b, 2: a})      # swapped from the first frame
        cands, _, _ = swaps.detect_block(f, k, x, y, 100, 199, FPS, BODY, p,
                                         carry={1: (500, 500, 99), 2: (800, 500, 99)})
        self.assertEqual({c["frame"] for c in cands}, {100})
        self.assertEqual(len(cands), 2)


if __name__ == "__main__":
    unittest.main()
