"""Annotate tab pre-download (annotate_pack.py): python -m unittest tests.test_annotate_pack

A synthetic recording (two chunks of moving dark blobs, imgstore layout: <chunk>.mp4 + index.db + metadata.yaml) is
packed by build_chunk; the stream must hold one access unit per frame, IDR frames where runs of frames start,
decode back to the frames, and its contours must round-trip exactly."""
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from idtrackerai_validator_server import annotate_pack as pack
from idtrackerai_validator_server import lab_paths

W = H = 96
CHUNKSIZE = 60
HAS_FFMPEG = shutil.which(lab_paths.FFMPEG or "ffmpeg") is not None


def frame_image(fn):
    img = np.full((H, W), 200, np.uint8)
    cv2.circle(img, (10 + (fn * 3) % 70, 30), 6, 20, -1)
    cv2.circle(img, (60, 10 + (fn * 2) % 70), 5, 30, -1)
    return img


def make_store(folder, chunks=2):
    """<chunk:06d>.mp4 (lossless-ish H.264, gray as yuv420p), index.db (frames), metadata.yaml"""
    with sqlite3.connect(os.path.join(folder, "index.db")) as con:
        con.execute("CREATE TABLE frames (chunk INTEGER, frame_idx INTEGER, frame_number INTEGER, frame_time INTEGER)")
        for c in range(chunks):
            proc = subprocess.Popen([lab_paths.FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                                     "-s", f"{W}x{H}", "-r", "10", "-i", "-", "-c:v", "libx264", "-qp", "0",
                                     "-pix_fmt", "yuv420p", os.path.join(folder, f"{c:06d}.mp4")], stdin=subprocess.PIPE)
            for i in range(CHUNKSIZE):
                fn = c * CHUNKSIZE + i
                proc.stdin.write(frame_image(fn).tobytes())
                con.execute("INSERT INTO frames VALUES (?, ?, ?, ?)", (c, i, fn, fn * 100))
            proc.stdin.close()
            assert proc.wait() == 0
    with open(os.path.join(folder, "metadata.yaml"), "w") as handle:
        handle.write(f"__store:\n  chunksize: {CHUNKSIZE}\n  framerate: 10\n  extension: .mp4\n  imgshape: [{H}, {W}]\n")


def decode(stream):
    """frames of an Annex B stream, decoded with ffmpeg"""
    out = subprocess.run([lab_paths.FFMPEG, "-loglevel", "error", "-f", "h264", "-i", stream, "-vsync", "0", "-pix_fmt", "gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, H, W)


class Contours(unittest.TestCase):

    def test_round_trip(self):
        by_frame = {
            5: [[[[1, 2]], [[2, 2]], [[2, 3]]], [[10, 10], [11, 10], [12, 11], [12, 12]]],   # cv2 (N,1,2) and (N,2)
            7: [],
            9: [[[300, 400], [299, 401], [298, 401]]],
        }
        back = pack.decode_contours(pack.encode_contours(by_frame))
        self.assertEqual(sorted(back), [5, 7, 9])
        for fn, cs in by_frame.items():
            self.assertEqual(len(back[fn]), len(cs))
            for a, b in zip(cs, back[fn]):
                np.testing.assert_array_equal(np.asarray(a).reshape(-1, 2), b)


class Stream(unittest.TestCase):

    def test_split_and_codec(self):
        aud = b"\x00\x00\x00\x01\x09\xf0"
        sps = b"\x00\x00\x00\x01\x67\x64\x00\x32\xac"
        idr = b"\x00\x00\x01\x65\x88\x84"
        p = b"\x00\x00\x01\x41\x9a\x00"
        data = aud + sps + idr + aud + p + aud + p + aud + sps + idr
        units = pack.split_access_units(data)
        self.assertEqual([u[2] for u in units], [True, False, False, True])
        self.assertEqual(sum(u[1] for u in units), len(data))
        self.assertEqual(units[1][0], len(aud + sps + idr))
        self.assertEqual(pack.codec_string(data), "avc1.640032")


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg with libx264")
class BuildChunk(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_annotate_pack_")
        self.store = os.path.join(self.tmp, "store")
        self.out = os.path.join(self.tmp, "pack")
        os.makedirs(self.store)
        os.makedirs(self.out)
        make_store(self.store)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build(self):
        # two runs in chunk 1 (a jump between scenes) and runs longer than a GOP
        frames = list(range(62, 62 + 20)) + list(range(100, 115))
        task = {"basedir": self.store, "out": self.out, "chunk": 1, "frames": frames,
                "frame_idx": [fn - CHUNKSIZE for fn in frames], "stamp": ["test"], "config": None,
                "ffmpeg": lab_paths.FFMPEG, "width": W, "height": H, "framerate": 10.0, "extension": ".mp4"}
        summary, _ = pack.build_chunk(task)
        self.assertEqual(summary["frames"], len(frames))
        with open(os.path.join(self.out, "000001.json")) as handle:
            index = json.load(handle)
        self.assertEqual(index["frames"], frames)
        self.assertEqual(index["vkey"], pack.vkey(["test"], frames))
        self.assertTrue(index["codec"].startswith("avc1."))
        keys = [i for i, k in enumerate(index["keys"]) if k]
        self.assertIn(0, keys)
        self.assertIn(20, keys)                        # the second run starts with an IDR frame
        self.assertTrue(all(b - a <= pack.GOP for a, b in zip(keys, keys[1:] + [len(frames)])))
        with open(os.path.join(self.out, "000001.h264"), "rb") as handle:
            data = handle.read()
        self.assertEqual(index["offsets"][-1] + index["sizes"][-1], len(data))
        # each access unit from an IDR frame on decodes alone (what the browser does after a jump)
        decoded = decode(os.path.join(self.out, "000001.h264"))
        self.assertEqual(len(decoded), len(frames))
        for i, fn in enumerate(frames):
            err = np.abs(decoded[i].astype(int) - frame_image(fn).astype(int)).mean()
            self.assertLess(err, 4, f"frame {fn}")
        part = os.path.join(self.tmp, "from_key.h264")
        with open(part, "wb") as handle:
            handle.write(data[index["offsets"][20]:])
        again = decode(part)
        self.assertEqual(len(again), len(frames) - 20)
        np.testing.assert_array_equal(again[0], decoded[20])


class Sigs(unittest.TestCase):
    """The data sig of a chunk changes with a save in that chunk only"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_annotate_pack_")
        self.basedir = os.path.join(self.tmp, "exp")
        os.makedirs(os.path.join(self.basedir, "flyhostel", "validation"))
        self.dbfile = os.path.join(self.basedir, "exp.db")
        sqlite3.connect(self.dbfile).close()
        self.state = {"experiment": "exp", "basedir": self.basedir, "dbfile": self.dbfile, "chunksize": CHUNKSIZE}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_changes_its_chunk(self):
        from idtrackerai_validator_server import box_annotations as ba
        before = pack.chunk_sigs(self.state, [0, 1, 2])
        con = ba._connect(self.basedir, create=True)
        with con:
            con.execute("INSERT INTO frames VALUES (?, ?, ?, ?)", (CHUNKSIZE + 5, "me", 1.5, 0))
        con.close()
        after = pack.chunk_sigs(self.state, [0, 1, 2])
        self.assertEqual(before[0], after[0])
        self.assertNotEqual(before[1], after[1])
        self.assertEqual(before[2], after[2])
        con = ba._connect(self.basedir)
        with con:
            con.execute("UPDATE frames SET updated = 2.5")
        con.close()
        self.assertNotEqual(after[1], pack.chunk_sigs(self.state, [1])[1])


if __name__ == "__main__":
    unittest.main()
