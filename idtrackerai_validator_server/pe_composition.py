"""
pe_composition.py  —  the PE tab's burst video ("video" button), on demand as an mp4.

The button (client `recordBurst.js`) records in the browser the media card of the tab:
top left the burst clip (`<media>/videos/<fly>_burst_<id>.mp4`, with the pose over it
when the "pose" box is ticked), top right the pose on white (`.pose.json` next to the
clip), below the distance trace (`BurstTrace.js`) and the proboscis confidence trace
(`ConfidenceTrace.js`) with the playhead; 560x560, 30 fps, .webm.

Two engines:
  browser (default): pe_record.py presses that very button in headless Chrome on the
          tab's link (`links.pe_url`) and converts the .webm to H.264 / yuv420p mp4:
          the button's output exactly (~10 s for a 6 s burst plus ~4 s server start).
  draw:   this module redraws the same layout with cv2 (panels) and matplotlib (each trace
          once, playhead per frame): no browser, ~2.5 s, any size / slow motion / trim, but
          an approximation (fonts, trace aspect, crisp pose instead of an upscaled canvas).

    GET /api/pe/composition?experiment=&identity=&burst_id=   (see register_pe_composition)
    GET /api/pe/link?experiment=&identity=&burst_id=          (the PE tab on that burst)

    python -m idtrackerai_validator_server.pe_composition EXPERIMENT IDENTITY BURST_ID... -o out.mp4|folder
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

from idtrackerai_validator_server.experiments import to_flat, to_relpath
from idtrackerai_validator_server.pe_validation import (
    _fly_id, _load_traces_cached, _media_dir, _traces_file, burst_segments, burst_trace,
)

logger = logging.getLogger(__name__)

# ---- layout (recordBurst.js: W = 2 panels, H = panel + one trace row per trace) ----
DEFAULT_PANEL = 400          # px of each square panel (the browser records 280: 560x560)
DEFAULT_FPS = 30             # recordBurst.js records the canvas at 30 fps
# The trace SVGs are laid out in their on-screen px (margins and fonts are fixed px) and
# the browser squeezes them into 2*panel x panel/2. Drawn here in a typical on-screen
# size of the same 4:1 shape, scaled to the output without distortion.
TRACE_VIEW = (880, 220)
MAX_SYNC_FRAMES = 1800       # longer renders run as a job (202)
MAX_FRAMES = 30000           # refused beyond this many output frames

# ---- pose (BurstVideo.js) ----
POSE_ALPHA = 0.55            # pose over the video
NODE_R = 1.8                 # in video px, scaled with the panel
IN_BOUT_SCALE = 1.5          # proboscis node radius while the frame is in a bout
EDGE_COLOR = "#00c8c8"
NODE_COLOR = "#00e5e5"
PROB_COLOR = "#ff0000"

# ---- traces (BurstTrace.js / ConfidenceTrace.js) ----
CONF_THRESH = 0.5
PLAYHEAD = "#d62728"


def _bgr(hex_color):
    h = hex_color.lstrip("#")
    return np.array([int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)], dtype=np.float32)


def _ffmpeg():
    """ffmpeg of this python's environment (has libx264), else the one in PATH"""
    local = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    return local if os.path.exists(local) else (shutil.which("ffmpeg") or "ffmpeg")


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------

def fly_name(experiment, identity):
    return _fly_id(to_relpath(experiment), identity)


def find_burst(experiment, fly, start, end):
    """burst_id of the fly's burst overlapping [start, end] the most (frame numbers)"""
    exp = to_relpath(experiment)
    tf = _traces_file(exp, fly)
    if not os.path.exists(tf):
        raise LookupError(f"no trace feather for {fly}")
    d = _load_traces_cached(tf, fly)
    d = d.loc[(d["frame_number"] >= start) & (d["frame_number"] <= end), "burst_id"]
    if d.empty:
        raise LookupError(f"no burst of {fly} between frames {start} and {end}")
    return int(d.value_counts().idxmax())


def bout_range(experiment, fly, burst_id):
    """first start_fn .. last end_fn of the burst's bouts (the browser's file name), or None"""
    exp = to_relpath(experiment)
    d = _load_traces_cached(_traces_file(exp, fly), fly)
    d = d.loc[d["burst_id"] == burst_id, ["start_fn", "end_fn"]].dropna()
    if d.empty:
        return None
    return int(d["start_fn"].min()), int(d["end_fn"].max())


def load_burst(experiment, fly, burst_id, method="visible"):
    """Everything the PE tab shows of one burst: clip path, pose, trace, segments."""
    exp = to_relpath(experiment)
    stem = f"{fly}_burst_{int(burst_id)}"
    videos = os.path.join(_media_dir(exp), "videos")
    clip = os.path.join(videos, f"{stem}.mp4")
    if not os.path.exists(clip):
        raise LookupError(f"no burst video {clip}")
    pose_file = os.path.join(videos, f"{stem}.pose.json")
    pose = None
    if os.path.exists(pose_file):
        with open(pose_file) as fh:
            pose = json.load(fh)
    else:
        logger.warning("No pose for %s (%s): pose panels left empty", stem, pose_file)
    trace = burst_trace(exp, fly, int(burst_id))
    segments = burst_segments(exp, fly, int(burst_id), method) if method else None
    return {"stem": stem, "clip": clip, "pose": pose, "trace": trace, "segments": segments}


def read_clip(path):
    """every frame of the burst clip (BGR), small (≈100x100 px crops)"""
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frame could be read from {path}")
    return frames


# ----------------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------------

def _paint(img, mask, color, alpha):
    """source-over of `color` at opacity alpha * mask (uint8 0..255) onto img (float32 BGR)"""
    if alpha <= 0:
        return
    a = (mask.astype(np.float32) * (alpha / 255.0))[..., None]
    img *= (1.0 - a)
    img += a * color


def draw_pose(img, pose, fr, sx, sy, alpha):
    """BurstVideo.js drawPose: cyan edges and nodes, the proboscis red with opacity =
    its confidence (bigger in a bout). Coordinates in video px, scaled by (sx, sy)."""
    if not fr or not fr.get("pts"):
        return
    pts = fr["pts"]
    conf = fr.get("conf")
    s = (sx + sy) / 2
    h, w = img.shape[:2]
    SHIFT = 4
    k = 1 << SHIFT

    def p(i):
        # canvas coordinates have pixel centres at +0.5, cv2's at integers
        x, y = pts[i]
        return (int(round((x * sx - 0.5) * k)), int(round((y * sy - 0.5) * k)))

    edges = np.zeros((h, w), np.uint8)
    thick = max(1, int(round(1.0 * s)))
    for a, b in pose["edges"]:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(edges, p(a), p(b), 255, thick, cv2.LINE_AA, SHIFT)
    _paint(img, edges, _bgr(EDGE_COLOR), alpha)

    nodes = np.zeros((h, w), np.uint8)
    prob_idx = pose.get("prob_idx")
    for i, pt in enumerate(pts):
        if pt is None or i == prob_idx:
            continue
        cv2.circle(nodes, p(i), int(round(NODE_R * s * k)), 255, -1, cv2.LINE_AA, SHIFT)
    _paint(img, nodes, _bgr(NODE_COLOR), alpha)

    if prob_idx is not None and prob_idx < len(pts) and pts[prob_idx] is not None:
        c = conf[prob_idx] if conf else None
        c_alpha = 0.35 if c is None else max(0.0, min(1.0, float(c)))
        r = NODE_R * (IN_BOUT_SCALE if fr.get("in_bout") else 1.0) * s
        prob = np.zeros((h, w), np.uint8)
        cv2.circle(prob, p(prob_idx), int(round(r * k)), 255, -1, cv2.LINE_AA, SHIFT)
        _paint(img, prob, _bgr(PROB_COLOR), alpha * c_alpha)


class TracePlot:
    """One trace panel (BurstTrace.js with which='dist', ConfidenceTrace.js with
    which='conf'): the static plot drawn once with matplotlib in the SVG's own px
    (a TRACE_VIEW viewBox, scaled to the output), the playhead per frame."""

    def __init__(self, trace, which, width, height, segments=None):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.patches import Circle, Rectangle

        self.which = which
        W, H = TRACE_VIEW
        self.k = width / W
        pts = trace["points"]
        ts = [p["t_s"] for p in pts]
        self.xmax = max(10.0, max(ts))
        show_lanes = which == "dist" and segments is not None
        m = {"t": 10, "r": 12, "b": 64 if show_lanes else 40, "l": 44}
        iw, ih = W - m["l"] - m["r"], H - m["t"] - m["b"]
        self.m, self.iw, self.ih = m, iw, ih

        key = "dist" if which == "dist" else "conf"
        ys = [p[key] for p in pts if p[key] is not None]
        if which == "dist":
            ymax = max([0.01] + ys)
            ymin = min([0.0] + ys)
            color, ylabel = "#1f77b4", "Δ head–proboscis (mm)"
        else:
            ymax = max([1.0] + ys)
            ymin = 0.0
            color, ylabel = "#8c564b", "proboscis confidence"

        X = lambda t: m["l"] + (t / self.xmax) * iw
        Y = lambda v: m["t"] + ih - (v / ymax) * ih
        self.X = X

        fig = Figure(figsize=(W / 72, H / 72), dpi=72 * self.k)
        canvas = FigureCanvasAgg(fig)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        line = dict(solid_capstyle="butt", solid_joinstyle="miter")
        font = dict(family="DejaVu Sans")

        def text(x, y, s, size, color="black", ha="left", va="baseline", **kw):
            ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, **font, **kw)

        if show_lanes:
            for s in segments.get("gt", []):
                ax.add_patch(Rectangle((X(s["t0"]), m["t"]), max(1, X(s["t1"]) - X(s["t0"])), ih,
                                       facecolor="#2ca02c", alpha=0.12, edgecolor="none"))

        # axes
        ax.plot([m["l"], m["l"]], [m["t"], m["t"] + ih], color="#999", lw=1, **line)
        ax.plot([m["l"], m["l"] + iw], [m["t"] + ih, m["t"] + ih], color="#999", lw=1, **line)
        text(m["l"] - 4, Y(ymin), f"{ymin:.2f}" if which == "dist" else "0", 9, "#666", "right", "center")
        text(m["l"] - 4, Y(ymax), f"{ymax:.2f}", 9, "#666", "right", "center")
        text(m["l"] + iw / 2, H - 4, "time within burst (s)", 11, ha="center")
        text(12, m["t"] + ih / 2, ylabel, 11, ha="center", rotation=90, rotation_mode="anchor")

        if which == "conf" and CONF_THRESH <= ymax:
            ax.plot([m["l"], m["l"] + iw], [Y(CONF_THRESH)] * 2, color="#bbb", lw=1,
                    dashes=(4, 3), **line)
            text(m["l"] + iw, Y(CONF_THRESH) - 2, f"{CONF_THRESH}", 8, "#999", "right")

        # trace, broken at nulls
        run = []
        for p in pts + [{"t_s": None, key: None}]:
            if p[key] is None:
                if run:
                    ax.plot([X(t) for t, _ in run], [Y(v) for _, v in run], color=color, lw=1.6, **line)
                    run = []
            else:
                run.append((p["t_s"], p[key]))

        for p in pts:
            if p["is_peak"] and p[key] is not None:
                ax.add_patch(Circle((X(p["t_s"]), Y(p[key])), 3, facecolor="#d62728", edgecolor="none"))

        lane_y, gap_y = m["t"] + ih + 10, m["t"] + ih + 24
        for s in trace["spans"]:
            ax.plot([X(s["t0"]), X(s["t1"])], [lane_y] * 2, color=color, lw=4, **line)
            text(X((s["t0"] + s["t1"]) / 2), lane_y + 11, f"{s['dur']:.2f}s", 9, color, "center")
        for g in trace["gaps"]:
            for xs, ys_ in (([X(g["g0"]), X(g["g1"])], [gap_y] * 2),
                            ([X(g["g0"])] * 2, [gap_y - 3, gap_y + 3]),
                            ([X(g["g1"])] * 2, [gap_y - 3, gap_y + 3])):
                ax.plot(xs, ys_, color="#888", lw=1, **line)
            text(X((g["g0"] + g["g1"]) / 2), gap_y + 12, f"{g['gap']:.2f}s", 8, "#888", "center")

        if show_lanes:
            auto_y, gt_y = m["t"] + ih + 42, m["t"] + ih + 52
            text(m["l"] - 4, auto_y, "auto", 8, "#ff7f0e", "right", "center")
            text(m["l"] - 4, gt_y, "GT" + (" ✓" if segments.get("done") else ""), 8, "#2ca02c", "right", "center")
            for s in segments.get("auto", []):
                ax.plot([X(s["t0"]), max(X(s["t1"]), X(s["t0"]) + 1)], [auto_y] * 2,
                        color="#ff7f0e", lw=5, **line)
            for s in segments.get("gt", []):
                ax.plot([X(s["t0"]), max(X(s["t1"]), X(s["t0"]) + 1)], [gt_y] * 2,
                        color="#2ca02c", lw=7, **line)

        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        bg = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        if bg.shape[:2] != (height, width):
            bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_AREA)
        self.background = bg
        self.ky = height / H
        self.kx = width / W

    def frame(self, play_t):
        img = self.background.copy()
        if play_t is not None and 0 <= play_t <= self.xmax:
            x = self.X(play_t) * self.kx - 0.5
            y0, y1 = self.m["t"] * self.ky, (self.m["t"] + self.ih) * self.ky
            SHIFT = 4
            k = 1 << SHIFT
            cv2.line(img, (int(round(x * k)), int(round(y0 * k))), (int(round(x * k)), int(round(y1 * k))),
                     tuple(int(c) for c in _bgr(PLAYHEAD)), max(1, int(round(1.5 * self.kx))),
                     cv2.LINE_AA, SHIFT)
        return img


# ----------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------

def plan_frames(n_clip, clip_start, fps_src, fps_out, slowdown, start=None, end=None):
    """Clip frame indices shown in each output frame: the clip played `slowdown` times
    slower than real time (1 = as the browser plays it), sampled at fps_out."""
    lo, hi = 0, n_clip - 1
    if start is not None:
        lo = max(lo, int(start) - clip_start)
    if end is not None:
        hi = min(hi, int(end) - clip_start)
    if hi < lo:
        raise ValueError(f"frames {start}..{end} are outside the burst clip "
                         f"({clip_start}..{clip_start + n_clip - 1})")
    step = fps_src / (fps_out * slowdown)
    out, j = [], 0
    while True:
        i = lo + int(round(j * step))
        if i > hi:
            break
        out.append(i)
        j += 1
    if out[-1] != hi:
        out.append(hi)
    return out


def render(burst, output, panel=DEFAULT_PANEL, fps=DEFAULT_FPS, slowdown=1.0, overlay=False,
           start=None, end=None, crf=18, progress=None):
    """Write the composition of `burst` (load_burst) to `output` (mp4). Returns a summary."""
    t0 = time.time()
    panel = max(64, int(round(panel / 4)) * 4)          # even sizes for yuv420p
    trace_h = panel // 2
    W, H = 2 * panel, panel + 2 * trace_h

    trace, pose = burst["trace"], burst["pose"]
    fps_src = float(trace["fps"])
    clip_start = int(trace["start_frame"])              # PEValidator.js: clip_start ?? trace.start_frame
    frames = read_clip(burst["clip"])
    indices = plan_frames(len(frames), clip_start, fps_src, fps, slowdown, start, end)
    if len(indices) > MAX_FRAMES:
        raise ValueError(f"{len(indices)} output frames, more than {MAX_FRAMES}: lower fps or slowdown")

    pose_by_frame = {fr["f"]: fr for fr in pose["frames"]} if pose else {}
    vh, vw = frames[0].shape[:2]
    sx, sy = panel / vw, panel / vh

    dist = TracePlot(trace, "dist", W, trace_h, burst.get("segments"))
    conf = TracePlot(trace, "conf", W, trace_h)

    tmp = output + ".part.mp4"
    cmd = [_ffmpeg(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-movflags", "+faststart", tmp]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    white = np.full((panel, panel, 3), 255, np.float32)
    try:
        for n, i in enumerate(indices):
            global_frame = clip_start + i
            fr = pose_by_frame.get(global_frame)
            canvas = np.full((H, W, 3), 255, np.uint8)

            video = cv2.resize(frames[i], (panel, panel), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            if overlay and pose:
                draw_pose(video, pose, fr, sx, sy, POSE_ALPHA)
            canvas[:panel, :panel] = np.clip(video, 0, 255).astype(np.uint8)

            plain = white.copy()
            if pose:
                draw_pose(plain, pose, fr, sx, sy, 1.0)
            canvas[:panel, panel:] = np.clip(plain, 0, 255).astype(np.uint8)

            play_t = (global_frame - trace["start_frame"]) / fps_src
            canvas[panel:panel + trace_h] = dist.frame(play_t)
            canvas[panel + trace_h:] = conf.frame(play_t)
            proc.stdin.write(canvas.tobytes())
            if progress and n % 25 == 0:
                progress(n / len(indices))
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()[-2000:]}")
        os.replace(tmp, output)
    except BaseException:
        proc.kill()
        proc.wait()
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    if progress:
        progress(1.0)
    return {"output": output, "width": W, "height": H, "fps": fps, "slowdown": slowdown,
            "frames": len(indices), "first_frame": clip_start + indices[0],
            "last_frame": clip_start + indices[-1], "seconds": round(time.time() - t0, 2)}


def trim_range(burst, start, end):
    """(first, last) frame shown when trimmed to start..end, else None"""
    if start is None and end is None:
        return None
    f0 = int(burst["trace"]["start_frame"])
    f1 = f0 + len(burst["trace"]["points"]) - 1
    return (max(f0, start) if start is not None else f0, min(f1, end) if end is not None else f1)


def output_name(fly, burst_id, experiment=None, frames=None):
    """`<fly>_burst_<id>_frame_<first start_fn>-<last end_fn>.mp4`, as the browser's .webm
    (`frames`: the range shown when trimmed)"""
    if frames is None and experiment is not None:
        try:
            frames = bout_range(experiment, fly, burst_id)
        except Exception:
            frames = None
    if frames:
        return f"{fly}_burst_{int(burst_id)}_frame_{frames[0]}-{frames[1]}.mp4"
    return f"{fly}_burst_{int(burst_id)}.mp4"


def compose(experiment, identity=None, burst_id=None, output=None, fly=None, start=None, end=None,
            engine="browser", method="visible", viewport=None, fps=None, progress=None, **options):
    """Find the burst (by id, or the one overlapping start..end) and render it.
    engine "browser": the PE tab's own video button in headless Chrome (pe_record.py), the
    exact composition; "draw": this module's redraw (options: panel, slowdown, overlay, crf;
    start/end with burst_id trim it). Returns the summary."""
    fly = fly or fly_name(experiment, identity)
    if burst_id is None:
        if start is None or end is None:
            raise ValueError("give burst_id or a frame range (start and end)")
        burst_id = find_burst(experiment, fly, int(start), int(end))
        start = end = None                              # the range only located the burst
    if engine not in ("browser", "draw"):
        raise ValueError("engine is browser or draw")
    if engine == "browser" and (start is not None or end is not None):
        raise ValueError("the video button records the whole burst: trimming needs engine=draw")
    burst = load_burst(experiment, fly, int(burst_id), method=method)   # 404s before starting anything
    name = output_name(fly, burst_id, experiment, trim_range(burst, start, end))
    if output is None:
        output = name
    elif os.path.isdir(output):
        output = os.path.join(output, name)
    if engine == "browser":
        from idtrackerai_validator_server.pe_record import record_bursts, VIEWPORT, RECORD_FPS
        summary = record_bursts(experiment, [(fly, int(burst_id), output)], viewport=viewport or VIEWPORT,
                                fps=fps or RECORD_FPS, progress=progress)[0]
        summary["engine"] = "browser"
        return summary
    summary = render(burst, output, start=start, end=end, fps=fps or DEFAULT_FPS, progress=progress, **options)
    summary.update({"fly": fly, "burst_id": int(burst_id), "engine": "draw"})
    return summary


def _parse_viewport(text):
    if not text:
        return None
    w, h = (int(v) for v in str(text).lower().split("x"))
    if not (800 <= w <= 3840 and 600 <= h <= 2160):
        raise ValueError("viewport between 800x600 and 3840x2160")
    return (w, h)


# ----------------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------------

def register_pe_composition(app, get_selected_experiment):
    """GET /api/pe/composition (the burst video as an mp4) and GET /api/pe/link."""
    from flask import jsonify, request, send_file
    from idtrackerai_validator_server.jobs import start_job
    from idtrackerai_validator_server.links import pe_url

    def _flag(args, key, default="0"):
        return args.get(key, default) not in ("0", "false", "")

    def _target(args):
        """(experiment, fly, burst_id or None, start, end) of the query; ValueError if incomplete"""
        experiment = args.get("experiment") or get_selected_experiment()
        if not experiment:
            raise ValueError("no experiment given or loaded")
        if args.get("fly"):
            fly = args["fly"]
        elif args.get("identity") not in (None, ""):
            fly = fly_name(experiment, int(args["identity"]))
        else:
            raise ValueError("give identity or fly")

        def num(key):
            return int(args[key]) if args.get(key) not in (None, "") else None
        burst_id = num("burst_id") if num("burst_id") is not None else num("burst")
        return experiment, fly, burst_id, num("start"), num("end")

    @app.route("/api/pe/link", methods=["GET"])
    def pe_link():
        """Link opening the PE tab on experiment + fly (+ burst): ?experiment=&identity=|fly=&burst_id=
        (or start & end locating the burst)"""
        try:
            experiment, fly, burst_id, start, end = _target(request.args)
            if burst_id is None and start is not None and end is not None:
                burst_id = find_burst(experiment, fly, start, end)
        except (ValueError, TypeError) as error:
            return jsonify({"error": str(error)}), 400
        except LookupError as error:
            return jsonify({"error": error.args[0]}), 404
        return jsonify({"url": pe_url(experiment, fly=fly, burst_id=burst_id, base_url=request.host_url),
                        "experiment": to_flat(experiment), "fly": fly, "burst_id": burst_id})

    @app.route("/api/pe/composition", methods=["GET"])
    def pe_composition():
        """The PE tab's burst video as an mp4. Query: experiment (default: the loaded one),
        identity or fly, burst_id or start & end (frame numbers locating the burst).
        engine=browser (default): the tab's video button recorded by headless Chrome, as a job
        (202, file at /api/jobs/<id>/download) or with wait=1 the mp4 itself; viewport (1920x1080),
        fps (30). engine=draw: the cv2/matplotlib redraw, see render(); start/end with
        burst_id trim it; the mp4 directly up to MAX_SYNC_FRAMES output frames, else a job."""
        args = request.args
        try:
            experiment, fly, burst_id, start, end = _target(args)
            engine = args.get("engine", "browser")
            if engine not in ("browser", "draw"):
                raise ValueError("engine is browser or draw")
            fps = float(args["fps"]) if args.get("fps") else None
            slowdown = float(args.get("slowdown", 1))
            panel = int(args.get("panel", DEFAULT_PANEL))
            viewport = _parse_viewport(args.get("viewport"))
            method = args.get("method", "visible") if _flag(args, "segments", "1") else None
            if (fps is not None and not 1 <= fps <= 240) or not (0.01 <= slowdown <= 100 and 64 <= panel <= 1600):
                raise ValueError("fps 1-240, slowdown 0.01-100, panel 64-1600")
            if burst_id is None:
                if start is None or end is None:
                    raise ValueError("give burst_id or start and end")
                burst_id = find_burst(experiment, fly, start, end)
                start = end = None
            if engine == "browser" and (start is not None or end is not None):
                raise ValueError("the video button records the whole burst: trimming needs engine=draw")
            burst = load_burst(experiment, fly, burst_id, method=method)
        except LookupError as error:
            return jsonify({"error": error.args[0]}), 404
        except (ValueError, TypeError) as error:
            return jsonify({"error": str(error)}), 400

        name = output_name(fly, burst_id, experiment, trim_range(burst, start, end))
        link = pe_url(experiment, fly=fly, burst_id=burst_id, base_url=request.host_url)

        if engine == "browser":
            def make(path, progress=None):
                return compose(experiment, fly=fly, burst_id=burst_id, output=path, engine="browser",
                               viewport=viewport, fps=fps, progress=progress)
            as_job = not _flag(args, "wait")
        else:
            n_out = len(plan_frames(len(burst["trace"]["points"]), int(burst["trace"]["start_frame"]),
                                    float(burst["trace"]["fps"]), fps or DEFAULT_FPS, slowdown, start, end))
            if n_out > MAX_FRAMES:
                return jsonify({"error": f"{n_out} output frames, more than {MAX_FRAMES}"}), 400
            options = dict(panel=panel, fps=fps or DEFAULT_FPS, slowdown=slowdown,
                           overlay=_flag(args, "overlay"), start=start, end=end)

            def make(path, progress=None):
                summary = render(burst, path, progress=progress, **options)
                summary.update({"fly": fly, "burst_id": int(burst_id), "engine": "draw"})
                return summary
            as_job = _flag(args, "job") or n_out > MAX_SYNC_FRAMES

        if as_job:
            def work(job):
                job.workdir = tempfile.mkdtemp(prefix="pe_composition_")
                job.update(0, status="running")
                path = os.path.join(job.workdir, name)
                summary = make(path, progress=lambda p: job.update(0, progress=p))
                job.update(0, status="done", progress=1.0)
                job.result = dict(summary, link=link)
                job.result_path, job.download_name = path, name
            job = start_job(f"{fly}: PE burst {burst_id} video", [name], work)
            return jsonify(dict(job.to_dict(), link=link)), 202

        workdir = tempfile.mkdtemp(prefix="pe_composition_")
        path = os.path.join(workdir, name)
        try:
            summary = make(path)
        except Exception as error:
            shutil.rmtree(workdir, ignore_errors=True)
            logger.exception("PE composition of %s burst %s failed", fly, burst_id)
            return jsonify({"error": str(error)}), 500
        handle = open(path, "rb")               # the open handle outlives the deleted file
        shutil.rmtree(workdir, ignore_errors=True)
        response = send_file(handle, mimetype="video/mp4", as_attachment=True, download_name=name)
        response.headers["X-Render-Seconds"] = str(summary.get("seconds", summary.get("total_s")))
        response.headers["X-PE-Link"] = link
        return response


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m idtrackerai_validator_server.pe_composition",
        description="Render the PE tab's burst video (clip | pose | traces) to an mp4: by default the "
                    "tab's own video button recorded in headless Chrome. Needs $FLYHOSTEL_VIDEOS.")
    parser.add_argument("experiment", help="FlyHostel3_3X_2024-12-10_16-00-00 or FlyHostel3/3X/2024-12-10_16-00-00")
    parser.add_argument("identity", type=int)
    parser.add_argument("burst_id", type=int, nargs="*",
                        help="one or more bursts (one browser session); omit and give --start/--end to locate one")
    parser.add_argument("-o", "--output", default=".",
                        help="mp4 path (one burst) or folder (default: here, <fly>_burst_<id>_frame_<a>-<b>.mp4)")
    parser.add_argument("--start", type=int, help="first frame (no burst_id: locates the burst; draw engine: trims)")
    parser.add_argument("--end", type=int, help="last frame")
    parser.add_argument("--engine", choices=("browser", "draw"), default="browser",
                        help="browser: the tab's video button in headless Chrome (default, exact); "
                             "draw: cv2/matplotlib redraw (no browser, ~2 s, options below)")
    parser.add_argument("--viewport", default=None, help="browser: window size, e.g. 1920x1080 (default)")
    parser.add_argument("--fps", type=float, default=None, help="output frame rate (default 30, as the button)")
    parser.add_argument("--slowdown", type=float, default=1.0, help="draw: play the burst this many times slower")
    parser.add_argument("--panel", type=int, default=DEFAULT_PANEL, help="draw: px of each square panel")
    parser.add_argument("--overlay", action="store_true", help="draw: pose over the video too")
    parser.add_argument("--no-segments", action="store_true", help="draw: leave out the auto / GT lanes")
    parser.add_argument("--method", default="visible",
                        help="draw: segmentation method shown (default %(default)s, as the tab)")
    parser.add_argument("--crf", type=int, default=18, help="draw: x264 quality")
    parser.add_argument("--link", action="store_true", help="only print the PE tab link of the burst(s)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    viewport = _parse_viewport(args.viewport)
    fly = fly_name(args.experiment, args.identity)
    bursts = list(args.burst_id) or [None]

    if args.link:
        from idtrackerai_validator_server.links import pe_url
        for b in bursts:
            if b is None and args.start is not None and args.end is not None:
                b = find_burst(args.experiment, fly, args.start, args.end)
            print(pe_url(args.experiment, fly=fly, burst_id=b))
        return

    if args.engine == "browser" and len(bursts) > 1:
        from idtrackerai_validator_server.pe_record import record_bursts, VIEWPORT, RECORD_FPS
        os.makedirs(args.output, exist_ok=True)
        todo = []
        for b in bursts:
            load_burst(args.experiment, fly, b, method=None)          # fail early on a missing burst
            todo.append((fly, b, os.path.join(args.output, output_name(fly, b, args.experiment))))
        summary = record_bursts(args.experiment, todo, viewport=viewport or VIEWPORT, fps=args.fps or RECORD_FPS)
    else:
        options = {} if args.engine == "browser" else dict(
            panel=args.panel, slowdown=args.slowdown, overlay=args.overlay, crf=args.crf)
        output = args.output if len(bursts) == 1 else (os.makedirs(args.output, exist_ok=True) or args.output)
        summary = [compose(args.experiment, args.identity, b, output=output,
                           start=args.start, end=args.end, engine=args.engine, viewport=viewport,
                           method=None if args.no_segments else args.method, fps=args.fps, **options)
                   for b in bursts]
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
