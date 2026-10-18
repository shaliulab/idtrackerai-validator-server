"""ROIs and segmentation settings of an experiment folder, for /api/rois and
/api/preprocess?source=.

Sources, in this order (those present):
    database        idtrackerai_conf of the database (idtrackerai v4 .conf format)
    <date>.toml     idtrackerai v5 settings the experiment was tracked with
    landmarks.toml  idtrackerai v5 settings marking the landmarks (food, ...)
    <date>.conf     idtrackerai v4 settings (used without database)

A v5 roi_list entry is "+ Polygon [[x, y], ...]" or "+ Ellipse {'center': .., 'axes': ..,
'angle': ..}" ("-" removes it from the mask); ROIs are returned as vertices, computed as
idtrackerai does (utils.py_utils.get_vertices_from_label / build_ROI_mask_from_list).
"""
import ast
import json
import logging
import math
import os

import cv2
import numpy as np

try:
    import tomllib
except ImportError:  # python < 3.11
    import tomli as tomllib

from idtrackerai_validator_server.experiments import experiment_folder

logger = logging.getLogger(__name__)


def _finite(value):
    """inf (area_ths = [50, inf]) is not valid JSON."""
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def roi_vertices(label):
    """(sign, shape, vertices) of a v5 roi_list entry."""
    label = label.strip()
    sign, shape = label[0], label[2:9]
    data = json.loads(label[10:].replace("'", '"'))
    if shape == "Polygon":
        vertices = np.asarray(data, dtype=float)
    elif shape == "Ellipse":
        vertices = np.asarray(cv2.ellipse2Poly(
            [int(v) for v in data["center"]], [int(v) for v in data["axes"]], int(data["angle"]), 0, 360, 2,
        ), dtype=float)
    else:
        raise ValueError(f"Not recognized ROI: {label!r}")
    if sign not in "+-":
        raise ValueError(f"ROI must start with + or -: {label!r}")
    return sign, shape, vertices


def _from_toml(source_id, path):
    with open(path, "rb") as handle:
        settings = tomllib.load(handle)
    rois = []
    for label in settings.get("roi_list") or []:
        sign, shape, vertices = roi_vertices(label)
        rois.append({"label": label, "sign": sign, "shape": shape, "vertices": vertices.tolist()})
    intensity = settings.get("intensity_ths") or [None, None]
    area = settings.get("area_ths") or [None, None]
    return {
        "id": source_id, "path": path, "format": "toml",
        "settings": {
            "intensity": [_finite(v) for v in intensity],
            "area": [_finite(v) for v in area],
            "number_of_animals": settings.get("number_of_animals"),
            "resolution_reduction": settings.get("resolution_reduction", 1.0),
            "use_bkg": bool(settings.get("use_bkg", False)),
            "track_wo_identities": bool(settings.get("track_wo_identities", False)),
            "exclusive_rois": bool(settings.get("exclusive_rois", False)),
            "video_paths": settings.get("video_paths") or [],
        },
        "rois": rois,
        "_segmentation": {
            "min_threshold": intensity[0], "max_threshold": intensity[1],
            "min_area": area[0], "max_area": area[1],
        },
    }


def _from_conf(source_id, path, config):
    """idtrackerai v4 config: _roi = [["[x, y], [x, y], ..."], ...], always added."""
    rois = []
    apply_roi = str(config.get("_applyroi", {}).get("value", "True")) == "True"
    for entry in config.get("_roi", {}).get("value") or []:
        text = entry[0] if isinstance(entry, (list, tuple)) else entry
        vertices = np.asarray(ast.literal_eval(f"[{text}]"), dtype=float)
        rois.append({"label": "+ Polygon", "sign": "+", "shape": "Polygon", "vertices": vertices.tolist()})
    intensity = config["_intensity"]["value"]
    area = config["_area"]["value"]
    return {
        "id": source_id, "path": path, "format": "conf",
        "settings": {
            "intensity": [_finite(v) for v in intensity],
            "area": [_finite(v) for v in area],
            "number_of_animals": int(config["_number_of_animals"]["value"]),
            "resolution_reduction": config.get("_resreduct", {}).get("value", 1.0),
            "use_bkg": str(config.get("_bgsub", {}).get("value", "False")) == "True",
            "apply_roi": apply_roi,
        },
        "rois": rois if apply_roi else [],
        "_segmentation": {
            "min_threshold": intensity[0], "max_threshold": intensity[1],
            "min_area": area[0], "max_area": area[1],
        },
    }


def _load(source_id, path, loader):
    try:
        source = loader()
        source["mtime"] = os.path.getmtime(path) if path else None
        return source
    except Exception as error:
        logger.warning("Cannot read %s: %s", path or source_id, error)
        return {"id": source_id, "path": path, "error": f"{type(error).__name__}: {error}", "rois": []}


def list_sources(basedir, relpath, db_config=None):
    """Every source of the experiment folder `basedir` (relpath: its slash form);
    db_config: the database's idtrackerai_conf, if any."""
    date_time = os.path.basename(experiment_folder(relpath))
    sources = []
    if db_config is not None:
        sources.append(_load("database", None, lambda: _from_conf("database", None, db_config)))
    for name in (f"{date_time}.toml", "landmarks.toml"):
        path = os.path.join(basedir, name)
        if os.path.exists(path):
            sources.append(_load(name, path, lambda name=name, path=path: _from_toml(name, path)))
    conf = os.path.join(basedir, f"{date_time}.conf")
    if os.path.exists(conf):
        def read_conf():
            with open(conf) as handle:
                return _from_conf(f"{date_time}.conf", conf, json.load(handle))
        sources.append(_load(f"{date_time}.conf", conf, read_conf))
    return sources


def public(source):
    """Source without the internal segmentation parameters."""
    return {k: v for k, v in source.items() if not k.startswith("_")}


def segment(frame, source):
    """(contours, areas) idtrackerai's segmentation finds in `frame` with the settings
    of `source` (resolution reduction ignored: contours stay in frame pixels). The
    background model is not available here: settings with use_bkg are applied without it."""
    from idtrackerai.animals_detection.segmentation import _process_frame

    seg = source["_segmentation"]
    shape = frame.shape[:2]
    if source["rois"]:
        mask = np.zeros(shape, np.uint8)
        for roi in source["rois"]:
            vertices = (np.asarray(roi["vertices"]) + 0.5).astype(np.int32)
            cv2.fillPoly(mask, (vertices,), color=[255 if roi["sign"] == "+" else 0])
    else:
        mask = np.full(shape, 255, np.uint8)

    params = {
        "min_threshold": seg["min_threshold"], "max_threshold": seg["max_threshold"],
        "min_area": seg["min_area"], "max_area": seg["max_area"],
        "mask": mask, "subtract_bkg": False, "bkg_model": None, "resolution_reduction": 1.0,
    }
    _, _, _, areas, _, contours, _ = _process_frame(frame, params, 0, "NONE", "NONE")
    return [contour.tolist() for contour in contours], [float(a) for a in areas]
