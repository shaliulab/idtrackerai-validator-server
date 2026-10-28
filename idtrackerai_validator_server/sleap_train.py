"""SLEAP model training (sleap-train, SLEAP 1.3.3) on a labels file of $SLEAP_LABELS_DIR, run
through the GPU queue (gpu_queue.py).

A training repeats a previous model's: the `initial_config.json` of a model folder of
MODELS_DIR (default the newest complete one: best_model.h5 + initial_config.json), with the
labels file as training labels (validation_fraction as it was), outputs in MODELS_DIR under
SLEAP's own name (`<yymmdd_HHMMSS>.<head>.n=<frames>`), no zmq (the GUI's progress socket)
and no tensorboard. The config and the log go to TRAIN_DIR (`<labels>_<time>.json/.log`);
sleap-train runs in SLEAP_LABELS_DIR (videos paths relative to it resolve). The file is
saved first, so the training sees every label made so far; labels made while it waits are in
it too (the file is read when it starts).

Progress: "Created run path: ..." in the log, then the rows of its training_log.csv against
optimization.epochs (training usually stops earlier: plateau_patience).

    GET  /api/sleap/models          models of MODELS_DIR, newest first
    POST /api/sleap/train           {file, base_model}: queue a training (202-like: the queue entry)
    GET  /api/sleap/train?file=     the file's queued / running training
"""
import csv
import datetime
import glob
import json
import os
import re

import psutil
from flask import jsonify, request

from idtrackerai_validator_server import gpu_queue
from idtrackerai_validator_server.sleap_labels import LABELS_DIR, file_path, get_file

SLEAP_TRAIN = os.environ.get("SLEAP_TRAIN", os.path.expanduser("~/mambaforge/envs/sleap/bin/sleap-train"))
MODELS_DIR = os.environ.get("SLEAP_MODELS_DIR", os.path.join(LABELS_DIR, "models"))
TRAIN_DIR = os.environ.get("SLEAP_TRAIN_DIR", os.path.expanduser("~/.cache/idtrackerai-validator/sleap_train"))
# GPU queue estimate until a run has been measured (sleap-train lets TensorFlow grow its memory)
SLEAP_TRAIN_MB = float(os.environ.get("SLEAP_TRAIN_MB", 12000))
RUN_PATH_RE = re.compile(r"Created run path: (.+)$", re.M)
MODEL_RE = re.compile(r"^(\d{6}_\d{6})\.([a-z_]+)\.n=(\d+)")


def list_models():
    models = []
    for folder in glob.glob(os.path.join(MODELS_DIR, "*")):
        name = os.path.basename(folder)
        match = MODEL_RE.match(name)
        config = os.path.join(folder, "initial_config.json")
        if not match or not os.path.isdir(folder) or not os.path.exists(config):
            continue
        best = os.path.join(folder, "best_model.h5")
        models.append({"name": name, "created": datetime.datetime.strptime(match.group(1), "%y%m%d_%H%M%S").timestamp(),
                       "head": match.group(2), "n": int(match.group(3)), "complete": os.path.exists(best),
                       **_training_log(folder)})
    models.sort(key=lambda m: -m["created"])
    return models


def _training_log(folder):
    """Epochs done and the best val_loss of a model folder's training_log.csv."""
    path = os.path.join(folder, "training_log.csv")
    try:
        with open(path) as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return {"epochs": None, "val_loss": None}
    losses = [float(r["val_loss"]) for r in rows if r.get("val_loss") not in (None, "")]
    return {"epochs": len(rows), "val_loss": min(losses) if losses else None}


def training_config(base_model, labels_path):
    with open(os.path.join(MODELS_DIR, base_model, "initial_config.json")) as handle:
        config = json.load(handle)
    config["data"]["labels"].update(training_labels=labels_path, validation_labels=None, test_labels=None)
    outputs = config["outputs"]
    outputs.update(runs_folder=MODELS_DIR, run_name=None, run_name_prefix="", run_name_suffix=None)
    outputs.setdefault("zmq", {}).update(subscribe_to_controller=False, publish_updates=False)
    outputs.setdefault("tensorboard", {}).update(write_logs=False)
    return config


def launch(entry):
    params = entry["params"]
    os.makedirs(TRAIN_DIR, exist_ok=True)
    config_path = entry["log"][:-len(".log")] + ".json"
    with open(config_path, "w") as handle:
        json.dump(training_config(params["base_model"], params["labels"]), handle, indent=2)
    cmd = [SLEAP_TRAIN, config_path, params["labels"]]
    return gpu_queue.spawn(cmd, LABELS_DIR, entry["log"], env=dict(os.environ, PYTHONUNBUFFERED="1"))


def progress(entry):
    info = entry.setdefault("info", {})
    if not info.get("run_path"):
        try:
            with open(entry["log"], errors="replace") as handle:
                match = RUN_PATH_RE.search(handle.read())
        except OSError:
            match = None
        if match:
            info["run_path"] = match.group(1).strip()
    if not info.get("epochs"):
        try:
            with open(entry["log"][:-len(".log")] + ".json") as handle:
                info["epochs"] = json.load(handle)["optimization"]["epochs"]
        except (OSError, ValueError, KeyError):
            pass
    log = _training_log(info["run_path"]) if info.get("run_path") else {"epochs": None, "val_loss": None}
    if not info.get("epochs"):
        return None
    return {"done": log["epochs"] or 0, "total": info["epochs"], "unit": "epochs", "val_loss": log["val_loss"],
            "run": os.path.basename(info["run_path"]) if info.get("run_path") else None}


def external():
    processes = [p for p in psutil.process_iter(["pid", "ppid", "cmdline", "create_time"])
                 if any(os.path.basename(part) == "sleap-train" for part in (p.info["cmdline"] or [])[:2])]
    pids = {p.pid for p in processes}
    return [{"pid": p.pid, "what": " ".join(p.info["cmdline"][-1:]), "started": p.info["create_time"]}
            for p in processes if p.info["ppid"] not in pids]


def register_sleap_train(app):
    gpu_queue.register_kind("sleap_train", "sleap-train", memory_mb=SLEAP_TRAIN_MB, launch=launch,
                            progress=progress, external=external, max_running=1)

    @app.route("/api/sleap/models", methods=["GET"])
    def sleap_models():
        return jsonify({"models": list_models(), "models_dir": MODELS_DIR})

    @app.route("/api/sleap/train", methods=["GET"])
    def sleap_train_status():
        return jsonify({"entry": gpu_queue.find("sleap_train", request.args.get("file"))})

    @app.route("/api/sleap/train", methods=["POST"])
    def sleap_train_start():
        body = request.get_json(force=True)
        try:
            labels = file_path(body.get("file"))
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 400
        if not body["file"].startswith("labels_train"):
            return jsonify({"error": "Train on a labels_train*.slp file"}), 400
        models = [m for m in list_models() if m["complete"]]
        base = body.get("base_model") or (models[0]["name"] if models else None)
        if base is None or not any(m["name"] == base for m in models):
            return jsonify({"error": f"No complete model {base!r} in {MODELS_DIR} to take the config from"}), 400
        if not os.path.exists(SLEAP_TRAIN):
            return jsonify({"error": f"{SLEAP_TRAIN} not found (set SLEAP_TRAIN)"}), 400
        labels_file = get_file(body["file"])
        try:
            labels_file.save()
        except RuntimeError as error:
            return jsonify({"error": f"Could not save {body['file']} first: {error}"}), 409
        stamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        log = os.path.join(TRAIN_DIR, f"{os.path.splitext(body['file'])[0]}_{stamp}.log")
        try:
            entry = gpu_queue.enqueue("sleap_train", f"{body['file']} (config of {base})", body["file"],
                                      {"labels": labels, "base_model": base}, log)
        except FileExistsError as error:
            return jsonify({"error": str(error)}), 409
        return jsonify({"queued": entry})
