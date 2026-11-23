"""Every lab-specific path, host and tool location the validator uses, in one place.

    from idtrackerai_validator_server import lab_paths
    lab_paths.SLEAP_LABELS_DIR

Each entry is read once, when this module is first imported: the environment variable named next
to it, else the default shown (the lab server's layout). To run the validator elsewhere, set the
variables; do not edit the defaults. `~` is the user's home folder (os.path.expanduser: %USERPROFILE%
on Windows, never $HOME). Where the existing code expanded `~` in the variable's value too, the entry
uses `_user_env` (else `_env`): kept exactly as before.

Not here: $FLYHOSTEL_VIDEOS (required, read where it is used: `os.environ["FLYHOSTEL_VIDEOS"]`), paths
inside an experiment folder (`<basedir>/flyhostel/validation/...`: the pipeline's layout, not the lab's),
and the system's temporary folder (tempfile: $TMPDIR / %TEMP%; the validator makes its temporary
files and folders there with prefixes `pe_record_`, `pe_composition_`, `fh_viewer_metadata_`, and
the Drive exchange's in $EXCHANGE_CACHE).
"""
import os
import shutil

from idtrackerai_validator_server.oscompat import find_tool


def _env(name, default):
    """$name, else default (as given: a `~` in $name is not expanded)."""
    return os.environ.get(name, default)


def _user_env(name, default):
    """$name, else default, with `~` expanded in either."""
    return os.path.expanduser(os.environ.get(name, default))


HOME = os.path.expanduser("~")


# --- Data roots on the lab server ---------------------------------------------------------------
# $FLYHOSTEL_VIDEOS is required by the server; only the Monitor tab falls back to this default
MONITOR_FLYHOSTEL_VIDEOS = _env("FLYHOSTEL_VIDEOS", "/flyhostel_data/videos")
# copy of $FLYHOSTEL_VIDEOS the Data management tab compares experiments with
FLYHOSTEL_BACKUP = _env("FLYHOSTEL_BACKUP", "/flyhostel_data/backup")
# CSV exports of the metadata sheets (Overview: experiments not analyzed yet)
FLYHOSTEL_METADATA = _env("FLYHOSTEL_METADATA", "/flyhostel_data/metadata")
# deepethogram projects (DEG labels: <DEG_PROJECTS>/DEG/FlyHostel_deepethogram[_47fps]; Dropbox restore)
DEG_PROJECTS = _env("DEG_PROJECTS", "/flyhostel_data/fiftyone/FlyBehaviors")
# SLEAP training files (labels_train*.slp, SLEAP_DATA/)
SLEAP_LABELS_DIR = _env("SLEAP_LABELS_DIR", "/flyhostel_data/fiftyone/FlyBehaviors")
# SLEAP models (sleap-train's runs folder)
SLEAP_MODELS_DIR = _env("SLEAP_MODELS_DIR", os.path.join(SLEAP_LABELS_DIR, "models"))
# output of the middle-leg movement detector (Leg movements tab)
LEG_MOVEMENTS_DIR = _env("LEG_MOVEMENTS_DIR", "/flyhostel_data/sync_paper_data/modelling/leg_movements")
# reviews of the Leg movements tab (default: leg_annotations.db next to the PE annotations, see pe_validation)
LEG_ANNOTATIONS_DB = os.environ.get("LEG_ANNOTATIONS_DB")       # None: next to PE_ANNOTATIONS_DB
# ethoscope result databases (<machine_id>/<machine_name>/<date>_<time>/*.db)
ETHOSCOPE_RESULTS = _env("ETHOSCOPE_RESULTS", "/ethoscope_data/results")
# audit of the PE classifier (PE tab's audit mode; calib.csv was the earlier default:
# /home/vibflysleep/FlySleepLab_Dropbox/Antonio/FSLLab/Projects/FlyHostel4/code/scripts/proboscis_extension/v3/calib.csv)
PE_AUDIT_CSV = _env("PE_AUDIT_CSV", "/home/vibflysleep/FlySleepLab_Dropbox/Antonio/FSLLab/Projects/FlyHostel4/"
                                    "code/scripts/proboscis_extension/v3/pnf_audit.csv")


def pe_annotations_db_path():
    """$PE_ANNOTATIONS_DB, else pe_annotations.db in $FLYHOSTEL_VIDEOS, else in $XDG_DATA_HOME/flyhostel
    (default ~/.local/share/flyhostel, made when missing)."""
    if os.environ.get("PE_ANNOTATIONS_DB"):
        return os.environ["PE_ANNOTATIONS_DB"]
    videos = os.environ.get("FLYHOSTEL_VIDEOS")
    if videos:
        return os.path.join(videos, "pe_annotations.db")
    base = os.environ.get("XDG_DATA_HOME", os.path.join(HOME, ".local", "share"))
    path = os.path.join(base, "flyhostel")
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, "pe_annotations.db")


# --- Lab tools: conda environments, executables and scripts ---------------------------------------
MAMBAFORGE = os.path.join(HOME, "mambaforge")
CONDA_ENVS = os.path.join(MAMBAFORGE, "envs")
# validation pipeline env (make-identogram, auto-annotate-qc, integrate-human-annotations, ...): Annotate tab, CVAT launch
FLYHOSTEL_VALIDATION_ENV = _user_env("FLYHOSTEL_VALIDATION_ENV", "~/mambaforge/envs/rapids-23.04")
# vsc-scripts' manual validation (main.sh: send an experiment to CVAT)
FLYHOSTEL_VALIDATION_SCRIPTS = _user_env("FLYHOSTEL_VALIDATION_SCRIPTS", "~/opt/vsc-scripts/manual/validation")
# YOLOv7 + solver proposal of the Annotate tab (~/opt/yolov7/yolov7/workflow/propose.py, installed in the yolov7 env)
YOLOV7_CVAT = _user_env("YOLOV7_CVAT", "~/mambaforge/envs/yolov7/bin/yolov7-cvat")
# python with deepethogram (makes missing DEG video folders)
DEG_PYTHON = _env("DEG_PYTHON", os.path.join(CONDA_ENVS, "rapids-23.04", "bin", "python"))
# metadata sheet downloader; its folder also holds download_metadata_sheets and the python of the batch download
DOWNLOAD_FLYHOSTEL_METADATA = _env("DOWNLOAD_FLYHOSTEL_METADATA",
                                   "/home/vibflysleep/mambaforge/envs/rapids-23.04/bin/download_flyhostel_metadata")
# SLEAP env (compress_slp.py's check, sleap-train)
SLEAP_PYTHON = _env("SLEAP_PYTHON", os.path.join(CONDA_ENVS, "sleap", "bin", "python"))
SLEAP_TRAIN = _env("SLEAP_TRAIN", os.path.join(CONDA_ENVS, "sleap", "bin", "sleap-train"))
# h5repack: $H5REPACK, else on PATH, else mambaforge's base env
H5REPACK = os.environ.get("H5REPACK") or shutil.which("h5repack") or os.path.join(MAMBAFORGE, "bin", "h5repack")
# Data collector: roi-proposer (its env's python also runs detect_arenas), idtrackerai v4 (conf-convert,
# preprocess-sample), imgstore-crop-arenas (flyhostel-pipeline env: ffmpeg with NVENC)
ROI_PROPOSER = _env("ROI_PROPOSER", os.path.join(CONDA_ENVS, "roi-proposer", "bin", "roi-proposer"))
IDTRACKERAI_V4_BIN = _env("IDTRACKERAI_V4_BIN", os.path.join(CONDA_ENVS, "idtrackeraiv4", "bin"))
CROP_ARENAS = _env("CROP_ARENAS", os.path.join(CONDA_ENVS, "flyhostel-pipeline", "bin", "imgstore-crop-arenas"))
# nextflow pipelines shown in the Monitor tab
NEXTFLOW_PIPELINES = _env("NEXTFLOW_PIPELINES", "/home/vibflysleep/opt/vsc-scripts/nextflow/pipelines")
# executables found by name: $VAR, else PATH (rclone / ffmpeg: else the server's conda env, Library\bin on Windows)
RCLONE = find_tool("rclone", "RCLONE")
FFMPEG = find_tool("ffmpeg", "FFMPEG")
NVIDIA_SMI = _env("NVIDIA_SMI", "nvidia-smi")
PE_RECORD_CHROME = _env("PE_RECORD_CHROME", "google-chrome")


# --- Remote hosts and remotes (ssh aliases of ~/.ssh/config, rclone remotes, URLs) ----------------
# recording computer (Data collector fetch): ssh alias, its videos folder, its read-only http listing
COLLECTOR_REMOTE_HOST = _env("COLLECTOR_REMOTE_HOST", "cv2")
COLLECTOR_REMOTE_ROOT = _env("COLLECTOR_REMOTE_ROOT", "/home/vibflysleep/flyhostel_data/videos")
COLLECTOR_HTTP = _env("COLLECTOR_HTTP", "http://10.43.205.5:8000")
# ethoscope computer (Ethoscope tab's download): ssh alias and its results folder
ETHOSCOPE_REMOTE_HOST = _env("ETHOSCOPE_REMOTE_HOST", "cv1")
ETHOSCOPE_REMOTE_RESULTS = _env("ETHOSCOPE_REMOTE_RESULTS", "/ethoscope_data/results")
# rclone path of the Dropbox copy of $FLYHOSTEL_VIDEOS (the leading slash reaches the team folder)
DROPBOX_REMOTE = _env("DROPBOX_REMOTE", "dropbox:/Data/flyhostel_data/videos").rstrip("/")
# rclone remote + folder shared with the annotators (Drive exchange)
GDRIVE_REMOTE = _env("GDRIVE_REMOTE", "gdrive:FlyHostelExchange")
# CVAT ($CVAT_HOST, $CVAT_USERNAME, $CVAT_PASSWORD, $CVAT_PUBLIC_URL; no default host) and the incubator
# ($FLYHOSTEL_ENVIRONMENT, $INCUBATOR_URL) are read where they are used (cvat.py) or by flyhostel.


# --- This program's configuration (~/.config) -------------------------------------------------------
CONFIG_DIR = os.path.join(HOME, ".config", "idtrackerai-validator")
# key signing the session cookie (made on first start)
VALIDATOR_SECRET_KEY_FILE = _env("VALIDATOR_SECRET_KEY_FILE", os.path.join(CONFIG_DIR, "secret_key"))
# TLS certificate made by scripts/make_tls_certificates.sh (cert.pem, key.pem, ca.pem); the server uses
# $VALIDATOR_CERTFILE / $VALIDATOR_KEYFILE, links.py only checks whether this one exists
DEFAULT_TLS_CERTFILE = os.path.join(CONFIG_DIR, "tls", "cert.pem")
# Drive exchange settings saved by the tab (remote, annotator name)
EXCHANGE_CONFIG = _user_env("EXCHANGE_CONFIG", "~/.config/idtrackerai-validator/exchange.json")
# experiments each person hid from the Annotate tab's validation queue
ANNOTATE_QUEUE_FILE = _env("ANNOTATE_QUEUE_FILE", os.path.join(CONFIG_DIR, "annotate_queue.json"))
# googleapi's list of metadata sheets (SHEETS); its token lives next to it
GOOGLEAPI_CONFIG = _user_env("GOOGLEAPI_CONFIG", "~/.config/googleapi/config.yaml")


# --- Caches, logs and state (~/.cache) --------------------------------------------------------------
CACHE_DIR = os.path.join(HOME, ".cache", "idtrackerai-validator")
# QC tab (qc.py; swap check in swaps/, pose QC in pose/)
QC_CACHE_DIR = _env("QC_CACHE_DIR", os.path.join(CACHE_DIR, "qc"))
# Annotate tab's easy scenes
CONTINUITY_CACHE_DIR = _env("CONTINUITY_CACHE_DIR", os.path.join(CACHE_DIR, "continuity"))
# last Data management scan (dropbox.json, dropbox_hashes.json, restore.json, ... next to it)
DATA_MANAGEMENT_CACHE = _user_env("DATA_MANAGEMENT_CACHE", "~/.cache/idtrackerai-validator/data_management.json")
# Drive exchange downloads and minimal bundles
EXCHANGE_CACHE = _user_env("EXCHANGE_CACHE", "~/.cache/idtrackerai-validator/exchange")
# GPU queue state (.lock and .scheduler next to it)
GPU_QUEUE_FILE = _env("GPU_QUEUE_FILE", os.path.join(CACHE_DIR, "gpu_queue.json"))
# scheduled fetches of the Data collector (.lock and .scheduler next to it)
COLLECTOR_SCHEDULE_FILE = _env("COLLECTOR_SCHEDULE_FILE", os.path.join(CACHE_DIR, "fetch_schedule.json"))
# logs of DEG label writes, SLEAP label changes and copies of .slp files before their first write
DEG_LABELS_LOG = _env("DEG_LABELS_LOG", os.path.join(CACHE_DIR, "deg_labels.jsonl"))
SLEAP_LABELS_LOG = _env("SLEAP_LABELS_LOG", os.path.join(CACHE_DIR, "sleap_labels.jsonl"))
SLEAP_BACKUPS = _env("SLEAP_BACKUPS", os.path.join(CACHE_DIR, "slp_backups"))
# configs and logs of the SLEAP trainings started here
SLEAP_TRAIN_DIR = _env("SLEAP_TRAIN_DIR", os.path.join(CACHE_DIR, "sleap_train"))
# CLIs: compressed / uploaded .slp files
COMPRESS_SLP_LOG = _user_env("COMPRESS_SLP_LOG", "~/.cache/idtrackerai-validator/compressed_slp.jsonl")
UPLOAD_SLP_LOG = _user_env("UPLOAD_SLP_LOG", "~/.cache/idtrackerai-validator/uploaded_slp.jsonl")
