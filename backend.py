import os.path
import glob
import json
import datetime
import tempfile
import logging
import math

import cv2
import numpy as np
import pandas as pd
from imgstore.interface import VideoCapture

from constants import database_pattern
from idtrackerai.animals_detection.segmentation import _process_frame

def process_config(config):

    user_defined_parameters = {
        "number_of_animals": int(config["_number_of_animals"]["value"]),
        "min_threshold": config["_intensity"]["value"][0],
        "max_threshold": config["_intensity"]["value"][1],
        "min_area": config["_area"]["value"][0],
        "max_area": config["_area"]["value"][1],
        "check_segmentation": True,
        "tracking_interval": [0, math.inf],
        "apply_ROI": True,
        "rois": config["_roi"]["value"],
        "subtract_bkg": False,
        "bkg_model": None,
        "resolution_reduction": config["_resreduct"]["value"],
        "identity_transfer": False,
        "identification_image_size": None,
    }
    return user_defined_parameters


def process_frame(frame, config):
    config=process_config(config)

    roi_mask = np.zeros_like(frame)
    roi_contour = np.array(eval(config["rois"][0][0])).reshape((-1, 1, 2))
    roi_mask = cv2.drawContours(roi_mask, [roi_contour], -1, 255, -1)
    config["mask"]=roi_mask
    # cv2.imwrite("mask.png", roi_mask)
    config["resolution_reduction"]=1.0

    (
        bounding_boxes,
        miniframes,
        centroids,
        areas,
        pixels,
        contours,
        estimated_body_lengths
    ) = _process_frame(
        frame,
        config,
        0,
        "NONE",
        "NONE",
    )

    contours_list = [contour.tolist() for contour in contours]
    return contours_list


def filter_by_date(experiment):
    date_time = os.path.basename(experiment)[:10]
    dt = datetime.datetime.strptime(date_time, "%Y-%m-%d")
    return dt >= datetime.datetime.strptime("2023-05-23", "%Y-%m-%d")


def list_experiments():
    
    experiments = []
    
    with open(os.path.join(os.environ["FLYHOSTEL_VIDEOS"], "index.txt"), "r") as filehandle:
        experiments = [experiment.strip() for experiment in filehandle.readlines()]
        experiments = [os.path.sep.join(experiment.split(os.path.sep)[-4:-1]) for experiment in experiments]
        experiments = sorted([experiment for experiment in experiments if filter_by_date(experiment)])
    
    experiments.append("FlyHostel1/1X/2023-06-16_14-00-00")
    return {"experiments": experiments}


def generate_database_filename(experiment):
    sqlite_file=os.path.join(
        os.environ["FLYHOSTEL_VIDEOS"], experiment, experiment.replace("/", "_") + ".db"
    )
    return sqlite_file


def load_experiment(experiment, chunk, TABLES):
    
    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], experiment)
    if not os.path.exists(basedir):
        return {"message": f"{basedir} does not exist"}, None

    store_path = os.path.join(basedir, "metadata.yaml")
    idtrackerai_config_file = os.path.join(basedir, os.path.basename(basedir) + ".conf")
    
    if os.path.exists(idtrackerai_config_file):
        logging.info(f"Loading {idtrackerai_config_file}")
        with open(idtrackerai_config_file, "r") as fh:
            idtrackerai_config = json.load(fh)
    else:
        idtrackerai_config=None

    cap = VideoCapture(store_path, chunk)  # Replace with your video file

    if cap is None:
        metadata=[None, None]
    else:
        metadata_table=TABLES[experiment]["METADATA"]
        try:
            metadata=load_experiment_metadata(metadata_table)
            chunksize=metadata[2]
            frame_number = chunksize * chunk
            frame, (frame_number, frame_timestamp) = cap.get_image(frame_number)
        
        except Exception as error:
            metadata = (None, None, None)

    return {"message": "success"}, cap , metadata, idtrackerai_config


def load_experiment_metadata(table):
    out = table.query.filter_by(field="date_time")
    experiment_start_time=int(float(out.all()[0].value)) % (24*3600)
    out = table.query.filter_by(field="ethoscope_metadata")

    ethoscope_metadata=out.all()[0].value
    ethoscope_metadata=str2pandas(ethoscope_metadata)
    reference_hour=ethoscope_metadata["reference_hour"].values
    assert np.all(np.diff(reference_hour) == 0)
    reference_hour=reference_hour[0].item()
    offset = experiment_start_time - reference_hour*3600

    out = table.query.filter_by(field="chunksize")
    chunksize=int(float(out.all()[0].value))

    out = table.query.filter_by(field="framerate")
    framerate=int(float(out.all()[0].value))
    return offset, chunksize, framerate


def str2pandas(string):
    temp_file=tempfile.NamedTemporaryFile(mode="w", suffix=".csv", prefix="fh_viewer_metadata_")

    with open(temp_file.name, "w") as filehandle:
        filehandle.write(string)

    return pd.read_csv(temp_file.name, index_col=0)

