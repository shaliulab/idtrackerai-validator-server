import os.path
import glob
import traceback
import json
import datetime
import tempfile
import logging
import sqlite3
import math
import re

import webcolors
import cv2
import numpy as np
import pandas as pd
from imgstore.interface import VideoCapture

from idtrackerai.utils.py_utils import get_spaced_colors_util
from idtrackerai.animals_detection.segmentation import _process_frame

logger=logging.getLogger(__name__)

RED=webcolors.name_to_rgb("red")[::-1]
GREEN=webcolors.name_to_rgb("green")[::-1]
BLACK=webcolors.name_to_rgb("black")[::-1]


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
    """
    Generate the contours that idtrackerai would obtain using the passed config

    Arguments:

        frame (np.ndarray):
        config (dict): idtrackerai config

    Returns
        contour_list (list): List of contours. See draw_frame on how to draw them on the frame
    """
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



def annotate_text(frame, color, text, org, fontScale=1):

    fontScale = 1
    thickness = 2
    fontFace = cv2.FONT_HERSHEY_SIMPLEX

    textSize = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]

    # Calculate the box in which the text will be placed (x, y, w, h)
    text_box = (org[0], org[1] - textSize[1], textSize[0], textSize[1])

    # Adjust the top-left corner to draw the rectangle
    rect_top_left = (text_box[0], text_box[1] - 5)  # Adjusted for padding
    rect_bottom_right = (text_box[0] + text_box[2], text_box[1] + text_box[3] + 10)  # Adjusted for padding

    # Draw a white rectangle
    cv2.rectangle(frame, rect_top_left, rect_bottom_right, (255, 255, 255), cv2.FILLED)

    # Then, put the text on the image
    frame = cv2.putText(frame, text, org, fontFace, fontScale, color, thickness)

    return frame
    
def annotate_frame(frame, row):
    if row["yolov7_qc"]:
        frame=annotate_text(frame, BLACK, "YOLOv7", (10, 50))
    else:
        frame=annotate_text(frame, RED, "YOLOv7", (10, 50))
    
    if row["inter_qc"]:
        frame=annotate_text(frame, BLACK, "Fragments", (10, 100))
    else:
        frame=annotate_text(frame, RED, "Fragments", (10, 100))

    chunk=row["chunk"]
    frame_number=row["frame_number"]
    frame=annotate_text(frame, BLACK, f"{str(chunk).zfill(6)} - {frame_number}", (700, 50), fontScale=.3)

    return frame



def draw_frame(frame, tracking_data, number_of_animals, blobs=None, field="identity"):
    colors=get_spaced_colors_util(number_of_animals, black=False)

    for i, row in tracking_data.iterrows():
        org=(int(row["x"]), int(row["y"]))
        identity=int(row[field])
        fragment=int(row["fragment"])
        if identity<=0:
            color=(0, 0, 0)
        else:
            color=colors[identity-1]
    
        frame=cv2.putText(
            frame, text=f"{identity} ({fragment})",
            org=org,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            thickness=2,
            color=color,
            fontScale=1,
        )

    if blobs:
        for blob in blobs:
            # TODO
            pass
    
    return frame



def filter_by_date(experiment):
    date_time = os.path.basename(experiment)[:10]
    dt = datetime.datetime.strptime(date_time, "%Y-%m-%d")
    return dt >= datetime.datetime.strptime("2023-05-23", "%Y-%m-%d")


def list_experiments():
    
    experiments = []
    
    with open(os.path.join(os.environ["FLYHOSTEL_VIDEOS"], "index.txt"), "r") as filehandle:
        experiments = [experiment.strip() for experiment in filehandle.readlines()]
        experiments = [path for path in experiments if re.search(r"FlyHostel\d_\d{1,2}X_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.db", os.path.basename(path))]
        experiments = [os.path.sep.join(experiment.split(os.path.sep)[-4:-1]) for experiment in experiments]
        experiments = sorted([experiment for experiment in experiments if filter_by_date(experiment)])
    return {"experiments": experiments}


def generate_database_filename(experiment):
    sqlite_file=os.path.join(
        os.environ["FLYHOSTEL_VIDEOS"], experiment, experiment.replace("/", "_") + ".db"
    )
    return sqlite_file


def load_idtrackerai_config(basedir):
    dbfile = os.path.join(basedir, "_".join(basedir.split(os.path.sep)[-3:]) + ".db")
    with sqlite3.connect(dbfile) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM METADATA WHERE field = 'idtrackerai_conf';")
        config_str = cursor.fetchone()

    idtrackerai_config = json.loads(config_str[0].rstrip('\n'))
    return idtrackerai_config

def load_flyhostel_metadata(basedir_suffix, db_manager):
    metadata_table=db_manager.tables["METADATA"]
    try:
        metadata=load_experiment_metadata(metadata_table)
    except Exception as error:
        logger.error(error)
        logger.error(traceback.print_exc())
        print(basedir_suffix)
        metadata = (None, None, None)

    return metadata

def load_experiment(basedir_suffix, chunk, db_manager):

    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], basedir_suffix)
    if not os.path.exists(basedir):
        logger.error(f"{basedir} not found")
        return {"message": f"{basedir} does not exist"}, None, None, None

    # load idtrackerai_config
    idtrackerai_config=load_idtrackerai_config(basedir)

    # load videocapture object
    store_path = os.path.join(basedir, "metadata.yaml")
    logger.debug("Initializing %s - chunk %s", store_path, chunk)
    cap = VideoCapture(store_path, chunk)  # Replace with your video file

    if cap is None:
        logger.error(f"Could not load VideoCapture {store_path}")
        metadata=(None, None, None)
    else:
        try:

            metadata=load_flyhostel_metadata(basedir_suffix, db_manager)
            chunksize=metadata[1]
            frame_number = chunksize * chunk
            frame, (frame_number, frame_timestamp) = cap.get_image(frame_number)
        except Exception as error:
            logger.error("Cannot load %s", basedir_suffix)
            raise error

    return {"message": "success"}, cap , metadata, idtrackerai_config


def load_experiment_metadata(table):
    out = table.query.filter_by(field="date_time")
    experiment_start_time=int(float(out.all()[0].value)) % (24*3600)
    out = table.query.filter_by(field="ethoscope_metadata")

    ethoscope_metadata=out.all()[0].value
    ethoscope_metadata=str2pandas(ethoscope_metadata)
    assert ethoscope_metadata.shape[0] > 0, f"Ethoscope metadata has no data in"
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
