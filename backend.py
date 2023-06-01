import os.path
import glob
import datetime
import tempfile

import numpy as np
import pandas as pd
from imgstore.interface import VideoCapture

from constants import database_pattern

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
    return {"experiments": experiments}


def generate_database_filename(experiment):
    return glob.glob(os.path.join(
        os.environ["FLYHOSTEL_VIDEOS"], experiment, database_pattern
    ))[0]


def load_experiment(experiment, chunk, TABLES):
    
    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], experiment)
    if not os.path.exists(basedir):
        return {"message": f"{basedir} does not exist"}, None

    store_path = os.path.join(basedir, "metadata.yaml")
    cap = VideoCapture(store_path, chunk)  # Replace with your video file

    if cap is None:
        metadata=[None, None]
    else:
        metadata_table=TABLES[experiment]["METADATA"]   
        metadata=load_experiment_metadata(metadata_table)
        chunksize=metadata[2]
        frame_number = chunksize * chunk
        frame, (frame_number, frame_timestamp) = cap.get_image(frame_number)
    return {"message": "success"}, cap , metadata


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

