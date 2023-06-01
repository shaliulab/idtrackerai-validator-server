import os.path
import glob
import datetime

from imgstore.interface import VideoCapture
from constants import database_pattern
def filter_by_date(experiment):
    date_time = os.path.basename(experiment)[:10]
    print(date_time)
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


def load_experiment(experiment, chunk):
    
    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], experiment)
    if not os.path.exists(basedir):
        return {"message": f"{basedir} does not exist"}, None

    store_path = os.path.join(basedir, "metadata.yaml")
    cap = VideoCapture(store_path, chunk)  # Replace with your video file
    
    frame_number = 45000 * 100
    frame, (frame_number, frame_timestamp) = cap.get_image(frame_number)
    return {"message": "success"}, cap