import os
import shutil
import argparse
import time
import re
from threading import Lock
import logging
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask import g
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
import cv2
from sqlalchemy import func
from sqlalchemy.orm import Session
from flask import session

from idtrackerai_validator_server.constants import (
    WITH_FRAGMENTS, first_chunk, FRAMES_DIR
)
from idtrackerai_validator_server.database import DatabaseManager
from idtrackerai_validator_server.backend import load_experiment, generate_database_filename, process_frame
from flyhostel.utils import (
    get_basedir,
    get_identities
)

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)
logging.getLogger("idtrackerai_validator_server.backend").setLevel(logging.DEBUG)
logging.getLogger("imgstore").setLevel(logging.WARNING)
logging.getLogger("watchdog.observers").setLevel(logging.WARNING)

try:
    SELECTED_EXPERIMENT=os.environ["VALIDATOR_EXPERIMENT"]
except KeyError:
    raise Exception("Please define VALIDATOR_EXPERIMENT to the path to some flyhostel experiment")

USE_VAL=os.environ.get("USE_VAL", None)
if USE_VAL is not None:
    USE_VAL=USE_VAL=="True"

lock=Lock()

# Initialize application with CORS settings
app = Flask(__name__)
app.config['SECRET_KEY'] = 'FLYHOSTEL_1234'
CORS(app)

# Clean up previous frames
if os.path.exists(FRAMES_DIR):
    shutil.rmtree(FRAMES_DIR)

# Database configuration
database_file=generate_database_filename(SELECTED_EXPERIMENT)
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{database_file}"
app.config['SQLALCHEMY_BINDS'] = {
    SELECTED_EXPERIMENT: f"sqlite:///{database_file}"
}
db=SQLAlchemy(app)


db_manager = DatabaseManager(app, db, with_fragments=WITH_FRAGMENTS, experiment=SELECTED_EXPERIMENT, use_val=USE_VAL)
print(f"Validation status: {db_manager.use_val}")

# Load default dataset
with app.app_context():
    out, cap, experiment_metadata, IDTRACKERAI_CONFIG = load_experiment(SELECTED_EXPERIMENT, first_chunk, db_manager)
    offset, CHUNKSIZE, FRAMERATE=experiment_metadata
    frame = None
    contours=[]


@app.route("/", methods=["GET"])
def get():
    return jsonify({"message": "success"})


@app.route("/api/list", methods=["GET"])
def list():
    global SELECTED_EXPERIMENT
    return jsonify({"experiments": SELECTED_EXPERIMENT})

def row2dict(row):
    d = {}
    for column in row.__table__.columns:
        d[column.name] = getattr(row, column.name)
    return d


@app.route('/api/frame/<int:frame_number>', methods=['GET'])
def get_frame(frame_number):

    global cap
    global frame
    global contours

    if frame is None:
        empty_frame=np.ones((1000, 1000), np.uint8)*255
    else:
        empty_frame=np.ones_like(frame, np.uint8)*255


    if cap is None:
        return jsonify({'error': 'Cap could not be loaded'}), 404

    app.logger.debug(f"Fetching frame {frame_number}")
    lock.acquire()
    frame, (frame_number, frame_timestamp) = cap.get_image(frame_number)
    lock.release()
    app.logger.debug(f"Fetching frame {frame_number} done")
    filename=f"{frame_number}.jpg"
    img_path = os.path.join(FRAMES_DIR, filename)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    try:
        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 50])       
        assert os.path.exists(img_path), f"Could not save {img_path}"
        app.logger.debug(f"{cap._basedir} -> {img_path}")
        contours=process_frame(frame, session.get("idtrackerai_config", IDTRACKERAI_CONFIG))
        
    except Exception as error:
        contours=[]
        logger.error(error)
        cv2.imwrite(img_path, empty_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])       


    if frame is None:
        return jsonify({'error': 'Frame not found'}), 404
    else:
        return send_from_directory(os.path.realpath(FRAMES_DIR), filename)



@app.route('/api/preprocess/<int:frame_number>', methods=['GET'])
def get_preprocess(frame_number):

    global contours
    return jsonify({"contours": contours})


def get_pose(db_manager, frame_number):
    identities=get_identities(SELECTED_EXPERIMENT.replace("/", "_"))
    pose={}
    for identity in identities:
        pose[str(identity)]=db_manager.get_pose_for_animal(identity, frame_number)
    return pose

def project_to_absolute(pose, centroids):
    centroids_coords={}
    for centroid in centroids:
        centroids_coords[str(centroid["identity"])]=(centroid["x"], centroid["y"])
    
    pose_abs={}
    for identity in centroids_coords:
        pose_={}
        for bodypart in pose[identity]:
            if any((coord is None for coord in pose[identity][bodypart])):
                pose_[bodypart]=(None, None)
            else:
                pose_[bodypart]=np.round(np.array(pose[identity][bodypart])-50 + centroids_coords[identity]).tolist()
        
        pose_abs[identity]=pose_
    return pose_abs


@app.route('/api/tracking/<int:frame_number>', methods=['GET'])
def get_tracking(frame_number):
    global SELECTED_EXPERIMENT
    number_of_animals=int(re.search(".*/(.*)X/.*", SELECTED_EXPERIMENT).group(1))
    logger.debug("Loading tracking data for %s", SELECTED_EXPERIMENT)
    tables = db_manager.tables

    try:
        output=tables["ROI_0"].query.filter_by(frame_number=frame_number)
        out=[]
        identity_table=tables["IDENTITY"].query.filter_by(frame_number=frame_number)
        number_of_animals_found=0

        for row in output.all():
            identity=None
            hit=False
            for id_row in identity_table:
                if row.in_frame_index == id_row.in_frame_index:
                    identity = id_row.identity
                    local_identity = id_row.local_identity
                    hit=True

            if not hit:
                identity = None

            if row.modified is None:
                modified=0
            else:
                modified=row.modified
            
            data={
                "frame_number": frame_number,
                "t": frame_number/session.get("framerate", FRAMERATE) + offset,
                "x": row.x,
                "y": row.y,
                "in_frame_index": row.in_frame_index,
                "fragment": getattr(row, "fragment", -1),
                "area": row.area,
                "identity": identity,
                "local_identity": local_identity,
                "modified": modified,
            }

            number_of_animals_found+=1
            out.append(data)
        out=sorted(out, key=lambda x: x["identity"])
    except Exception as error:
        app.logger.error(error)
        out= []
    
    if number_of_animals_found==0:
        app.logger.warning("No animals found for frame %s", frame_number)
    else:
        app.logger.info("Number of animals found = %s", number_of_animals_found)


    pose=get_pose(db_manager, frame_number)
    pose_absolute=project_to_absolute(pose, out)
    data={"tracking_data": out, "number_of_animals": number_of_animals, "pose": pose_absolute}
    return jsonify(data)

@app.route('/api/prev_rejection/<int:frame_number>', methods=['GET'])
def get_prev_rejection(frame_number):
    return get_rejection(frame_number, "previous")


@app.route('/api/next_rejection/<int:frame_number>', methods=['GET'])
def get_next_rejection(frame_number):
    return get_rejection(frame_number, "next")


@app.route('/api/prev_error/<int:frame_number>', methods=['GET'])
def get_prev_error(frame_number):
    return get_error(frame_number, "previous")


@app.route('/api/next_error/<int:frame_number>', methods=['GET'])
def get_next_error(frame_number):
    return get_error(frame_number, "next")


@app.route('/api/prev_ok/<int:frame_number>', methods=['GET'])
def get_prev_ok(frame_number):
    return get_ok(frame_number, "previous")


@app.route('/api/next_ok/<int:frame_number>', methods=['GET'])
def get_next_ok(frame_number):
    return get_ok(frame_number, "next")

@app.route('/api/prev_ai/<int:frame_number>', methods=['GET'])
def get_prev_ai(frame_number):
    return get_ai(frame_number, "previous")


@app.route('/api/next_ai/<int:frame_number>', methods=['GET'])
def get_next_ai(frame_number):
    return get_ai(frame_number, "next")


def get_first_non_zero_frame(sql_session: Session, frame_number: int, direction=True):
    global SELECTED_EXPERIMENT
    tables = db_manager.tables

    if direction == "next":
        filter_condition = tables["IDENTITY"].frame_number > frame_number
    elif direction == "previous":
        filter_condition = tables["IDENTITY"].frame_number < frame_number
    else:
        raise Exception(f"direction must be either next or previous. direction={direction}")

    subquery = (
        sql_session.query(
            tables["IDENTITY"].frame_number,
            func.min(tables["IDENTITY"].identity).label("min_identity")
        )
        .filter(filter_condition)
        .group_by(tables["IDENTITY"].frame_number)
        .subquery()
    )

    if direction=="previous":
        result = (
            sql_session.query(subquery)
            .filter(subquery.c.min_identity != 0)
            .order_by(-subquery.c.frame_number)
            .first()
        )
    
    elif direction=="next":
        result = (
            sql_session.query(subquery)
            .filter(subquery.c.min_identity != 0)
            .order_by(subquery.c.frame_number)
            .first()
        )

    return result.frame_number if result else None


@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    message='Shutting down gracefully...'
    logger.debug(message)
    return jsonify({"message": message})

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')

    logger.debug("Gracefully shutting down...")
    db.session.close()  # or however you close your DB connection
    logger.debug("DB connection closed. Bye!")

def get_ok(frame_number, direction):
    frame_number= get_first_non_zero_frame(db.session, frame_number, direction)
    logger.debug("get_ok %s", frame_number)
    return jsonify({"frame_number": frame_number})

def get_rejection(frame_number, direction):

    global SELECTED_EXPERIMENT
    experiment=SELECTED_EXPERIMENT.replace("/", "_")
    csv_file=os.path.join(
        get_basedir(experiment), "interactions", f"{experiment}_rejections.csv"
    )
    rejections=pd.read_csv(csv_file)
    frames_with_rejection=rejections["first_frame"].values
    diff=frames_with_rejection-frame_number
    fn=frame_number

    try:
        if direction=="next":
            frames_with_rejection=frames_with_rejection[diff>0]
            diff=diff[diff>0]
            index=np.argmin(diff)
            fn=frames_with_rejection[index]

        elif direction=="previous":
            frames_with_rejection=frames_with_rejection[diff<0]
            diff=diff[diff<0]
            index=np.argmin(-diff)
            fn=frames_with_rejection[index]
    except KeyError:
        logger.warning("Cannot find %s rejection", direction)
    except FileNotFoundError as error:
        logger.warning(error)


    return jsonify({"frame_number": int(fn)})


def get_error(frame_number, direction):

    global SELECTED_EXPERIMENT
    tables = db_manager.tables

    if direction=="next":
        query=tables["IDENTITY"].query.filter(tables["IDENTITY"].frame_number>frame_number, tables["IDENTITY"].identity==0)
        hit=query.first()
    elif direction=="previous":
        query=tables["IDENTITY"].query.filter(tables["IDENTITY"].frame_number<frame_number, tables["IDENTITY"].identity==0)
        hit=query.order_by(-tables["IDENTITY"].id).first()
    else:
        raise Exception(f"direction must be either next or previous. direction={direction}")

    if hit:
        frame_number=hit.frame_number
    else:
        frame_number=None

    return jsonify({"frame_number": frame_number})


def get_ai(frame_number, direction):
    global SELECTED_EXPERIMENT
    tables = db_manager.tables

    if direction=="next":
        query=tables["AI"].query.filter(tables["AI"].frame_number>frame_number)
        hit=query.order_by(tables["AI"].frame_number).first()
    elif direction=="previous":
        query=tables["AI"].query.filter(tables["AI"].frame_number<frame_number)
        hit=query.order_by(-tables["AI"].frame_number).first()
    else:
        raise Exception(f"direction must be either next or previous. direction={direction}")

    if hit:
        frame_number=hit.frame_number
        ai=hit.ai
    else:
        frame_number=None
        ai=None

    return jsonify({"frame_number": frame_number, "ai": ai})


def get_parser():

    ap=argparse.ArgumentParser()
    ap.add_argument("--port", default=5000, type=int)
    ap.add_argument("--host",default="0.0.0.0")
    return ap

if __name__ == "__main__":

    ap=get_parser()
    args=ap.parse_args()
    app.run(port=args.port, host=args.host, debug=True)  # or set debug=False for production
