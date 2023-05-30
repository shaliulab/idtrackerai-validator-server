import os.path
import time
import logging

from flask import request
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

import cv2
import numpy as np
import pandas as pd
import glob

from database import make_templates

start_time = time.time()

from imgstore.interface import VideoCapture

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "*",
            # "https://yourdomain.com"
        ],
        "allow_headers": "Content-Type",
        "methods": ["OPTIONS", "GET", "POST"],
    }
})

first_chunk = 50
cap = None
DEFAULT_EXPERIMENT = "FlyHostel1/5X/2023-05-23_14-00-00"
SELECTED_EXPERIMENT=DEFAULT_EXPERIMENT

def list_experiments():
    experiments = [DEFAULT_EXPERIMENT]
    with open(os.path.join(os.environ["FLYHOSTEL_VIDEOS"], "index.txt"), "r") as filehandle:
        experiments = [experiment.strip() for experiment in filehandle.readlines()]
        experiments = [os.path.sep.join(experiment.split(os.path.sep)[-4:-1]) for experiment in experiments]
    return {"experiments": experiments}

EXPERIMENTS=list_experiments()["experiments"]

def generate_database_filename(experiment):
    return glob.glob(os.path.join(
        os.environ["FLYHOSTEL_VIDEOS"], experiment, database_pattern
    ))[0]

basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], DEFAULT_EXPERIMENT)
# database_pattern="database.db"
database_pattern="FlyHostel*.db"
database_file = glob.glob(os.path.join(basedir, database_pattern))[0]
store_path = os.path.join(basedir, "metadata.yaml")
cap = VideoCapture(store_path, first_chunk)  # Replace with your video file

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{database_file}'
SQLALCHEMY_BINDS = {
    experiment: f'sqlite:///{generate_database_filename(experiment)}' for experiment in EXPERIMENTS
}
app.config['SQLALCHEMY_BINDS'] = SQLALCHEMY_BINDS
db = SQLAlchemy(app)
TABLES={}

for experiment in EXPERIMENTS:
    db, tables = make_templates(db, experiment)
    TABLES[experiment]=tables


@app.route("/api/list", methods=["GET"])
def list():
    return jsonify(list_experiments())


@app.route("/api/load", methods=['POST'])
def load():
    """
    Reload the VideoCapture
    """
    global cap
    global SELECTED_EXPERIMENT

    experiment_folder = request.json.get('experiment', None)
    
    if cap is not None:
        cap.release()

    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], experiment_folder)
    if not os.path.exists(basedir):
        return jsonify({"message": f"{basedir} does not exist"})

    store_path = os.path.join(basedir, "metadata.yaml")
    cap = VideoCapture(store_path, first_chunk)  # Replace with your video file
    
    frame_number = 45000 * 100
    frame, (frame_number, frame_timestamp) = cap.get_image(frame_number)
    last_time = time.time() - start_time
    print(last_time, frame_number, frame.shape)
    SELECTED_EXPERIMENT=experiment_folder
    return jsonify({"message": "success"})

def row2dict(row):
    d = {}
    for column in row.__table__.columns:
        d[column.name] = getattr(row, column.name)
    return d


@app.route('/api/frame/<int:frame_number>', methods=['GET'])
def get_frame(frame_number):

    global last_time
    global cap
    print(frame_number)

    logging.debug(f"Fetching frame {frame_number}")

    now = time.time() - start_time
    # if ((now - last_time) < .5):
    #     print(f"Too quick: {now-last_time}")
    #     return jsonify({"error": "Wait"})

    frame, (frame_number, frame_timestamp) = cap.get_image(frame_number)
    last_time=time.time()-start_time
    filename=f"{frame_number}.jpg"
    print(cap._basedir)

    cv2.imwrite(os.path.join("frames", filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

    if frame is None:
        return jsonify({'error': 'Frame not found'}), 404
    else:
        return send_from_directory('frames', filename)
    
@app.route('/api/tracking/<int:frame_number>', methods=['GET'])
def get_tracking(frame_number):
    output=TABLES[SELECTED_EXPERIMENT]["ROI_0"].query.filter_by(frame_number=frame_number)
    out=[]
    identity_table=TABLES[SELECTED_EXPERIMENT]["IDENTITY"].query.filter_by(frame_number=frame_number)

    for row in output.all():
        identity=None
        hit=False
        for id_row in identity_table:
            if row.in_frame_index == id_row.in_frame_index:
                identity = id_row.identity
                hit=True

        if not hit:
            identity = None
        
        data={"frame_number": frame_number, "x": row.x, "y": row.y, "in_frame_index": row.in_frame_index, "identity": identity, "modified": row.modified}
        out.append(data)

    return jsonify(out)


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


def get_first_non_zero_frame(session: Session, frame_number: int, direction=True):
    if direction == "next":
        filter_condition = TABLES[SELECTED_EXPERIMENT]["IDENTITY"].frame_number > frame_number
    elif direction == "previous":
        filter_condition = TABLES[SELECTED_EXPERIMENT]["IDENTITY"].frame_number < frame_number
    else:
        raise Exception(f"direction must be either next or previous. direction={direction}")

    subquery = (
        session.query(
            TABLES[SELECTED_EXPERIMENT]["IDENTITY"].frame_number,
            func.min(TABLES[SELECTED_EXPERIMENT]["IDENTITY"].identity).label("min_identity")
        )
        .filter(filter_condition)
        .group_by(TABLES[SELECTED_EXPERIMENT]["IDENTITY"].frame_number)
        .subquery()
    )

    if direction=="previous":
        result = (
            session.query(subquery)
            .filter(subquery.c.min_identity != 0)
            .order_by(-subquery.c.frame_number)
            .first()
        )
    
    elif direction=="next":
        result = (
            session.query(subquery)
            .filter(subquery.c.min_identity != 0)
            .order_by(subquery.c.frame_number)
            .first()
        )

    return result.frame_number if result else None




@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    message='Shutting down gracefully...'
    print(message)
    return jsonify({"message": message})

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    # if func is None:
    #     raise RuntimeError('Not running with the Werkzeug Server')
    
    print("Gracefully shutting down...")
    db.session.close()  # or however you close your DB connection
    print("DB connection closed. Bye!")

def get_ok(frame_number, direction):
    frame_number= get_first_non_zero_frame(db.session, frame_number, direction)
    print(frame_number)
    return jsonify({"frame_number": frame_number})


def get_error(frame_number, direction):
    if direction=="next":
        query=TABLES[SELECTED_EXPERIMENT]["IDENTITY"].query.filter(TABLES[SELECTED_EXPERIMENT]["IDENTITY"].frame_number>frame_number, TABLES[SELECTED_EXPERIMENT]["IDENTITY"].identity==0)
        hit=query.first()
    elif direction=="previous":
        query=TABLES[SELECTED_EXPERIMENT]["IDENTITY"].query.filter(TABLES[SELECTED_EXPERIMENT]["IDENTITY"].frame_number<frame_number, TABLES[SELECTED_EXPERIMENT]["IDENTITY"].identity==0)
        hit=query.order_by(-TABLES[SELECTED_EXPERIMENT]["IDENTITY"].id).first()   
    else:
        raise Exception(f"direction must be either next or previous. direction={direction}")

    if hit:
        frame_number=hit.frame_number
    else:
        frame_number=None

    return jsonify({"frame_number": frame_number})


def get_ai(frame_number, direction):
    if direction=="next":
        query=TABLES[SELECTED_EXPERIMENT]["AI"].query.filter(TABLES[SELECTED_EXPERIMENT]["AI"].frame_number>frame_number)
        hit=query.order_by(TABLES[SELECTED_EXPERIMENT]["AI"].frame_number).first()   
    elif direction=="previous":
        query=TABLES[SELECTED_EXPERIMENT]["AI"].query.filter(TABLES[SELECTED_EXPERIMENT]["AI"].frame_number<frame_number)
        hit=query.order_by(-TABLES[SELECTED_EXPERIMENT]["AI"].frame_number).first()   
    else:
        raise Exception(f"direction must be either next or previous. direction={direction}")

    if hit:
        frame_number=hit.frame_number
        ai=hit.ai
    else:
        frame_number=None
        ai=None

    return jsonify({"frame_number": frame_number, "ai": ai})
