import os.path
import time
import logging

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

import cv2
import numpy as np
import pandas as pd

start_time=time.time()

from imgstore.interface import VideoCapture
app = Flask(__name__)
CORS(app) 
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5000",
    # "https://yourdomain.com"
]}})

basedir="/staging/leuven/stg_00115/Data/flyhostel_data/videos/FlyHostel1/5X/2023-05-23_14-00-00/"
database_file=os.path.join(basedir, "database.db")
store_path=os.path.join(basedir, "metadata.yaml")
first_chunk=50
cap = VideoCapture(store_path, first_chunk)  # Replace with your video file
frame_number=45000*100


frame, (frame_number, frame_timestamp) = cap.get_image(frame_number)
last_time=time.time()-start_time
print(last_time, frame_number, frame.shape)

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{database_file}'
db = SQLAlchemy(app)

def row2dict(row):
    d = {}
    for column in row.__table__.columns:
        d[column.name] = getattr(row, column.name)
    return d

class ROI_0(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    frame_number = db.Column(db.Integer)
    in_frame_index = db.Column(db.Integer)
    x = db.Column(db.Integer)
    y = db.Column(db.Integer)
    modified = db.Column(db.String(80))
    area = db.Column(db.Integer)

class METADATA(db.Model):
    field = db.Column(db.String(100), primary_key=True)
    value = db.Column(db.String(4000))

class IDENTITY(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    frame_number = db.Column(db.Integer)
    in_frame_index = db.Column(db.Integer)
    local_identity = db.Column(db.Integer)
    identity = db.Column(db.Integer)

class CONCATENATION(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chunk = db.Column(db.Integer)
    local_identity = db.Column(db.Integer)
    local_identity_after = db.Column(db.Integer)
    is_inferred = db.Column(db.Integer)
    is_broken = db.Column(db.Integer)

@app.route('/api/frame/<int:frame_number>', methods=['GET'])
def get_frame(frame_number):

    global last_time
    print(frame_number)

    logging.debug(f"Fetching frame {frame_number}")

    now = time.time() - start_time
    if ((now - last_time) < .5):
        print(f"Too quick: {now-last_time}")
        return jsonify({"error": "Wait"})

    frame, (frame_number, frame_timestamp) = cap.get_image(frame_number)
    last_time=time.time()-start_time
    filename=f"{frame_number}.jpg"

    cv2.imwrite(os.path.join("frames", filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

    if frame is None:
        return jsonify({'error': 'Frame not found'}), 404
    else:
        return send_from_directory('frames', filename)
    
@app.route('/api/tracking/<int:frame_number>', methods=['GET'])
def get_tracking(frame_number):
    output=ROI_0.query.filter_by(frame_number=frame_number)
    out=[]
    identity_table=IDENTITY.query.filter_by(frame_number=frame_number)

    for row in output.all():
        identity=None
        for id_row in identity_table:
            if row.in_frame_index == id_row.in_frame_index:
                identity = id_row.identity
        
        data={"x": row.x, "y": row.y, "in_frame_index": row.in_frame_index, "identity": identity}
        out.append(data)

    return jsonify(out)

@app.route('/api/next_error/<int:frame_number>', methods=['GET'])
def get_next_error(frame_number):
    return get_error(frame_number, "next")

@app.route('/api/prev_error/<int:frame_number>', methods=['GET'])
def get_prev_error(frame_number):
    return get_error(frame_number, "previous")


def get_error(frame_number, direction):
    if direction=="next":
        query=IDENTITY.query.filter(IDENTITY.frame_number>frame_number, IDENTITY.identity==0)
        hit=query.first()
    elif direction=="previous":
        query=IDENTITY.query.filter(IDENTITY.frame_number<frame_number, IDENTITY.identity==0)
        hit=query.order_by(-IDENTITY.id).first()   
    else:
        raise Exception(f"direction must be either next or previous. direction={direction}")

    if hit:
        frame_number=hit.frame_number
    else:
        frame_number=None

    return jsonify({"frame_number": frame_number})
