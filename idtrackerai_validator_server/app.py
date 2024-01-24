import os.path
import time
import logging
logging.basicConfig(level=logging.INFO)
import glob
import shutil

from flask import request
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Session
from sqlalchemy import func

import cv2


from idtrackerai_validator_server.database import make_templates
from idtrackerai_validator_server.backend import (
    load_experiment,
    generate_database_filename,
    list_experiments,
    process_frame,
)
from idtrackerai_validator_server.constants import (
    SELECTED_EXPERIMENT,
    DEFAULT_EXPERIMENT,
    first_chunk,
    FRAMES_DIR,
    database_pattern,
) 

start_time = time.time()

# Clean up previous frames
if os.path.exists(FRAMES_DIR):
    shutil.rmtree(FRAMES_DIR)


# Initialize application
# with wight CORS settings
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "allow_headers": "Content-Type",
        "methods": ["OPTIONS", "GET", "POST"],
    }
})


basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], DEFAULT_EXPERIMENT)
database_file = glob.glob(os.path.join(basedir, database_pattern))[0]
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{database_file}'


# Load database connection to all experiments
TABLES={}
EXPERIMENTS=list_experiments()["experiments"]
SQLALCHEMY_BINDS = {
    experiment: f'sqlite:///{generate_database_filename(experiment)}' for experiment in EXPERIMENTS
}
app.config['SQLALCHEMY_BINDS'] = SQLALCHEMY_BINDS
db = SQLAlchemy(app)
for experiment in EXPERIMENTS:
    db, tables = make_templates(db, experiment)
    TABLES[experiment]=tables

# Load default dataset
with app.app_context():
    out, (cap, caps), (offset, CHUNKSIZE, FRAMERATE), idtrackerai_config = load_experiment(DEFAULT_EXPERIMENT, first_chunk, TABLES)
    frame = None
    contours=None

@app.route("/api/list", methods=["GET"])
def list():
    return jsonify(list_experiments())


@app.route("/api/load", methods=['POST'])
def load():
    """
    Reload the VideoCapture
    """
    global cap
    global offset
    global SELECTED_EXPERIMENT
    global CHUNKSIZE
    global FRAMERATE
    global idtrackerai_config

    experiment_folder = request.json.get('experiment', None)
    
    if cap is not None:
            cap.release()

    out, (cap, caps), (offset, CHUNKSIZE, FRAMERATE), idtrackerai_config = load_experiment(experiment_folder, first_chunk, TABLES)
    assert idtrackerai_config is not None

    if cap is not None:
        SELECTED_EXPERIMENT=experiment_folder
    return jsonify(out)


def row2dict(row):
    d = {}
    for column in row.__table__.columns:
        d[column.name] = getattr(row, column.name)
    return d


@app.route('/api/frame/<int:frame_number>', methods=['GET'])
def get_frame(frame_number):

    global cap
    global frame
    global idtrackerai_config
    global contours


    logging.debug(f"Fetching frame {frame_number}")
    frame, (frame_number, frame_timestamp) = cap.get_image(frame_number)
    filename=f"{frame_number}.jpg"
    img_path = os.path.join(FRAMES_DIR, filename)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    assert os.path.exists(img_path), f"Could not save {img_path}"
    logging.debug(f"{cap._basedir} -> {img_path}")
    contours=process_frame(frame, idtrackerai_config)

    if frame is None:
        return jsonify({'error': 'Frame not found'}), 404
    else:
        return send_from_directory(os.path.realpath(FRAMES_DIR), filename)


@app.route('/api/behavior/<int:identity>/<int:frame_number>', methods=['GET'])
def get_behavior_frame(identity, frame_number):
    raise NotImplementedError()

    global caps
    global frame
    global idtrackerai_config
    global contours

    logging.debug(f"Fetching frame {frame_number}")
    frame, (frame_number, frame_timestamp) = caps[identity].get_image(frame_number)
    filename=f"{identity}_{frame_number}_pose.jpg"
    img_path = os.path.join(FRAMES_DIR, filename)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    assert os.path.exists(img_path), f"Could not save {img_path}"
    contours=process_frame(frame, idtrackerai_config)

    if frame is None:
        return jsonify({'error': 'Frame not found'}), 404
    else:
        return send_from_directory('frames', filename)
    

@app.route('/api/preprocess/<int:frame_number>', methods=['GET'])
def get_preprocess(frame_number):

    global contours
    return jsonify({"contours": contours})

    
@app.route('/api/tracking/<int:frame_number>', methods=['GET'])
def get_tracking(frame_number):

    global FRAMERATE

    try:
        output=TABLES[SELECTED_EXPERIMENT]["ROI_0"].query.filter_by(frame_number=frame_number)
        out=[]
        identity_table=TABLES[SELECTED_EXPERIMENT]["IDENTITY"].query.filter_by(frame_number=frame_number)

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
            
            data={
                "frame_number": frame_number,
                "t": frame_number/FRAMERATE + offset,
                "x": row.x,
                "y": row.y,
                "in_frame_index": row.in_frame_index,
                "fragment": getattr(row, "fragment", -1),
                "area": row.area,
                "identity": identity,
                "local_identity": local_identity,
                "modified": row.modified,
            }

            logging.warning(data)

            out.append(data)
    except Exception as error:
        logging.error(error)
        out= []

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


if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0", debug=True)  # or set debug=False for production
