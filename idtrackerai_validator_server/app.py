import os
import shutil
import argparse
import re
import traceback
from threading import Lock
import logging
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask import g
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import cv2
from sqlalchemy import func, create_engine
from sqlalchemy.orm import Session
from flask import session
import h5py
from pathlib import Path


from idtrackerai_validator_server.constants import (
    WITH_FRAGMENTS, first_chunk, FRAMES_DIR, INCLUDE_POSE
)
from idtrackerai_validator_server.database import DatabaseManager
from idtrackerai_validator_server.backend import (
    load_experiment,
    generate_database_filename,
    process_frame,
    list_experiments
)
from idtrackerai_validator_server.utils import load_rejections
from flyhostel.utils import (
    get_identities,
    get_square_width,
    get_square_height,    
)
from flyhostel.utils.pose_export import recreate_pose_file

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)
logging.getLogger("idtrackerai_validator_server.backend").setLevel(logging.DEBUG)
logging.getLogger("imgstore").setLevel(logging.WARNING)
logging.getLogger("watchdog.observers").setLevel(logging.WARNING)

SELECTED_EXPERIMENT_ = os.environ.get("VALIDATOR_EXPERIMENT", None)
if SELECTED_EXPERIMENT_ is not None:
    if "/" not in SELECTED_EXPERIMENT_:
        tokens = SELECTED_EXPERIMENT_.split("_")
        SELECTED_EXPERIMENT = "/".join([tokens[0], tokens[1], "_".join(tokens[2:4])])
    else:
        SELECTED_EXPERIMENT = SELECTED_EXPERIMENT_
else:
    SELECTED_EXPERIMENT = None

USE_VAL = os.environ.get("USE_VAL", None)
if USE_VAL is not None:
    USE_VAL = USE_VAL == "True"

lock = Lock()

# Initialize application with CORS settings
app = Flask(__name__)
app.config['SECRET_KEY'] = 'FLYHOSTEL_1234'
CORS(app)

# Clean up previous frames
if os.path.exists(FRAMES_DIR):
    shutil.rmtree(FRAMES_DIR)

# Use a placeholder URI until an experiment is loaded via POST /api/load
if SELECTED_EXPERIMENT is not None:
    database_file = generate_database_filename(SELECTED_EXPERIMENT)
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{database_file}"
    app.config['SQLALCHEMY_BINDS'] = {SELECTED_EXPERIMENT: f"sqlite:///{database_file}"}
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///:memory:"
    app.config['SQLALCHEMY_BINDS'] = {}

db = SQLAlchemy(app)

cap = None
frame = None
contours = []
offset = None
CHUNKSIZE = None
FRAMERATE = None
IDTRACKERAI_CONFIG = None
db_manager = None

if SELECTED_EXPERIMENT is not None:
    db_manager = DatabaseManager(app, db, with_fragments=WITH_FRAGMENTS, experiment=SELECTED_EXPERIMENT, use_val=USE_VAL)
    print(f"Validation status: {db_manager.use_val}")
    with app.app_context():
        out, cap, experiment_metadata, IDTRACKERAI_CONFIG = load_experiment(SELECTED_EXPERIMENT, first_chunk, db_manager)
        offset, CHUNKSIZE, FRAMERATE = experiment_metadata
# H5 file handle cache
_h5_file_cache = {}
_h5_cache_lock = Lock()

# Bodypart indices to keep (figure this out from step 1)
BODYPARTS_TO_IGNORE = [12, 13, 14, 15, 16, 17]  # ← UPDATE THIS
BODYPARTS_TO_KEEP = [i for i in range(18) if i not in BODYPARTS_TO_IGNORE]

# Map bodypart index to name
BODYPART_NAMES = {
    0: "proboscis",
    1: "thorax",
    2: "abdomen",
    3: "fLL",
    4: "mLL",
    5: "rLL",
    6: "fRL",
    7: "mRL",
    8: "rRL",
    9: "head",
    10: 'lW',
    11: 'rW',
    
}

def get_h5_file(fly_id_str, experiment):
    """Get or open H5 file with caching"""
    cache_key = f"{experiment}__{fly_id_str}"
    
    with _h5_cache_lock:
        if cache_key in _h5_file_cache:
            return _h5_file_cache[cache_key]
        
        folder=f"/flyhostel_data/videos/{experiment}/motionmapper/{fly_id_str}/pose_raw"
        pose_file = f"{folder}/{cache_key}/{cache_key}.h5"
        
        if not Path(pose_file).exists():
            recreate_pose_file(experiment, int(fly_id_str), output=folder)
            logger.debug(f"Pose file not found: {pose_file}")
            return None
        
        try:
            f = h5py.File(pose_file, 'r')
            _h5_file_cache[cache_key] = f
            logger.debug(f"Opened pose file: {pose_file}")
            return f
        except Exception as e:
            logger.warning(f"Failed to open pose file {pose_file}: {e}. Attempting to recreate...")
            try:
                os.remove(pose_file)
                recreate_pose_file(experiment, int(fly_id_str), output=folder)
                f = h5py.File(pose_file, 'r')
                _h5_file_cache[cache_key] = f
                logger.info(f"Successfully recreated and opened pose file: {pose_file}")
                return f
            except Exception as e2:
                logger.error(f"Failed to recreate and open pose file {pose_file}: {e2}")
                return None


def close_h5_files():
    """Close all cached H5 file handles"""
    with _h5_cache_lock:
        for f in _h5_file_cache.values():
            try:
                f.close()
            except:
                pass
        _h5_file_cache.clear()


def get_pose_from_h5(fly_id_str, frame_number, experiment, chunksize):
    """
    Extract pose from H5 file, properly handling chunk offsets.
    
    CRITICAL: The experiment doesn't start at frame 0 of the H5 file.
    It starts at first_chunk * chunksize.
    
    Args:
        fly_id_str: e.g., "FlyHostel1_1X_2026-06-13_17-00-00__01"
        frame_number: frame number in the EXPERIMENT (0-indexed, where 0 is first_chunk*chunksize)
        experiment: experiment path
        chunksize: frames per chunk (from METADATA table)
        first_chunk: first chunk number of this experiment (from constants or METADATA)
    
    Returns:
        Dict {bodypart_name: [x, y]} (coords RELATIVE to centroid) or None
    
    Example:
        If first_chunk=5, chunksize=10000:
        - Experiment frame 0 = Video frame 50000 (chunk 5, position 0)
        - Experiment frame 15000 = Video frame 65000 (chunk 6, position 5000)
    """
    try:
        # Step 6: Open the H5 file and validate bounds
        h5_file = get_h5_file(fly_id_str, experiment)

        if h5_file is None:
            logger.debug(f"Could not open H5 file for {fly_id_str}")
            return None
        
        first_chunk=int(os.path.basename(h5_file["files"][0].decode()).split(".")[0])

        # Step 5: Calculate the index WITHIN this H5 file
        # Each chunk spans [chunk_id * chunksize, (chunk_id + 1) * chunksize)
        # So for a frame in chunk N, the index within that chunk is:
        h5_frame_index = frame_number - (first_chunk * chunksize)

        # tracks shape: (1, 2, 18, total_frames_in_chunk)
        # Validate the index
        if h5_frame_index >= h5_file['tracks'].shape[3]:
            logger.error(
                f"H5 frame index {h5_frame_index} out of bounds "
                f"(max: {h5_file['tracks'].shape[3]}) for {fly_id_str}"
            )
            return None
        
        # Step 7: Extract the coordinates for this frame
        xy_coords = h5_file['tracks'][0, :, :, h5_frame_index]  # shape: (2, 18)
        
        # Step 8: Build and return the pose dict
        pose_dict = {}
        for bp_idx in BODYPARTS_TO_KEEP:
            name = BODYPART_NAMES.get(bp_idx, f"bp_{bp_idx}")
            x = float(xy_coords[0, bp_idx])
            y = float(xy_coords[1, bp_idx])
            pose_dict[name] = [x, y]
        
        return pose_dict
    
    except Exception as e:
        logger.error(f"Error extracting pose for {fly_id_str} frame {frame_number}: {e}")
        logger.error(traceback.print_exc())
        return None
    

def _experiment_required():
    return jsonify({"error": "No experiment loaded. POST to /api/load first."}), 503


@app.route("/", methods=["GET"])
def get():
    return jsonify({"message": "success"})


@app.route("/api/list", methods=["GET"])
def list():
    try:
        return jsonify(list_experiments())
    except Exception as error:
        logger.error("Error listing experiments: %s", error)
        return jsonify({"experiments": [SELECTED_EXPERIMENT] if SELECTED_EXPERIMENT else []})

@app.route("/api/load", methods=["POST"])
def load():
    global SELECTED_EXPERIMENT, cap, frame, contours, db_manager
    global offset, CHUNKSIZE, FRAMERATE, IDTRACKERAI_CONFIG

    data = request.get_json()
    if not data or "experiment" not in data:
        return jsonify({"error": "experiment field required"}), 400

    new_experiment = data["experiment"]
    if "/" not in new_experiment:
        tokens = new_experiment.split("_")
        new_experiment = "/".join([tokens[0], tokens[1], "_".join(tokens[2:4])])

    new_database_file = generate_database_filename(new_experiment)
    if not os.path.exists(new_database_file):
        return jsonify({"error": f"Experiment database not found: {new_database_file}"}), 404

    lock.acquire()
    try:
        SELECTED_EXPERIMENT = new_experiment

        db.session.remove()

        app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{new_database_file}"
        app.config["SQLALCHEMY_BINDS"] = {new_experiment: f"sqlite:///{new_database_file}"}

        try:
            engines_dict = db._app_engines[app]
            for eng in engines_dict.values():
                eng.dispose()
            engines_dict.clear()
            new_engine = create_engine(f"sqlite:///{new_database_file}")
            engines_dict[None] = new_engine
            engines_dict[new_experiment] = new_engine
        except (AttributeError, KeyError):
            pass

        db_manager = DatabaseManager(app, db, with_fragments=WITH_FRAGMENTS, experiment=SELECTED_EXPERIMENT, use_val=USE_VAL)

        out, cap, experiment_metadata, IDTRACKERAI_CONFIG = load_experiment(SELECTED_EXPERIMENT, first_chunk, db_manager)
        if experiment_metadata is None:
            return jsonify({"error": f"Failed to load experiment metadata for {new_experiment}"}), 500

        offset, CHUNKSIZE, FRAMERATE = experiment_metadata
        frame = None
        contours = []
        logger.info("Switched to experiment %s", SELECTED_EXPERIMENT)

    except Exception as error:
        print(traceback.print_exc())

        logger.error("Error loading experiment %s: %s", new_experiment, error)
        return jsonify({"error": str(error)}), 500
    finally:
        lock.release()

    return jsonify({"message": "success", "experiment": SELECTED_EXPERIMENT, "first_frame": first_chunk * CHUNKSIZE})


def row2dict(row):
    d = {}
    for column in row.__table__.columns:
        d[column.name] = getattr(row, column.name)
    return d

@app.route("/api/framerate", methods=['GET'])
def get_framerate():
    if db_manager is None:
        return _experiment_required()
    tables = db_manager.tables
    framerate=tables["METADATA"].query.filter_by(field="framerate").first().value
    return framerate

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

    lock.acquire()
    try:
        assert frame_number is not None
        app.logger.warning(f"frame_number = {frame_number}")

        app.logger.debug(f"Fetching frame {frame_number}")
        frame, (frame_number, frame_timestamp) = cap.get_image(frame_number)
        app.logger.debug(f"Fetching frame {frame_number} done")

    except ValueError or AssertionError as error:
        frame=empty_frame.copy()
        frame_number=first_chunk*CHUNKSIZE
        frame_timestamp=0
        app.logger.error(f"Can't fetch frame {frame_number}")
        app.logger.error(error)

    lock.release()

    # frame=cv2.resize(frame, (1000, 1000))
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
        if identity not in pose:
            continue
        for bodypart in pose[identity]:
            if any((coord is None for coord in pose[identity][bodypart])):
                pose_[bodypart]=(None, None)
            else:
                pose_[bodypart]=np.round(np.array(pose[identity][bodypart])-50 + centroids_coords[identity]).tolist()
        
        pose_abs[identity]=pose_
    return pose_abs


@app.route('/api/tracking/<int:frame_number>', methods=['GET'])
def get_tracking(frame_number):
    if db_manager is None:
        return _experiment_required()
    
    global SELECTED_EXPERIMENT
    number_of_animals = int(re.search(".*/(.*)X/.*", SELECTED_EXPERIMENT).group(1))
    logger.debug("Loading tracking data for %s", SELECTED_EXPERIMENT)
    tables = db_manager.tables
 
    out = []
    number_of_animals_found = 0
    try:
        chunksize = int(float(tables["METADATA"].query.filter_by(field="chunksize").all()[0].value))
 
        output = tables["ROI_0"].query.filter_by(frame_number=frame_number)
        identity_table = tables["IDENTITY"].query.filter_by(frame_number=frame_number)
 
        for row in output.all():
            identity = None
            local_identity = None
            hit = False
            for id_row in identity_table:
                if row.in_frame_index == id_row.in_frame_index:
                    identity = id_row.identity
                    local_identity = id_row.local_identity
                    hit = True
 
            if not hit:
                identity = None
 
            if row.modified is None:
                modified = 0
            else:
                modified = row.modified
            
            t = frame_number / session.get("framerate", FRAMERATE) + offset
            hours = str(int(t // 3600)).zfill(2)
            minutes = str(int((t % 3600) // 60)).zfill(2)
            seconds = str(int(t % 60)).zfill(2)
            zt = f"{hours}:{minutes}:{seconds}"
    
            data = {
                "frame_number": frame_number,
                "t": t,
                "ZT": zt,
                "x": row.x,
                "y": row.y,
                "in_frame_index": row.in_frame_index,
                "fragment": getattr(row, "fragment", -1),
                "area": row.area,
                "identity": identity,
                "local_identity": local_identity,
                "modified": modified,
                "chunksize": chunksize,
            }
 
            number_of_animals_found += 1
            out.append(data)
        
        out = sorted(out, key=lambda x: x["identity"] if x["identity"] is not None else -1)
    except Exception as error:
        app.logger.error(error)
    
    if number_of_animals_found == 0:
        app.logger.warning("No animals found for frame %s", frame_number)
    else:
        app.logger.info("Number of animals found = %s", number_of_animals_found)
 
    # ===== NEW: FETCH POSE DATA FOR EACH ANIMAL FROM H5 FILES =====
    pose_absolute = {}
    
    if INCLUDE_POSE:
        try:
            # Get list of fly identities
            experiment=SELECTED_EXPERIMENT.replace("/", "_")
            square_width=get_square_width(experiment)
            square_height=get_square_height(experiment)


            identities = get_identities(experiment)
            identity_to_fly_id = {i: fly_id for i, fly_id in enumerate(identities)}
           
            for animal in out:
                if animal['identity'] is not None and animal['identity'] in identity_to_fly_id.values():
                    try:
                        fly_id_str=str(animal['identity']).zfill(2)
                        pose_relative = get_pose_from_h5(
                            fly_id_str,
                            frame_number,
                            experiment,
                            chunksize
                        )
                        
                        if pose_relative:
                            # Convert from relative (centroid-relative) to absolute coordinates
                            pose_absolute_animal = {}
                            # Pose is relative to top-left of 200x200 square centered at centroid
                            # Top-left corner is at (centroid_x - 100, centroid_y - 100)
                            square_top_left_x = animal['x'] - square_width//2
                            square_top_left_y = animal['y'] - square_height//2
                            
                            for bodypart_name, (rel_x, rel_y) in pose_relative.items():
                                if rel_x is not None and rel_y is not None:
                                    # The H5 coordinates are relative to centroid
                                    abs_x = round(square_top_left_x  + rel_x, 2)
                                    abs_y = round(square_top_left_y  + rel_y, 2)
                                    pose_absolute_animal[bodypart_name] = [None if np.isnan(abs_x) else abs_x, None if np.isnan(abs_y) else abs_y]
                                else:
                                    pose_absolute_animal[bodypart_name] = [None, None]
                            
                            pose_absolute[str(animal['identity'])] = pose_absolute_animal
                    except Exception as e:
                        logger.error(f"Failed to load pose for identity {animal['identity']}: {e}")
                        logger.error(traceback.print_exc())
        except Exception as e:
            logger.error(f"Error in pose processing: {e}")


     
    data = {
        "tracking_data": out,
        "number_of_animals": number_of_animals,
        "pose": pose_absolute
    }

    response=jsonify(data)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/api/prev_rejection/<int:frame_number>', methods=['GET'])
def get_prev_rejection(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_rejection(frame_number, "previous")


@app.route('/api/next_rejection/<int:frame_number>', methods=['GET'])
def get_next_rejection(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_rejection(frame_number, "next")


@app.route('/api/prev_error/<int:frame_number>', methods=['GET'])
def get_prev_error(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_error(frame_number, "previous")


@app.route('/api/next_error/<int:frame_number>', methods=['GET'])
def get_next_error(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_error(frame_number, "next")


@app.route('/api/prev_ok/<int:frame_number>', methods=['GET'])
def get_prev_ok(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_ok(frame_number, "previous")


@app.route('/api/next_ok/<int:frame_number>', methods=['GET'])
def get_next_ok(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_ok(frame_number, "next")

@app.route('/api/prev_ai/<int:frame_number>', methods=['GET'])
def get_prev_ai(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_ai(frame_number, "previous")


@app.route('/api/next_ai/<int:frame_number>', methods=['GET'])
def get_next_ai(frame_number):
    if db_manager is None:
        return _experiment_required()
    return get_ai(frame_number, "next")



@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    message='Shutting down gracefully...'
    logger.debug(message)
    return jsonify({"message": message})


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    
    logger.debug("Gracefully shutting down...")
    close_h5_files()  # ← ADD THIS
    db.session.close()
    logger.debug("DB connection closed. Bye!")
 


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

def get_ok(frame_number, direction):
    frame_number= get_first_non_zero_frame(db.session, frame_number, direction)
    logger.debug("get_ok %s", frame_number)
    return jsonify({"frame_number": frame_number})


def get_rejection(frame_number, direction):

    global SELECTED_EXPERIMENT
    experiment=SELECTED_EXPERIMENT.replace("/", "_")

    fn=frame_number
    
    try:
        rejections, features=load_rejections(experiment)

        frames_with_rejection=rejections["first_frame"].values
        diff=frames_with_rejection-frame_number

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