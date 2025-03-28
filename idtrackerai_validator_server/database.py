import logging
import numpy as np
from flyhostel.data.human_validation.utils import check_if_validated
from flyhostel.utils import (
    get_pose_file,
    get_identities
)
from flyhostel.data.pose.loaders.movement import from_sleap_file
POSE_NAME="raw"
logger = logging.getLogger(__name__)


def pose_arr_to_dict(ds):
    bodyparts=ds.keypoints.values.tolist()
    pose={}

    coordinates=list(ds.coords.keys())
    time_i=coordinates.index("time")
    individuals_i=coordinates.index("individuals")
    keypoints_i=coordinates.index("keypoints")
    space_i=coordinates.index("space")


    # check data for a single frame is provided
    assert ds.position.shape[time_i]==1

    # check data for a single animal is provided
    assert ds.position.shape[individuals_i]==1

    for i, bodypart in enumerate(bodyparts):
        # first two axis have length 1 because it captures frames and animals
        coords=np.round(np.squeeze(ds.sel(keypoints=bodypart).position.values)).tolist()
        coords=[None if np.isnan(e) else e for e in coords]
        pose[bodypart]=coords
    return pose



class DatabaseManager:
    def __init__(self, app, db, experiment, with_fragments=True, use_val=None):
        self.app = app
        self.db = db
        self.with_fragments = with_fragments
        self.dbfile = app.config['SQLALCHEMY_DATABASE_URI'].replace("sqlite:///", "")
        print(f"Opening {self.dbfile}")
        if use_val is None:
            self.use_val = check_if_validated(self.dbfile)
        else:
            self.use_val = "_VAL" if use_val else ""
        
        logger.debug("dbfile %s", self.dbfile)
        single_housed = "1X" in self.dbfile    
        if not self.use_val == "_VAL" and not single_housed:
            logger.warning("%s not validated", self.dbfile)
        
        self.tables = make_templates(self.db, experiment, fragments=self.with_fragments, use_val=self.use_val)
        self.experiment = experiment

        # Initialize a dictionary to hold the pose data from HDF5 files.
        self.pose_data = {}

        identities=get_identities(experiment.replace("/", "_"))
        pose_files={
            identity: get_pose_file(experiment.replace("/", "_"), identity, pose_name=POSE_NAME)
            for identity in identities
        }

        if pose_files:
            # If pose_files is a dictionary mapping animal_id to file path:
            if isinstance(pose_files, dict):
                for identity, file_path in pose_files.items():
                    try:
                        self.pose_data[identity] = from_sleap_file(file_path)
                        logger.info("Loaded pose data for animal %s from %s", identity, file_path)
                    except Exception as e:
                        logger.error("Error loading HDF5 file for animal %s: %s", identity, e)


    def get_pose_for_animal(self, animal_id, frame_number):
        """
        Retrieve the HDF5 file object for the given animal_id.
        You can add further methods to read specific datasets (e.g., poses for a specific frame).
        """
        if animal_id in self.pose_data:
            ds=self.pose_data[animal_id]
            mask = (ds.frame_number >= frame_number) & (ds.frame_number < frame_number+1)
            pose=ds.where(mask, drop=True)
            pose=pose_arr_to_dict(pose)
            return pose
        else:
            logger.warning("Pose data for animal %s not found. Available %s", animal_id, list(self.pose_data.keys()))
            return {}

    def close_pose_files(self):
        """Close all open HDF5 file handles."""
        for animal_id, h5file in self.pose_data.items():
            try:
                h5file.close()
                logger.info("Closed pose file for animal %s", animal_id)
            except Exception as e:
                logger.error("Error closing HDF5 file for animal %s: %s", animal_id, e)

# Existing function remains unchanged.
def make_templates(db, key=None, fragments=False, use_val="_VAL"):
    class ROI_ABS(db.Model):
        __abstract__ = True
        id = db.Column(db.Integer, primary_key=True)
        frame_number = db.Column(db.Integer)
        in_frame_index = db.Column(db.Integer)
        x = db.Column(db.Integer)
        y = db.Column(db.Integer)
        modified = db.Column(db.String(80))
        fragment = db.Column(db.String(80), nullable=True)
        area = db.Column(db.Integer)

    class IDENTITY_ABS(db.Model):
        __abstract__ = True
        id = db.Column(db.Integer, primary_key=True)
        frame_number = db.Column(db.Integer)
        in_frame_index = db.Column(db.Integer)
        local_identity = db.Column(db.Integer)
        identity = db.Column(db.Integer)

    class CONCATENATION_ABS(db.Model):
        __abstract__ = True
        id = db.Column(db.Integer, primary_key=True)
        chunk = db.Column(db.Integer)
        local_identity = db.Column(db.Integer)
        local_identity_after = db.Column(db.Integer)
        is_inferred = db.Column(db.Integer)
        is_broken = db.Column(db.Integer)

    def get_roi_model(suffix, fragments=False):
        tablename = f'ROI_0{suffix}'
        class_name = f'ROI_0{suffix}'
        attributes = {'__tablename__': tablename, '__table_args__': {'extend_existing': True}}
        if fragments:
            attributes['fragment'] = db.Column(db.String(80))
        return type(class_name, (ROI_ABS,), attributes)
    
    def get_identity_model(suffix):
        tablename = f'IDENTITY{suffix}'
        class_name = f'IDENTITY{suffix}'
        attributes = {'__tablename__': tablename, '__table_args__': {'extend_existing': True}}
        return type(class_name, (IDENTITY_ABS,), attributes)
    
    def get_concatenation_model(suffix):
        tablename = f'CONCATENATION{suffix}'
        class_name = f'CONCATENATION{suffix}'
        attributes = {'__tablename__': tablename, '__table_args__': {'extend_existing': True}}
        return type(class_name, (CONCATENATION_ABS,), attributes)

    ROI_0 = get_roi_model(use_val, fragments=fragments)
    IDENTITY = get_identity_model(use_val)
    CONCATENATION = get_concatenation_model(use_val)

    class METADATA(db.Model):
        __bind_key__ = key
        __table_args__ = {'extend_existing': True}
        field = db.Column(db.String(100), primary_key=True)
        value = db.Column(db.String(4000))

    class AI(db.Model):
        __bind_key__ = key
        __table_args__ = {'extend_existing': True}
        frame_number = db.Column(db.Integer, primary_key=True)
        ai = db.Column(db.String(30))

    tables = {"ROI_0": ROI_0, "METADATA": METADATA, "IDENTITY": IDENTITY, "CONCATENATION": CONCATENATION, "AI": AI}
    return tables
