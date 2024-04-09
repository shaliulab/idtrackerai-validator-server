import logging

logger=logging.getLogger(__name__)

from flyhostel.data.human_validation.utils import check_if_validated


class DatabaseManager:
    def __init__(self, app, db, experiment, with_fragments=True):
        self.app = app
        self.db = db
        self.with_fragments=with_fragments
        self.dbfile=app.config['SQLALCHEMY_DATABASE_URI'].replace("sqlite:///", "")
        self.use_val=check_if_validated(self.dbfile)
        logger.debug("dbfile %s", self.dbfile)
        single_housed="1X" in self.dbfile    
        if not self.use_val=="_VAL" and not single_housed:
            logger.warning("%s not validated", self.dbfile)
        self.tables=make_templates(self.db, experiment, fragments=self.with_fragments, use_val=self.use_val)
        self.experiment=experiment            


def make_templates(db, key=None, fragments=False, use_val="_VAL"):
    
    # Updated abstract model class using db.Model
    class ROI_ABS(db.Model):
        __abstract__ = True
        id = db.Column(db.Integer, primary_key=True)
        frame_number = db.Column(db.Integer)
        in_frame_index = db.Column(db.Integer)
        x = db.Column(db.Integer)
        y = db.Column(db.Integer)
        modified = db.Column(db.String(80))
        fragment = db.Column(db.String(80), nullable=True)  # Make nullable based on the use case
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

    # Function to dynamically create model class
    def get_roi_model(suffix, fragments=False):
        tablename = f'ROI_0{suffix}'  # Adjust table name based on suffix
        class_name = f'ROI_0{suffix}'  # Similarly, adjust class name

        # Conditionally add 'fragment' column based on the fragments flag
        attributes = {'__tablename__': tablename, '__table_args__': {'extend_existing': True}}
        if fragments:
            attributes['fragment'] = db.Column(db.String(80))

        Model = type(class_name, (ROI_ABS,), attributes)
        return Model
    
    # Function to dynamically create model class
    def get_identity_model(suffix):
        tablename = f'IDENTITY{suffix}'  # Adjust table name based on suffix
        class_name = f'IDENTITY{suffix}'  # Similarly, adjust class name

        # Conditionally add 'fragment' column based on the fragments flag
        attributes = {'__tablename__': tablename, '__table_args__': {'extend_existing': True}}


        Model = type(class_name, (IDENTITY_ABS,), attributes)
        return Model
    
    # Function to dynamically create model class
    def get_concatenation_model(suffix):
        tablename = f'CONCATENATION{suffix}'  # Adjust table name based on suffix
        class_name = f'CONCATENATION{suffix}'  # Similarly, adjust class name

        # Conditionally add 'fragment' column based on the fragments flag
        attributes = {'__tablename__': tablename, '__table_args__': {'extend_existing': True}}


        Model = type(class_name, (CONCATENATION_ABS,), attributes)
        return Model


    ROI_0=get_roi_model(use_val, fragments=fragments)
    IDENTITY=get_identity_model(use_val)
    CONCATENATION=get_concatenation_model(use_val)

    class METADATA(db.Model):
        __bind_key__ = key
        __table_args__ = {'extend_existing': True}
        # __tablename__ = f'metadata_{key}'
        field = db.Column(db.String(100), primary_key=True)
        value = db.Column(db.String(4000))

    class AI(db.Model):
        __bind_key__ = key
        __table_args__ = {'extend_existing': True}
        # __tablename__ = f'ai_{key}'
        frame_number = db.Column(db.Integer, primary_key=True)
        ai = db.Column(db.String(30))


    tables = {"ROI_0": ROI_0, "METADATA": METADATA, "IDENTITY": IDENTITY, "CONCATENATION": CONCATENATION, "AI": AI}
    return tables

