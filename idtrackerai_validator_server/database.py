import logging

logger=logging.getLogger(__name__)

from idtrackerai_validator_server.backend import generate_database_filename, list_experiments
from flyhostel.data.human_validation.utils import check_if_validated
class DatabaseManager:
    def __init__(self, app, db, with_fragments=True):
        self.app = app
        self.db = db
        self.database_uris = {}
        self.experiment=None
        self.init_database_uris()
        self.tables={}
        self.with_fragments=with_fragments
        self.dbfile=app.config['SQLALCHEMY_DATABASE_URI'].replace("sqlite:///", "")
        self.use_val=check_if_validated(self.dbfile)
        logger.debug("dbfile %s", self.dbfile)

    def init_database_uris(self):
        experiments = list_experiments()["experiments"]
        for experiment in experiments:
            db_uri = f'sqlite:///{generate_database_filename(experiment)}'
            self.database_uris[experiment] = db_uri

    def get_tables(self, experiment):
        if self.experiment is None or self.experiment != experiment:
            self.switch_database(experiment)
            self.tables=make_templates(self.db, experiment, fragments=self.with_fragments, use_val=self.use_val)
            self.experiment=experiment
        
        elif self.experiment==experiment:
            pass

        return self.tables
            

    def switch_database(self, experiment):
        if experiment in self.database_uris:
            self.app.config['SQLALCHEMY_DATABASE_URI'] = self.database_uris[experiment]
            self.db.engine.dispose()  # Dispose the current engine
            self.db.create_all()      # Reflect new database

    # Additional methods as needed for database operations


def make_templates(db, key=None, fragments=False, use_val=True):

    if use_val:

        if fragments:

            class ROI_0(db.Model):
                __bind_key__ = key
                __tablename__ = 'ROI_0_VAL'
                __table_args__ = {'extend_existing': True}

                id = db.Column(db.Integer, primary_key=True)
                frame_number = db.Column(db.Integer)
                in_frame_index = db.Column(db.Integer)
                x = db.Column(db.Integer)
                y = db.Column(db.Integer)
                modified = db.Column(db.String(80))
                fragment = db.Column(db.String(80))
                area = db.Column(db.Integer)
        else:
            class ROI_0(db.Model):
                __bind_key__ = key
                __tablename__ = 'ROI_0_VAL'
                __table_args__ = {'extend_existing': True}
                # __tablename__ = f'roi_0_{key}'
                id = db.Column(db.Integer, primary_key=True)
                frame_number = db.Column(db.Integer)
                in_frame_index = db.Column(db.Integer)
                x = db.Column(db.Integer)
                y = db.Column(db.Integer)
                modified = db.Column(db.String(80))
                area = db.Column(db.Integer)


    else:

        if fragments:

            class ROI_0(db.Model):
                __bind_key__ = key
                __tablename__ = 'ROI_0'
                __table_args__ = {'extend_existing': True}

                id = db.Column(db.Integer, primary_key=True)
                frame_number = db.Column(db.Integer)
                in_frame_index = db.Column(db.Integer)
                x = db.Column(db.Integer)
                y = db.Column(db.Integer)
                modified = db.Column(db.String(80))
                fragment = db.Column(db.String(80))
                area = db.Column(db.Integer)
        else:
            class ROI_0(db.Model):
                __bind_key__ = key
                __tablename__ = 'ROI_0'
                __table_args__ = {'extend_existing': True}
                # __tablename__ = f'roi_0_{key}'
                id = db.Column(db.Integer, primary_key=True)
                frame_number = db.Column(db.Integer)
                in_frame_index = db.Column(db.Integer)
                x = db.Column(db.Integer)
                y = db.Column(db.Integer)
                modified = db.Column(db.String(80))
                area = db.Column(db.Integer)


    class METADATA(db.Model):
        __bind_key__ = key
        __table_args__ = {'extend_existing': True}
        # __tablename__ = f'metadata_{key}'
        field = db.Column(db.String(100), primary_key=True)
        value = db.Column(db.String(4000))

    if use_val:        
        class IDENTITY(db.Model):

            __bind_key__ = key
            __tablename__ = 'IDENTITY_VAL'
            __table_args__ = {'extend_existing': True}
            id = db.Column(db.Integer, primary_key=True)
            frame_number = db.Column(db.Integer)
            in_frame_index = db.Column(db.Integer)
            local_identity = db.Column(db.Integer)
            identity = db.Column(db.Integer)


    else:
        
        class IDENTITY(db.Model):
            __bind_key__ = key
            __tablename__ = 'IDENTITY'
            __table_args__ = {'extend_existing': True}
            id = db.Column(db.Integer, primary_key=True)
            frame_number = db.Column(db.Integer)
            in_frame_index = db.Column(db.Integer)
            local_identity = db.Column(db.Integer)
            identity = db.Column(db.Integer)

    if use_val:
        class CONCATENATION(db.Model):
            __bind_key__ = key
            __table_args__ = {'extend_existing': True}
            __tablename__ = 'CONCATENATION_VAL'
            id = db.Column(db.Integer, primary_key=True)
            chunk = db.Column(db.Integer)
            local_identity = db.Column(db.Integer)
            local_identity_after = db.Column(db.Integer)
            is_inferred = db.Column(db.Integer)
            is_broken = db.Column(db.Integer)


    else:
        
        class CONCATENATION(db.Model):
            __bind_key__ = key
            __table_args__ = {'extend_existing': True}
            __tablename__ = 'CONCATENATION'
            id = db.Column(db.Integer, primary_key=True)
            chunk = db.Column(db.Integer)
            local_identity = db.Column(db.Integer)
            local_identity_after = db.Column(db.Integer)
            is_inferred = db.Column(db.Integer)
            is_broken = db.Column(db.Integer)


    class AI(db.Model):
        __bind_key__ = key
        __table_args__ = {'extend_existing': True}
        # __tablename__ = f'ai_{key}'
        frame_number = db.Column(db.Integer, primary_key=True)
        ai = db.Column(db.String(30))


    tables = {"ROI_0": ROI_0, "METADATA": METADATA, "IDENTITY": IDENTITY, "CONCATENATION": CONCATENATION, "AI": AI}
    return tables

