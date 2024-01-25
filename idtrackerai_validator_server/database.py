
def make_templates(db, key=None, fragments=False):

    if fragments:

        class ROI_0(db.Model):
            __bind_key__ = key
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
            id = db.Column(db.Integer, primary_key=True)
            frame_number = db.Column(db.Integer)
            in_frame_index = db.Column(db.Integer)
            x = db.Column(db.Integer)
            y = db.Column(db.Integer)
            modified = db.Column(db.String(80))
            area = db.Column(db.Integer)


    class METADATA(db.Model):
        __bind_key__ = key
        field = db.Column(db.String(100), primary_key=True)
        value = db.Column(db.String(4000))

    class IDENTITY(db.Model):
        __bind_key__ = key
        id = db.Column(db.Integer, primary_key=True)
        frame_number = db.Column(db.Integer)
        in_frame_index = db.Column(db.Integer)
        local_identity = db.Column(db.Integer)
        identity = db.Column(db.Integer)

    class CONCATENATION(db.Model):
        __bind_key__ = key
        id = db.Column(db.Integer, primary_key=True)
        chunk = db.Column(db.Integer)
        local_identity = db.Column(db.Integer)
        local_identity_after = db.Column(db.Integer)
        is_inferred = db.Column(db.Integer)
        is_broken = db.Column(db.Integer)


    class AI(db.Model):
        __bind_key__ = key
        frame_number = db.Column(db.Integer, primary_key=True)
        ai = db.Column(db.String(30))


    tables = {"ROI_0": ROI_0, "METADATA": METADATA, "IDENTITY": IDENTITY, "CONCATENATION": CONCATENATION, "AI": AI}
    return db, tables

