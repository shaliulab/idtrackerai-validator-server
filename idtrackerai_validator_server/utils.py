import os.path
import pickle
import pandas as pd
from flyhostel.utils import (
    get_basedir,
)

def load_rejections(experiment):
    """
    This function is also implemented in
    from flyhostel.data.interactions.sociability.behavior_integration.load_rejections
    """
    csv_file=os.path.join(
        get_basedir(experiment), "interactions", f"{experiment}_rejections.csv"
    )
    index_file=os.path.join(
        get_basedir(experiment), "interactions", f"{experiment}_index.csv"
    )
    features_file=os.path.join(
        get_basedir(experiment), "interactions", f"{experiment}_features.hdf5"
    )
    features=pd.read_hdf(features_file)

    rejections=pd.read_csv(csv_file)
    index=pd.read_csv(index_file)
    index=index.loc[index["keep"]]
    rejections=rejections.merge(index[["first_frame", "id", "nn"]].reset_index(), how="left", on=["first_frame", "id", "nn"])
    return rejections, features
