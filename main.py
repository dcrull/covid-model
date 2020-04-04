from sklearn.pipeline import Pipeline
from transformers.create_ts import CreateTS
from lib.cv19nyt import nyt_county_api, nyt_state_api, county_shp, state_shp, df_from_api, load_gdfs



STEPS = [
    ("create_ts", CreateTS()),
    ("station_grp", StationGroup()),
]

def make_pipeline(self, geo_col):
    return pipeline

def
