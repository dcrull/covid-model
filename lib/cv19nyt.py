import pandas as pd
from pathlib import Path

nyt_state_api = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
nyt_county_api = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
county_shp =


def df_from_api(apipath):
    return pd.read_csv(apipath)