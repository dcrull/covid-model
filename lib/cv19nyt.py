import pandas as pd
from pathlib import Path
import geopandas as gpd

nyt_state_api = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
nyt_county_api = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
county_shp = Path('data/tl_2019_us_county/tl_2019_us_county.shp')
state_shp = Path('data/tl_2018_us_state/tl_2018_us_state.shp')

def df_from_api(apipath):
    return pd.read_csv(apipath, parse_dates=['date'])

def load_gdfs(dpath):
    return gpd.read_file(dpath, geometry='geometry', crs='EPGS:4426')