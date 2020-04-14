import pandas as pd
from pathlib import Path
import geopandas as gpd
import numpy as np

county_shp = Path('data/tl_2019_us_county/tl_2019_us_county.shp')
state_shp = Path('data/tl_2018_us_state/tl_2018_us_state.shp')


def get_census_df(query, year='2018', data_url='pep/charagegroups'):
    df = requests.get
    '?get=POP,GEONAME&for=county:*&in=state:06'


def load_gdfs(dpath):
    return gpd.read_file(dpath, geometry='geometry', crs='EPSG:4326')


def merge_gdf(df, gdf):
    df = convert_fips(df)
    gdf['fips'] = gdf['STATEFP']
    if 'county' in df: gdf['fips'] += gdf['COUNTYFP']
    return df.merge(gdf[['fips','geometry']], on='fips', how='left')

