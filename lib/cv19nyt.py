import pandas as pd
from pathlib import Path
import geopandas as gpd
import numpy as np

nyt_state_api = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
nyt_county_api = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
county_shp = Path('data/tl_2019_us_county/tl_2019_us_county.shp')
state_shp = Path('data/tl_2018_us_state/tl_2018_us_state.shp')

def get_geoid(df):
    # nytimes makes certain exceptions so unique geoid is needed
    # https://github.com/nytimes/covid-19-data#geographic-exceptions
    if 'county' in df:
        df['geoid'] = df['state'] + ', ' + df['county']
    else:
        df['geoid'] = df['state']

    return df

def df_from_api(apipath, **kwargs):
    return get_geoid(pd.read_csv(apipath, parse_dates=['date'], **kwargs))

def load_gdfs(dpath):
    return gpd.read_file(dpath, geometry='geometry', crs='EPSG:4326')

def convert_fips(df):
    df['fips'] = [f'{i:.0f}' for i in df['fips']]
    max_len = max([len(i) for i in df['fips']])
    df['fips'] = [i.zfill(max_len) for i in df['fips']]
    return df

def merge_gdf(df, gdf):
    df = convert_fips(df)
    gdf['fips'] = gdf['STATEFP']
    if 'county' in df: gdf['fips'] += gdf['COUNTYFP']
    return df.merge(gdf[['fips','geometry']], on='fips', how='left')


