import pandas as pd
from pathlib import Path
import geopandas as gpd
import numpy as np
from utils import api_to_json
from sklearn.base import BaseEstimator, TransformerMixin


class CensusShapes(BaseEstimator, TransformerMixin):
    def __init__(self,
                 county_shp_path='data/tl_2019_us_county/tl_2019_us_county.shp',
                 state_shp_path='data/tl_2018_us_state/tl_2018_us_state.shp'):
        self.county_shp = Path(county_shp_path)
        self.state_shp = Path(state_shp_path)


class CensusErich(BaseEstimator, TransformerMixin):
    def __init__(self,
                 base_url='https://api.census.gov/data',
                 env_api_keyname='CENSUS_API_KEY'):
        self.base_url = base_url
        self.env_api_keyname = env_api_keyname

    def get_census_df(self, query_url):
        df = pd.DataFrame(api_to_json(query_url, api_keyname=self.env_api_keyname))
        df.columns = df.iloc[0].values
        return df[1:].reset_index(drop=True)

    def build_query(self, query='get=POP,GEONAME&for=county:*&in=state:01', year='2018', data_url='pep/charagegroups'):
        return f"{self.base_url}/{year}/{data_url}?{query}&key="




# query census on FIPS - by state
# merge on FIPS
#: merged df with NYT data mapped to census data for each record
# record is a geographie (state, county)



def load_gdfs(dpath):
    return gpd.read_file(dpath, geometry='geometry', crs='EPSG:4326')

def merge_gdf(df, gdf):
    df = convert_fips(df)
    gdf['fips'] = gdf['STATEFP']
    if 'county' in df: gdf['fips'] += gdf['COUNTYFP']
    return df.merge(gdf[['fips','geometry']], on='fips', how='left')

