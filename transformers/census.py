import pandas as pd
from pathlib import Path
import geopandas as gpd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from utils import api_to_json
from sklearn.base import BaseEstimator, TransformerMixin
from utils import string_padder

#TODO: deal with NYTimes geo exceptions
# https://github.com/nytimes/covid-19-data#geographic-exceptions

class CensusShapes(BaseEstimator, TransformerMixin):
    def __init__(self,
                 county_shp_path='data/tl_2019_us_county/tl_2019_us_county.shp',
                 state_shp_path='data/tl_2018_us_state/tl_2018_us_state.shp'):
        self.county_shp = Path(county_shp_path)
        self.state_shp = Path(state_shp_path)

    # def load_gdfs(self, dpath):
    #     return gpd.read_file(dpath, geometry='geometry', crs='EPSG:4326')
    #
    # def merge_gdf(self, df, gdf):
    #     df = convert_fips(df)
    #     gdf['fips'] = gdf['STATEFP']
    #     if 'county' in df: gdf['fips'] += gdf['COUNTYFP']
    #     return df.merge(gdf[['fips', 'geometry']], on='fips', how='left')

class CensusEnrich(BaseEstimator, TransformerMixin):
    def __init__(self,
                 query='?get=POP',
                 year='2018',
                 group='pep/charagegroups',
                 api_url='https://api.census.gov/data',
                 env_api_keyname='CENSUS_API_KEY'):
        self.query_url = f"{api_url}/{year}/{group}{query}"
        self.env_api_keyname = env_api_keyname

    def get_census_df(self, query_url):
        try:
            df = pd.DataFrame(api_to_json(query_url+'&key=', api_keyname=self.env_api_keyname))
            df.columns = df.iloc[0].values
            return df[1:].reset_index(drop=True)
        except:
            print(query_url)
            pass

    def build_location_queries(self, X):
        # chunks to state for county queries in order to parallelize api calls
        # assumes fips col is either a 2-char string for state or 5-char 0-padded str for state + county
        if 'county' in X:
            return [self.query_url+f'&for=county:*&in=state:{st}' for st in X['fips'].str.slice(0,2).unique() if st != '00']
        else:
            return [self.query_url+'&for=state:*']

    def get_data_pool(self, queries):
        p = Pool(cpu_count())
        census_data = list(tqdm(p.imap(self.get_census_df, queries), total=len(queries)))
        p.close()
        p.join()
        return pd.concat(census_data)

    def merge_data(self, X, census_data):
        census_data = string_padder(census_data, 'state').rename(columns={'state':'fips'})
        if 'county' in census_data:
            census_data = string_padder(census_data, 'county')
            census_data['fips'] = census_data['fips'] + census_data['county']
            del census_data['county']
        return X.merge(census_data, on='fips', how='left')

    def fit(self, X):
        return self

    def transform(self, X):
        queries = self.build_location_queries(X)
        census_data = self.get_data_pool(queries)
        return self.merge_data(X, census_data)
