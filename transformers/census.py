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

# for KC this will get census data for greater KC area (i.e. all 4 counties in MO)
FIPS_MAPPER = {('geoid','New York, New York City','00xx0'):['36005','36047','36061','36081','36085'],
               ('geoid', 'Missouri, Kansas City','00xx1'):['29037','29047','29095','29165']}

def agg_sum(df_slice, newfips):
    new_data = df_slice.astype(int).sum()
    new_data['fips'] = newfips
    return new_data

def dissolve_counties(df_slice, newfips):
    df_slice.loc[:, 'fips'] = newfips
    new_data = df_slice.dissolve(by='fips', aggfunc='sum').reset_index()
    new_data['INTPTLAT'] = "+"+f"{df_slice['INTPTLAT'].astype(float).mean():.7f}"
    new_data['INTPTLON'] = f"{df_slice['INTPTLON'].astype(float).mean():.7f}"
    return new_data

def exception_aggregator(target_data, census_data, mapper, agg_func, drop_cols=False):
    for k,v in mapper.items():
        new_row = agg_func(census_data.loc[census_data['fips'].isin(v), :], k[2])
        census_data = census_data.append(new_row, ignore_index=True)
        if not np.any(target_data['fips'] == k[2]):
            target_data.loc[target_data[k[0]] == k[1], 'fips'] = k[2]
        if drop_cols: census_data = census_data.loc[~(census_data['fips'].isin(v)), :]
    return target_data, census_data

class CensusShapes(BaseEstimator, TransformerMixin):
    def __init__(self,
                 county_shp_path='data/tl_2019_us_county/tl_2019_us_county.shp',
                 state_shp_path='data/tl_2018_us_state/tl_2018_us_state.shp',
                 exception_agg_func = dissolve_counties):
        self.county_shp = Path(county_shp_path)
        self.state_shp = Path(state_shp_path)
        self.exception_agg_func = exception_agg_func

    def load_gdfs(self, dpath):
        return gpd.read_file(dpath, geometry='geometry', crs='EPSG:4326').rename(columns={'GEOID': 'fips'})[
            ['fips', 'ALAND', 'INTPTLAT', 'INTPTLON', 'geometry']]

    def handle_exceptions(self, X, census_data):
        select_mapper = [('geoid', 'New York, New York City', '00xx0')]
        return exception_aggregator(X,
                                    census_data,
                                    {k: FIPS_MAPPER[k] for k in select_mapper},
                                    self.exception_agg_func,
                                    drop_cols=True)

    def fit(self, X):
        return self

    def transform(self, X):
        if 'county' in X:
            X, census_data = self.handle_exceptions(X, self.load_gdfs(self.county_shp))
        else:
            census_data = self.load_gdfs(self.state_shp)
        return X.merge(census_data, on='fips', how='left')

class CensusEnrich(BaseEstimator, TransformerMixin):
    def __init__(self,
                 query='?get=POP',
                 year='2018',
                 group='pep/charagegroups',
                 api_url='https://api.census.gov/data',
                 env_api_keyname='CENSUS_API_KEY',
                 exception_agg_func=agg_sum):
        self.query_url = f"{api_url}/{year}/{group}{query}"
        self.env_api_keyname = env_api_keyname
        self.exception_agg_func = exception_agg_func

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
            X, census_data = exception_aggregator(X, census_data, FIPS_MAPPER, self.exception_agg_func)
        return X.merge(census_data, on='fips', how='left')

    def fit(self, X):
        return self

    def transform(self, X):
        queries = self.build_location_queries(X)
        census_data = self.get_data_pool(queries)
        return self.merge_data(X, census_data)
