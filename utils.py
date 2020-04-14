import os, requests, json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

def census_api_json(query, base_url='https://api.census.gov/data', year='2018', data_url='pep/charagegroups'):
    df = pd.DataFrame(requests.get(f"{base_url}/{year}/{data_url}?{query}&key={os.getenv('API_KEY')}").json())
    df.columns = df.iloc[0].values
    return df[1:].reset_index(drop=True)

def string_padder(df, col):
    # df[col] = [f'{i:.0f}' for i in df[col]] # convert to string/int
    max_len = max([len(i) for i in df[col]])
    df[col] = [i.zfill(max_len) for i in df[col]]
    return df

def exp_error(y, yhat, p=2):
    return pd.DataFrame((np.asarray(y) - np.asarray(yhat)) ** p, columns=y.columns, index=y.index)

def perc_error(y, yhat):
    return pd.DataFrame((np.asarray(y) - np.asarray(yhat)) * 100.0 / np.asarray(y), columns=y.columns, index=y.index)

def abs_perc_error(y, yhat):
    return abs(perc_error(y, yhat))