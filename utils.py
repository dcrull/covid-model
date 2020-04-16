import os, requests, json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

def api_to_json(query_url, api_keyname):
    return requests.get(f"{query_url}{os.getenv(api_keyname)}").json()

def string_padder(df, col):
    max_len = max([len(i) for i in df[col]])
    df[col] = [i.zfill(max_len) for i in df[col]]
    return df

def exp_error(y, yhat, p=2):
    return pd.DataFrame((np.asarray(y) - np.asarray(yhat)) ** p, columns=y.columns, index=y.index)

def perc_error(y, yhat):
    return pd.DataFrame((np.asarray(y) - np.asarray(yhat)) * 100.0 / np.asarray(y), columns=y.columns, index=y.index)

def abs_perc_error(y, yhat):
    return abs(perc_error(y, yhat))