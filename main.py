import pandas as pd
import numpy as np
from functools import partial
from sklearn.pipeline import Pipeline
from config import NYT_COUNTY_URL, NYT_STATE_URL
from plotting import plot_mean_ts
from transformers.nyt import PrepNYT
from transformers.create_ts import CreateTS
from transformers.transform_ts import TSRate, GF
from transformers.model import Naive, SimpleARIMA, SimpleGBM, FBProph
from transformers.clean import DropNA


PREP_STEPS = [
    ('prepnyt', PrepNYT()),
    ('create_ts', CreateTS(response_var='cases')),
]

FEAT_STEPS = [
    ('first_diff', TSRate(get_dxdy=False, periods=7, order=1)),
    # ('gf_s1.5', GF(sigma=0.5)),
    ('dropna', DropNA())
]
#TODO: other transformations (log, etc)

MODELS = {'naive':Naive(method=np.mean, kwargs={'axis':1}),
          'arima':SimpleARIMA(lag_order=7, degree_of_diff=0, ma_window=0),
          'gbm':SimpleGBM(n_estimators=500, n_jobs=-1),
          'prophet':FBProph()}

class CVPredict:
    def __init__(self,
                 n_forecast,
                 nyt_county_url=NYT_COUNTY_URL,
                 nyt_state_url=NYT_STATE_URL,
                 prep_steps=PREP_STEPS,
                 feat_steps=FEAT_STEPS,
                 models=MODELS,
                 ):
        self.nyt_county_url = nyt_county_url
        self.nyt_state_url = nyt_state_url
        self.prep_pipe = Pipeline(prep_steps)
        self.feature_pipe = Pipeline(feat_steps)
        self.models = models
        self.__set_forecast__(n_forecast)

    def __set_forecast__(self, n_forecast):
        self.n_forecast = n_forecast
        for k,v in self.models.items():
            v.n_forecast = n_forecast
            self.models[k] = v

    def load_nyt(self, url):
        return pd.read_csv(url, parse_dates=['date'])

    def split_data(self, data):
        return data.iloc[:, :-self.n_forecast], data.iloc[:, -self.n_forecast:]

    def data_prep(self, urlpath):
        data = self.load_nyt(urlpath)
        data = self.prep_pipe.transform(data)
        in_sample, out_sample = self.split_data(data)
        return in_sample, out_sample

    #TODO: add model as step in pipeline
    def fit_model(self, data, model_id):
        X, y = self.split_data(data)
        X = self.feature_pipe.transform(X)
        return self.models[model_id].fit(X, y)

    def final_predict(self, urlpath, model_id):
        in_sample, self.test_y = self.data_prep(urlpath)
        fit_model = self.fit_model(in_sample, model_id)
        self.test_X = self.feature_pipe.transform(in_sample)
        self.test_yhat = fit_model.predict(self.test_X)


        # X = self.feat_pipe.fit_transform(X)
        # self.fit_model = self.models[model_id].fit(X, y)

