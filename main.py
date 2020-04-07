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

MODELS = {'naive':partial(Naive, method=np.mean, kwargs={'axis':1}),
          'arima':partial(SimpleARIMA, lag_order=7, degree_of_diff=0, ma_window=0),
          'gbm':partial(SimpleGBM, n_estimators=500, n_jobs=-1),
          'prophet':partial(FBProph,)}

class CVPredict:
    def __init__(self,
                 n_forecast,
                 nyt_county_url=NYT_COUNTY_URL,
                 nyt_state_url=NYT_STATE_URL,
                 prep_steps=PREP_STEPS,
                 feat_steps=FEAT_STEPS,
                 models=MODELS,
                 ):
        self.n_forecast = n_forecast
        self.nyt_county_url = nyt_county_url
        self.nyt_state_url = nyt_state_url
        self.prep_pipe = Pipeline(prep_steps)
        self.feat_pipe = Pipeline(feat_steps)
        self.models = models

    def load_nyt(self, url):
        return pd.read_csv(url, parse_dates=['date'])

    def get_holdout_data(self, data):
        return data.iloc[:, :-self.n_forecast], data.iloc[:, -self.n_forecast:]

    def run_inference(self, urlpath, model_id):
        data = self.load_nyt(urlpath)
        data = self.prep_pipe.fit_transform(data)
        self.in_sample, self.y = self.get_holdout_data(data)
        self.train = self.feat_pipe.transform(self.in_sample)
        X, y = self.get_holdout_data(self.train)
        X = self.feat_pipe.fit_transform(X)
        fit_model = self.models[model_id](n_forecast=self.n_forecast).fit(X, y)
        self.yhat = fit_model.predict(self.train)


