import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from sklearn.pipeline import Pipeline
from config import NYT_COUNTY_URL, NYT_STATE_URL
from utils import mse, rmse, mdpe, mdape
from plotting import plot_ts
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
          'gbm':SimpleGBM(n_estimators=1000, n_jobs=-1),
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

    @staticmethod
    def loss_func(y, yhat, func):
        return func(y, yhat)

    @staticmethod
    def expandingsplit(seq, k):
        q, r = divmod(len(seq), k)
        return (seq[0:(i + 1) * q + min(i + 1, r)] for i in range(k))

    # def get_metrics(self, y, yhat, model_id, funcs):
    #     return pd.DataFrame.from_dict(
    #         {i[0]: self.loss_func(y, yhat, i[1]) for i in funcs},
    #         orient="index",
    #         columns=[f"{model_id}_test"],
    #     )

    def data_prep(self, urlpath):
        data = self.load_nyt(urlpath)
        data = self.prep_pipe.transform(data)
        in_sample, out_sample = self.split_data(data)
        return in_sample, out_sample

    #TODO: streamline as sklearn pipeline
    def transform_fit(self, X, y, model_id):
        X = self.feature_pipe.transform(X)
        return self.models[model_id].fit(X, y)

    def transform_predict(self, X, fitted_model):
        X = self.feature_pipe.transform(X)
        return fitted_model.predict(X)

    def run_inference(self, data, model_id, cols):
        X, y = self.split_data(data.loc[:, cols])
        fold_X, fold_y = self.split_data(X)
        fitted_model = self.transform_fit(fold_X, fold_y, model_id)
        yhat = self.transform_predict(X, fitted_model)
        yhat.columns = y.columns
        return X, y, yhat

    def expanding_window(self, k, data, model_id):
        col_chunks = self.expandingsplit(data.columns, k)
        return {f'{model_id}__fold_{ct}': self.run_inference(data, model_id, cols) for ct, cols in enumerate(col_chunks)}

    def final_test(self, urlpath, model_id):
        data = self.load_nyt(urlpath)
        data = self.prep_pipe.transform(data)
        return self.run_inference(data, model_id, data.columns)

    def plot_folds_ts(self, in_sample, results, idx, target):
        plot_ts(in_sample, idx=idx, c='steelblue',lw=2, label='actual')
        for k, v in results.items():
            model_id, fold_id = k.split('__')
            plot_ts(v[2], idx=idx, c='indianred', lw=3.5, label=fold_id+' forecast')

        title_suffix = 'median across obs
        if isinstance(idx, str): title_suffix = idx
        plt.title(f'actual vs {model_id} predicted {target} by cross-validation fold: {title_suffix}')
        plt.legend()
        plt.show()
        return

def testing():
    cv = CVPredict(n_forecast=3)
    in_sample, out_sample = cv.data_prep(cv.nyt_county_url)
    cvout = cv.expanding_window(5, in_sample, 'gbm')
    return cv, in_sample, out_sample, cvout
    #TODO plots
    #TODO eval
    #TODO enrich