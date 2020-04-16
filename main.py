import dill
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from sklearn.pipeline import Pipeline
from config import NYT_COUNTY_URL, NYT_STATE_URL
from utils import exp_error, perc_error, abs_perc_error
from plotting import plot_ts, heatmap, plot_forecast
from transformers.nyt import PrepNYT
from transformers.create_ts import CreateTS
from transformers.differencing import Diff
from transformers.power_transformer import PowerT, LogT
from transformers.ts_models import Naive, FBProph
from transformers.clean import DropNA

PREP_STEPS = [
    ('prepnyt', PrepNYT()),
    ('create_ts', CreateTS(response_var='cases')),
    ('first_diff', Diff())
]

FEATURE_STEPS = [
    ('logtrans', LogT(func='log1p')),
    ('first_diff', Diff()),
    ('dropna', DropNA()),
]

MODELS = {'naive':Naive(method=np.mean, kwargs={'axis':1}),
          'prophet': FBProph(),
          }

class COVPredict:
    def __init__(self,
                 n_forecast,
                 nyt_county_url=NYT_COUNTY_URL,
                 nyt_state_url=NYT_STATE_URL,
                 prep_steps=PREP_STEPS,
                 feat_steps=FEATURE_STEPS,
                 models=MODELS,
                 ):
        self.nyt_county_url = nyt_county_url
        self.nyt_state_url = nyt_state_url
        self.prep_pipe = Pipeline(prep_steps)
        self.feature_steps = feat_steps
        self.models = models
        self.__set_forecast__(n_forecast)

    def __set_forecast__(self, n_forecast):
        self.n_forecast = n_forecast
        for k,v in self.models.items():
            v.n_forecast = n_forecast
            self.models[k] = v

    @staticmethod
    def expandingsplit(seq, k):
        q, r = divmod(len(seq), k)
        return (seq[0:(i + 1) * q + min(i + 1, r)] for i in range(k))

    @staticmethod
    def load_nyt(url):
        return pd.read_csv(url, parse_dates=['date'])

    def split_data(self, data):
        return data.iloc[:, :-self.n_forecast], data.iloc[:, -self.n_forecast:]

    def data_prep(self, urlpath):
        data = self.load_nyt(urlpath)
        data = self.prep_pipe.transform(data)
        in_sample, out_sample = self.split_data(data)
        return in_sample, out_sample

    def build_final_pipe(self, model_id):
        pipe = Pipeline(self.feature_steps + [(model_id, self.models[model_id])])
        return pipe

    def run_inference(self, data, final_pipe, cols):
        X, y = self.split_data(data.loc[:, cols])
        yhat = final_pipe.fit(X).predict(X)
        yhat.columns = y.columns
        yhat = final_pipe[:-1].inverse_transform(yhat)
        return X, y, yhat

    def expanding_window(self, k, data, model_id):
        col_chunks = self.expandingsplit(data.columns, k)
        final_pipe = self.build_final_pipe(model_id)
        return {f'{model_id}__fold_{ct}': self.run_inference(data, final_pipe, cols) for ct, cols in enumerate(col_chunks)}

    def final_test(self, urlpath, model_id):
        data = self.load_nyt(urlpath)
        data = self.prep_pipe.transform(data)
        self.final_pipe = self.build_final_pipe(model_id)
        self.final_X, self.final_y, self.final_yhat = self.run_inference(data, self.final_pipe, data.columns)

    def plot_folds_ts(self, in_sample, results, idx, target):
        plot_ts(in_sample, idx=idx, c='steelblue',lw=2, label='actual')
        for k, v in results.items():
            model_id, fold_id = k.split('__')
            plot_ts(v[2], idx=idx, c='indianred', lw=3.5, label=fold_id+' forecast')

        title_suffix = 'mean across obs'
        if isinstance(idx, str): title_suffix = idx
        plt.title(f'actual and {model_id} predicted {target} by cross-validation fold: {title_suffix}')
        plt.legend()
        plt.show()
        return

    def fold_error(self, results, err_func):
        return [err_func(v[1], v[2]).mean().mean() for v in results.values()]

    def final_plots_and_error(self, idx, target, err_func):
        self.final_err = err_func(self.final_y, self.final_yhat).mean().mean()

        fig = plt.figure()
        plot_ts(pd.concat([self.final_X, self.final_y], axis=1), idx=idx, c='steelblue',lw=2, label='actual')
        label_suffix = 'mean across obs'
        if isinstance(idx, str): label_suffix = idx
        plot_ts(self.final_yhat, idx=idx, c='indianred', lw=3.5, label=f'forecast for {label_suffix}')
        plt.title(f'actual and predicted {target}; err: {self.final_err:.4f}')

        fig = plt.figure()
        heatmap(df=pd.concat([self.final_X, self.final_yhat], axis=1), target=target, sort_col=self.final_X.columns[-1], forecast_line=self.n_forecast)
        return

    def get_forecast(self, urlpath, model_id):
        data = self.load_nyt(urlpath)
        X = self.prep_pipe.transform(data)
        pipe = self.build_final_pipe(model_id)
        yhat = pipe.fit(X).predict(X)
        yhat = pipe[:-1].inverse_transform(yhat)
        yhat.columns = pd.date_range(start=X.columns[-1] + datetime.timedelta(days=1), periods=self.n_forecast, freq='D')
        return X, yhat

    def save_obj(self, opath):
        with open(opath, 'wb') as f:
            dill.dump(self, f)


# TODO: add maps
# TODO: enrich