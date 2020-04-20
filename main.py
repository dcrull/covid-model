import dill
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from sklearn.pipeline import Pipeline
from config import NYT_COUNTY_URL, NYT_STATE_URL
from utils import exp_error, perc_error, abs_perc_error
from plotting import plot_ts, heatmap, plot_forecast, boxplot_error
from transformers.nyt import PrepNYT
from transformers.census import CensusEnrich
from transformers.scaler import TargetScaler
from transformers.differencing import Diff
from transformers.power_transformer import PowerT, LogT
from transformers.ts_models import Naive, FBProph
from transformers.clean import DropNA

# some common pipeline steps
PREP_DICT = {'prepnyt': PrepNYT(),
              'add_pop': CensusEnrich(query='?get=POP', year='2018', group='pep/charagegroups', api_url='https://api.census.gov/data'),
              'per_capita': TargetScaler(target_cols=['cases','deaths'], divisor_col='POP', multiplier=100000.0),
              }

TS_DICT = {'logtrans': LogT(func='log1p'),
                      'first_diff': Diff(),
                      'dropna': DropNA(),
                      'naive': Naive(method=np.mean, kwargs={'axis':1}),
                      'prophet': FBProph(),
                      }

class COVPredict:
    def __init__(self,
                 n_forecast,
                 target,
                 nyt_county_url=NYT_COUNTY_URL,
                 nyt_state_url=NYT_STATE_URL,
                 prep_dict = PREP_DICT,
                 prep_steps= ['prepnyt','add_pop','per_capita'],
                 ts_dict = TS_DICT,
                 ts_steps=['dropna','prophet'],
                 ):
        self.n_forecast = n_forecast
        self.target = target
        self.nyt_state_url = nyt_state_url
        self.nyt_county_url = nyt_county_url
        self.prep_pipe = Pipeline([(k,prep_dict[k]) for k in prep_steps])
        self.ts_pipe = Pipeline([(k,ts_dict[k]) for k in ts_steps])

    @staticmethod
    def expandingsplit(seq, k):
        q, r = divmod(len(seq), k)
        return (seq[0:(i + 1) * q + min(i + 1, r)] for i in range(k))

    @staticmethod
    def load_nyt(url):
        return pd.read_csv(url, parse_dates=['date'])

    def split_data(self, data):
        return data.iloc[:, :-self.n_forecast], data.iloc[:, -self.n_forecast:]

    def load_and_prep(self, urlpath):
        data = self.load_nyt(urlpath)
        return self.prep_pipe.fit_transform(data)

    def create_ts_feature(self, data, target, diff=True):
        data = data.pivot(index='geoid', columns='date', values=target).fillna(0)
        if diff: data = data.diff(axis=1)
        return data

    def run_inference(self, data, final_pipe, cols):
        X, y = self.split_data(data.loc[:, cols])
        yhat = final_pipe.fit(X, y).predict(X)
        yhat.columns = y.columns
        yhat = final_pipe[:-1].inverse_transform(yhat)
        return X, y, yhat

    def cv(self, k, urlpath):
        data = self.load_and_prep(urlpath)
        data = self.create_ts_feature(data, self.target)
        col_chunks = self.expandingsplit(data.columns, k)
        final_pipe = self.ts_pipe
        return {f'fold_{ct}': self.run_inference(data, final_pipe, cols) for ct, cols in enumerate(col_chunks)}

    def cv_plot_ts(self, results, idx):
        for k, v in results.items():
            plot_ts(v[0], idx=idx, c='steelblue', s=5, label='actual')
            plot_ts(v[2], idx=idx, c='indianred', s=20, label=k+' forecast')

        title_suffix = 'mean across obs'
        if isinstance(idx, str): title_suffix = idx
        plt.title(f'actual and predicted {self.target} by cross-validation fold: {title_suffix}')
        plt.legend()
        plt.show()
        return

    def cv_error(self, results, err_func):
        return [err_func(v[1], v[2]).mean().mean() for v in results.values()]

    def cv_plot_error(self, results, err_func):
        err = pd.concat([err_func(v[1], v[2]) for v in results.values()], axis=1)
        boxplot_error(err)
        return

    def final_test(self, urlpath):
        data = self.load_and_prep(urlpath)
        data = self.create_ts_feature(data, self.target)
        self.final_pipe = self.ts_pipe
        self.final_X, self.final_y, self.final_yhat = self.run_inference(data, self.final_pipe, data.columns)

    def final_plots_and_error(self, idx, err_func):
        self.final_err = err_func(self.final_y, self.final_yhat).mean().mean()

        fig = plt.figure()
        plot_ts(pd.concat([self.final_X, self.final_y], axis=1), idx=idx, c='steelblue',lw=2, label='actual')
        label_suffix = 'mean across obs'
        if isinstance(idx, str): label_suffix = idx
        plot_ts(self.final_yhat, idx=idx, c='indianred', lw=3.5, label=f'forecast for {label_suffix}')
        plt.title(f'actual and predicted {self.target}; err: {self.final_err:.4f}')

        fig = plt.figure()
        heatmap(df=pd.concat([self.final_X, self.final_yhat], axis=1), target=self.target, sort_col=self.final_X.columns[-1], forecast_line=self.n_forecast)
        return

    def get_forecast(self, urlpath):
        data = self.load_and_prep(urlpath)
        X = self.create_ts_feature(data, self.target)
        forecast_cols = pd.date_range(start=X.columns[-1] + datetime.timedelta(days=1), periods=self.n_forecast, freq='D')
        pipe = self.ts_pipe
        # empty df to set forecast dims
        y = pd.DataFrame(index=X.index, columns=forecast_cols)
        yhat = pipe.fit(X, y).predict(X)
        yhat = pipe[:-1].inverse_transform(yhat)
        yhat.columns = forecast_cols
        return X, yhat

    def plot_forecast(self, X, yhat, idx):
        plot_ts(X, idx=idx, c='steelblue', label='actual')
        plot_ts(yhat, idx=idx, c='indianred', label='forecast')
        label_suffix = 'mean across obs'
        if isinstance(idx, str): label_suffix = idx
        scale = 'count'
        if 'per_capita' in self.prep_pipe.named_steps.keys(): scale = f"per {self.prep_pipe.get_params()['per_capita__multiplier']:0f} people"
        plt.title(f'actual {self.target} + {self.n_forecast} day forecast for {label_suffix} ({scale})')
        plt.legend()
        plt.show()

    def save_obj(self, opath):
        with open(opath, 'wb') as f:
            dill.dump(self, f)


# TODO: add maps
# TODO: enrich
# TODO: optimize prophet (logistic growth? (need capacities) changepts based on policy?, MCMC)