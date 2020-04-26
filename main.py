import dill
import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
from sklearn.pipeline import Pipeline
from config import NYT_COUNTY_URL, NYT_STATE_URL
from utils import exp_error, perc_error, abs_perc_error
from plotting import plot_ts, heatmap, plot_forecast, boxplot_error
from transformers.nyt import PrepNYT
from transformers.census import CensusEnrich, CensusShapes
from transformers.scaler import TargetScaler
from transformers.differencing import Diff
from transformers.power_transformer import PowerT, LogT
from transformers.ts_models import Naive, FBProph
from transformers.clean import DropNA

MODEL_ID = 'test'

# some common pipeline steps
PREP_DICT = {'prepnyt': PrepNYT(),
             }

SPATIAL_DICT = {'add_geo': CensusShapes(),
               'add_pop': CensusEnrich(query='?get=POP', year='2018', group='pep/charagegroups', api_url='https://api.census.gov/data'),
               }

SCALER_DICT = {'per_1000': ('POP', 1000.0),
               'per_km': ('ALAND', 1000.0)
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
                 nyt_county_url=NYT_COUNTY_URL,
                 nyt_state_url=NYT_STATE_URL,
                 scaler_dict=SCALER_DICT,
                 prep_dict = PREP_DICT,
                 prep_steps= ['prepnyt'],
                 spatial_dict=SPATIAL_DICT,
                 spatial_steps=['add_geo', 'add_pop'],
                 ts_dict = TS_DICT,
                 ts_steps=['dropna','prophet'],
                 ):
        self.n_forecast = n_forecast
        self.nyt_state_url = nyt_state_url
        self.nyt_county_url = nyt_county_url
        self.scaler_dict = scaler_dict
        self.prep_pipe = Pipeline([(k,prep_dict[k]) for k in prep_steps])
        self.spatial_pipe = Pipeline([(k, spatial_dict[k]) for k in spatial_steps])
        self.ts_pipe = Pipeline([(k,ts_dict[k]) for k in ts_steps])

    @staticmethod
    def expandingsplit(seq, k):
        q, r = divmod(len(seq), k)
        return (seq[0:(i + 1) * q + min(i + 1, r)] for i in range(k))

    @staticmethod
    def make_scale_pipe(spatial_data, scaler_dict):
        return Pipeline([(k, TargetScaler(divisor=spatial_data[v[0]].astype(float), multiplier=v[1])) for k,v in scaler_dict.items()])

    def split_data(self, data):
        return data.iloc[:, :-self.n_forecast], data.iloc[:, -self.n_forecast:]

    def load_and_prep(self, urlpath):
        data = pd.read_csv(urlpath, parse_dates=['date'])
        return self.prep_pipe.fit_transform(data)

    def create_spatial_data(self, data):
        data = self.spatial_pipe.fit_transform(data.groupby('geoid').agg('last').drop(['date','cases','deaths'], axis=1).reset_index())
        return data.loc[data['geoid'].notnull(), :]

    def create_ts(self, obs_data, spatial_data, target, get_diff):
        spatial_data.set_index('geoid', inplace=True)
        obs_data = obs_data.pivot(index='geoid', columns='date', values=target).fillna(0)
        if get_diff: obs_data = obs_data.diff(axis=1)
        obs_data = obs_data.reindex(spatial_data.index, fill_value=0.0)
        if self.scaler_dict is not None:
            self.scaler_pipe = self.make_scale_pipe(spatial_data, self.scaler_dict)
            obs_data = self.scaler_pipe.fit_transform(obs_data)
        return obs_data

    def create_features(self, urlpath, target, get_diff):
        obs_data = self.load_and_prep(urlpath)
        spatial_data = self.create_spatial_data(obs_data)
        features = self.create_ts(obs_data, spatial_data, target, get_diff)
        return obs_data, spatial_data, features

    def run_inference(self, data, final_pipe, cols):
        X, y = self.split_data(data.loc[:, cols])
        yhat = final_pipe.fit(X, y).predict(X)
        yhat.columns = y.columns
        yhat = final_pipe[:-1].inverse_transform(yhat)
        return X, y, yhat

    def cv(self, k, urlpath, target, get_diff):
        _, _, X = self.create_features(urlpath, target, get_diff)
        col_chunks = self.expandingsplit(X.columns, k)
        return {f'fold_{ct}': self.run_inference(X, self.ts_pipe, cols) for ct, cols in enumerate(col_chunks)}

    def cv_plot_ts(self, results, idx, target):
        for k, v in results.items():
            plot_ts(v[0], idx=idx, c='steelblue', s=5, label='actual')
            plot_ts(v[2], idx=idx, c='indianred', s=20, label=k+' forecast')

        title_suffix = 'mean across obs'
        if isinstance(idx, str): title_suffix = idx
        plt.title(f'actual and predicted {target} by cross-validation fold: {title_suffix}')
        plt.legend()
        plt.show()
        return

    def cv_error(self, results, err_func):
        return [err_func(v[1], v[2]).mean().mean() for v in results.values()]

    def cv_plot_error(self, results, err_func):
        err = pd.concat([err_func(v[1], v[2]) for v in results.values()], axis=1)
        boxplot_error(err)
        return

    # def final_test(self, urlpath, target):
    #     self.prep_data = self.load_and_prep(urlpath)
    #     self.spatial_data = self.create_spatial_data(self.prep_data)
    #     X = self.create_ts(self.prep_data, target)
    #     self.final_pipe = self.ts_pipe
    #     self.final_X, self.final_y, self.final_yhat = self.run_inference(X, self.final_pipe, X.columns)
    #
    # def final_plots_and_error(self, idx, target, err_func):
    #     self.final_err = err_func(self.final_y, self.final_yhat).mean().mean()
    #
    #     fig = plt.figure()
    #     plot_ts(pd.concat([self.final_X, self.final_y], axis=1), idx=idx, c='steelblue',lw=2, label='actual')
    #     label_suffix = 'mean across obs'
    #     if isinstance(idx, str): label_suffix = idx
    #     plot_ts(self.final_yhat, idx=idx, c='indianred', lw=3.5, label=f'forecast for {label_suffix}')
    #     plt.title(f'actual and predicted {target}; err: {self.final_err:.4f}')
    #
    #     fig = plt.figure()
    #     heatmap(df=pd.concat([self.final_X, self.final_yhat], axis=1), target=target, sort_col=self.final_X.columns[-1], forecast_line=self.n_forecast)
    #     return

    def get_forecast(self, urlpath, target, get_diff=True):
        obs_data, spatial_data, X = self.create_features(urlpath, target, get_diff)
        forecast_cols = pd.date_range(start=X.columns[-1] + datetime.timedelta(days=1), periods=self.n_forecast, freq='D')
        self.final_pipe = self.ts_pipe
        # empty df to set forecast dims
        y = pd.DataFrame(index=X.index, columns=forecast_cols)
        yhat = self.final_pipe.fit(X, y).predict(X)  # TODO: cache ts model?
        yhat = self.final_pipe[:-1].inverse_transform(yhat)
        yhat.columns = forecast_cols
        return obs_data, spatial_data, X, yhat

    def plot_forecast(self, X, yhat, idx, inverse_scale=False):
        if inverse_scale:
            X = self.scaler_pipe.inverse_transform(X)
            yhat = self.scaler_pipe.inverse_transform(yhat)
        plot_ts(X, idx=idx, c='steelblue', label='actual')
        plot_ts(yhat, idx=idx, c='indianred', label='forecast')
        label_suffix = 'mean across obs'
        if isinstance(idx, str): label_suffix = idx
        plt.title(f'actual + {self.n_forecast} day forecast for {label_suffix}')
        plt.legend()
        plt.show()




    def save_obj(self, opath):
        with open(opath, 'wb') as f:
            dill.dump(self, f)


# TODO: add viz
# TODO: optimize prophet (logistic growth? (need capacities) changepts based on policy?, MCMC)
# TODO: data enrich