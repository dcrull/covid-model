from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from fbprophet import Prophet
from tqdm import tqdm

class Naive(BaseEstimator, TransformerMixin):
    def __init__(self, method, kwargs, n_forecast=1):
        self.method = method
        self.kwargs = kwargs
        self.n_forecast = n_forecast

    def fit(self, X, y=None):
        self.model = self.method(X, **self.kwargs)
        return self

    def predict(self, X):
        yhat = pd.DataFrame(index=X.index)
        for i in tqdm(range(self.n_forecast)):
            yhat.loc[:, f'forecast_{i}'] = self.model
            self.fit(pd.concat([X, yhat], axis=1))
        return np.maximum(yhat, 0).round()

class FBProph(BaseEstimator, TransformerMixin):
    def __init__(self, n_forecast=1, model=Prophet, **kwargs):
        self.n_forecast = n_forecast
        self.model = model(**kwargs)

    def get_forecast(self, rowdata):
        rowdata = rowdata.reset_index()
        rowdata.columns = ["ds", "y"]
        fit_model = self.model.fit(rowdata)
        future = fit_model.make_future_dataframe(periods=self.n_forecast)
        forecast = fit_model.predict(future)
        return pd.Series(forecast.iloc[-self.n_forecast:, :]['yhat'])

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        p = Pool(cpu_count())
        yhat = list(
            tqdm(p.imap(self.get_forecast, [row for i,row in X.iterrows()]), total=X.shape[0]))
        p.close()
        p.join()
        yhat = pd.DataFrame(yhat)
        yhat.index = X.index
        return yhat.round()
