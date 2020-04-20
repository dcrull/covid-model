from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from fbprophet import Prophet
from tqdm import tqdm

class Naive(BaseEstimator, TransformerMixin):
    def __init__(self, method, kwargs):
        self.method = method
        self.kwargs = kwargs

    def fit(self, X, y):
        self.n_forecast = y.shape[1]
        self.model = self.method(X, **self.kwargs)
        return self

    def predict(self, X):
        yhat = pd.DataFrame(index=X.index)
        for i in tqdm(range(self.n_forecast)):
            yhat.loc[:, f'forecast_{i}'] = self.model
            self.fit(pd.concat([X, yhat], axis=1))
        return np.maximum(yhat, 0)

class FBProph(BaseEstimator, TransformerMixin):
    def __init__(self, model=Prophet, **kwargs):
        self.model = model(**kwargs)

    def get_forecast(self, rowdata, n_forecast):
        rowdata = rowdata.reset_index()
        rowdata.columns = ["ds", "y"]
        fit_model = self.model.fit(rowdata)
        future = fit_model.make_future_dataframe(periods=n_forecast)
        forecast = fit_model.predict(future)
        return pd.Series(forecast.iloc[-n_forecast:, :]['yhat'])

    def fit(self, X, y):
        self.n_forecast = y.shape[1]
        return self

    def predict(self, X):
        p = Pool(cpu_count())
        func = partial(self.get_forecast, n_forecast=self.n_forecast)
        yhat = list(
            tqdm(p.imap(func, [row for i,row in X.iterrows()]), total=X.shape[0]))
        p.close()
        p.join()
        yhat = pd.DataFrame(yhat)
        yhat.index = X.index
        return yhat
