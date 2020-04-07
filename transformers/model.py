from statsmodels.tsa.arima_model import ARIMA
from functools import partial
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor as mor
from multiprocessing import Pool, cpu_count
from fbprophet import Prophet
from tqdm import tqdm

class Naive:
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

class SimpleARIMA:
    def __init__(self, lag_order, degree_of_diff, ma_window, n_forecast=1, model=ARIMA):
        self.lag_order = lag_order
        self.degree_of_diff = degree_of_diff
        self.ma_window = ma_window
        self.model = partial(
            model, order=(self.lag_order, self.degree_of_diff, self.ma_window)
        )
        self.n_forecast = n_forecast

    def fit(self, X, y=None):
        self.models = [self.model(endog=list(X.iloc[i, :])).fit(disp=0) for i in range(X.shape[0])]
        return self

    def predict(self, X):
        yhat = pd.DataFrame(index=X.index)
        for i in tqdm(range(self.n_forecast)):
            yhat.loc[:, f'forecast_{i}'] = [model.forecast()[0][0] for model in self.models]
            self.fit(pd.concat([X, yhat], axis=1))
        return np.maximum(yhat, 0).round()

class SimpleGBM:
    def __init__(self, n_forecast=1, model=XGBRegressor, **params):
        self.model = mor(model(**params), n_jobs=-1)
        self.n_forecast = n_forecast

    def col_map(self, X):
        X.columns = [f"t_{i}" for i in reversed(range(len(X.columns)))]
        return X

    def fit(self, X, y):
        X = self.col_map(X)
        self.cols = X.columns
        self.model.fit(X, y.values)
        return self

    def predict(self, X):
        X = self.col_map(X)
        X = X.loc[:, [col for col in X.columns if col in self.cols]]
        yhat = self.model.predict(X)
        yhat = pd.DataFrame(yhat)
        yhat.index = X.index
        yhat.columns = [f'forecast_{i}' for i in range(self.n_forecast)]
        return np.maximum(yhat, 0).round()

class FBProph:
    def __init__(self, n_forecast=1, model=Prophet):
        self.model = model
        self.n_forecast = n_forecast

    def get_forecast(self, rowdata):
        rowdata = rowdata.reset_index()
        rowdata.columns = ["ds", "y"]
        fit_model = self.model().fit(rowdata)
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
        yhat.columns = [f'forecast_{i}' for i in range(self.n_forecast)]
        return np.maximum(yhat, 0).round()
