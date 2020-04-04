from statsmodels.tsa.arima_model import ARIMA
from functools import partial
import pandas as pd
from xgboost import XGBRegressor
from fbprophet import Prophet
from utils import reduce_to_sum


class Naive:
    def __init__(self, method, kwargs, frequency):
        self.method = method
        self.kwargs = kwargs
        self.frequency = frequency

    def fit(self, X, y=None):
        self.model = self.method(X, **self.kwargs)
        return self

    def predict(self, X=None):
        if self.frequency == "D":
            self.model *= 30.0
        yhat = self.model
        return yhat


class SimpleARIMA:
    def __init__(self, lag_order, degree_of_diff, ma_window, model=ARIMA):
        self.lag_order = lag_order
        self.degree_of_diff = degree_of_diff
        self.ma_window = ma_window
        self.model = partial(
            model, order=(self.lag_order, self.degree_of_diff, self.ma_window)
        )

    def fit(self, X, y=None):
        if not hasattr(self, "history"):
            X = reduce_to_sum(X)
            self.history = list(X)
        self.model_fit = self.model(endog=self.history).fit(disp=0)
        return self

    def append_refit(self, obs):
        self.history.append(obs)
        self.fit(X=None)

    def predict(self, X):
        n_obs = len(reduce_to_sum(X))
        yhat = []
        for i in range(n_obs):
            pred = self.model_fit.forecast()[0][0]
            yhat.append(pred)
            self.append_refit(pred)
        return pd.Series(sum(yhat))


class SimpleGBM:
    def __init__(self, model=XGBRegressor, **params):
        self.model = model(**params)

    def col_map(self, X):
        X.columns = [f"t_{i}" for i in reversed(range(len(X.columns)))]
        return X

    def fit(self, X):
        y = X.iloc[:, -30:].sum(axis=1)
        X = X.iloc[:, :-30]

        X = self.col_map(X)
        self.cols = X.columns
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = self.col_map(X)
        X = X.loc[:, [col for col in X.columns if col in self.cols]]
        yhat = self.model.predict(X)
        return yhat


class FBProph:
    def __init__(self, model=Prophet):
        self.model = model

    def fit(self, X):
        X = reduce_to_sum(X).reset_index()
        X.columns = ["ds", "y"]
        self.fit_model = self.model().fit(X)
        return self

    def predict(self, X):
        n_obs = reduce_to_sum(X).shape[0]
        future = self.fit_model.make_future_dataframe(periods=n_obs)
        self.forecast = self.fit_model.predict(future)
        return pd.Series(self.forecast.iloc[-n_obs:, :]["yhat"].sum())
