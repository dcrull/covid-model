from statsmodels.tsa.arima_model import ARIMA
from functools import partial
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor as mor
from fbprophet import Prophet
from tqdm import tqdm

class Naive:
    def __init__(self, method, kwargs, n_forecast):
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
        return yhat

class SimpleARIMA:
    def __init__(self, n_forecast, lag_order, degree_of_diff, ma_window, model=ARIMA):
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
        return yhat

class SimpleGBM:
    def __init__(self, n_forecast, model=XGBRegressor, **params):
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
        return yhat


class FBProph:
    def __init__(self, n_forecast, model=Prophet):
        self.model = model
        self.n_forecast = n_forecast

    def get_row(self, data, idx):
        rowdata = data.iloc[idx, :].reset_index()
        rowdata.columns = ["ds", "y"]
        return rowdata

    def row_predict(self, model):
        future = model.make_future_dataframe(periods=self.n_forecast)
        forecast = model.predict(future)
        return pd.Series(forecast.iloc[-self.n_forecast:, :]['yhat'])

    def fit(self, X, y=None):
        self.models = [self.model().fit(self.get_row(X, i)) for i in range(X.shape[0])]
        return self

    def predict(self, X):
        yhat = pd.DataFrame(index=X.index, columns=[f'forecast_{i}' for i in self.n_forecast])
        for i, model in enumerate(self.models):
            yhat.iloc[i, :] = self.row_predict(model)
        return yhat