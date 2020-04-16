from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor as mor
from tqdm import tqdm

class SimpleGBM(BaseEstimator, TransformerMixin):
    def __init__(self, n_forecast=1, model=XGBRegressor, **params):
        self.model = mor(model(**params), n_jobs=-1)
        self.n_forecast = n_forecast

    def col_map(self, X):
        X.columns = [f"t_{i}" for i in reversed(range(len(X.columns)))]
        return X

    def fit(self, X, y=None):
        X, y = X.iloc[:, :-self.n_forecast], X.iloc[:, -self.n_forecast:]
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
        return yhat.round()
