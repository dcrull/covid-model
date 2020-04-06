from sklearn.base import BaseEstimator, TransformerMixin
from scipy.ndimage import gaussian_filter1d as gf
from functools import partial
import pandas as pd

class TSRate(BaseEstimator, TransformerMixin):
    '''
    get_dxdy: if True, get derivative (dx / dy), otherwise just difference
    periods: interval in time (dx)
    order: order of diff or dx/dy
    direction: direction of calculation (forward = 1, backward = -1)
    overwrite: if True, overwrite data as identically-shaped df, if False, append new features
    '''
    def __init__(self, get_dxdy=False, periods=1, order=1, direction=1):
        self.get_dxdy = get_dxdy
        self.periods = periods
        self.order = order
        self.direction = direction

    def process(self, X):
        X = X.iloc[:, ::self.direction]
        X = X.diff(periods=self.periods, axis=1)
        if self.get_dxdy: X /= self.periods
        return X.iloc[:, ::self.direction]

    def fit(self, X):
        return self

    def transform(self, X):
        for i in range(self.order):
            X = self.process(X)
        return X

class GF(BaseEstimator, TransformerMixin):
    '''
    smoothing w/ Gf
    '''
    def __init__(self, **kwargs):
        self.filter = partial(gf, **kwargs)

    def fit(self, X):
        return self

    def transform(self, X):
        return pd.DataFrame(self.filter(X), index=X.index, columns=X.columns)