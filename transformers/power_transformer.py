from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
import pandas as pd
import numpy as np


class PowerT(BaseEstimator, TransformerMixin):
    def __init__(self, func=PowerTransformer, **kwargs):
        self.func = func(standardize=False, **kwargs)

    def fit(self, X, y=None):
        self.func.fit(X, y=None)
        return self

    def transform(self, X):
        return pd.DataFrame(self.func.transform(X), index=X.index, columns=X.columns)

    def inverse_transform(self, X):
        N = X.shape[1]
        self.func.lambdas_ = [self.func.lambdas_[-N:].mean() for _ in range(N)]
        return pd.DataFrame(self.func.inverse_transform(X), index=X.index, columns=X.columns).round()

class LogT(BaseEstimator, TransformerMixin):
    def __init__(self, method='log1p'):
        #TODO: these likely cover the bases, but if base is useful hyperparam to tune,
        # consider generalizing these funcs to base n (would require vectorizing math.log())
        if method == 'log':
            self.func = np.log
            self.inverse_func = np.exp
        if method == 'log1p':
            self.func = np.log1p
            self.inverse_func = np.expm1
        if method == 'log2':
            self.func = np.log2
            self.inverse_func = lambda x: 2 ** x
        if method == 'log2_1p':
            self.func = lambda x: np.log2(x + 1)
            self.inverse_func = lambda x: 2 ** x - 1
        if method == 'log10':
            self.func = np.log10
            self.inverse_func = lambda x: 10 ** x
        if method == 'log10_1p':
            self.func = lambda x: np.log10(x + 1)
            self.inverse_func = lambda x: 10 ** x - 1

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(self.func(X), index=X.index, columns=X.columns)

    def inverse_transform(self, X):
        return pd.DataFrame(self.inverse_func(X), index=X.index, columns=X.columns).round()