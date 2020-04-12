from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
import pandas as pd


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