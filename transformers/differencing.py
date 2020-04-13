from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class Diff(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.last_col = X.iloc[:, -1]
        return self

    def transform(self, X):
        return X.diff(axis=1)

    def inverse_transform(self, X):
        return pd.concat([self.last_col, X], axis=1).cumsum(axis=1).iloc[:, 1:]
