from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class StationGroup(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["station"] = [i.split("__")[0] for i in X.index]
        return X.groupby("station").sum()
