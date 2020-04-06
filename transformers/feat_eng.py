from sklearn.base import BaseEstimator, TransformerMixin

class PivotData(BaseEstimator, TransformerMixin):
    def __init__(self, target):
        self.target = target

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.pivot(index='geoid', columns='date', values=self.target).fillna(0)


class MakeDiff(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.diff(axis=1).dropna(how='all', axis=1)
