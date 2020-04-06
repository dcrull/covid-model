from sklearn.base import BaseEstimator, TransformerMixin

class TSDiff(BaseEstimator, TransformerMixin):
    def __init__(self):

    def fit(self, X):
        return self

    def transform(self, X):
        return X.pivot(index='geoid', columns='date', values=self.response_var).fillna(0)
