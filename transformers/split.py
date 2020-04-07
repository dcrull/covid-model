from sklearn.base import BaseEstimator, TransformerMixin

class SplitData(BaseEstimator, TransformerMixin):
    def __init__(self, n_forecast):
        self.n_forecast = n_forecast

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.iloc[:, :-self.n_forecast]
        y = X.iloc[:, -self.n_forecast:]
        return X, y