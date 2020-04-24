from sklearn.base import BaseEstimator, TransformerMixin

class TargetScaler(BaseEstimator, TransformerMixin):
    def __init__(self, divisor, multiplier=100.0):
        self.divisor = divisor
        self.multiplier = multiplier

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X * self.multiplier).div(self.divisor, axis=0)

    def inverse_transform(self, X):
        return X.multiply(self.divisor, axis=0) / self.multiplier
