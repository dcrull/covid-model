from sklearn.base import BaseEstimator, TransformerMixin

class TargetScaler(BaseEstimator, TransformerMixin):
    def __init__(self, target_cols, divisor_col, multiplier=100.0):
        self.target_cols = target_cols
        self.divisor_col = divisor_col
        self.multiplier = multiplier

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[:, self.target_cols] = (X.loc[:, self.target_cols] * self.multiplier).div(X[self.divisor_col].astype(float), axis=0)
        return X

    def inverse_transform(self, X):
        X.loc[:, self.target_cols] = (X.loc[:, self.target_cols].multiply(X[self.divisor_col].astype(float), axis=0) / self.multiplier).round()
        return X
