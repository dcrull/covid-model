from sklearn.base import BaseEstimator, TransformerMixin


class CreateTS(BaseEstimator, TransformerMixin):
    """
    create ts
    """
    def __init__(self, geo_col, val_col):
        self.geo_col = geo_col
        self.val_col = val_col

    def pivot_data(self, data):
        return data.pivot(index=self.geo_col, columns='date', values=self.val_col).fillna(0)

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        return self.pivot_data(X), self.pivot_data(y)
