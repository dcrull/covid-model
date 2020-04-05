from sklearn.base import BaseEstimator, TransformerMixin


class CreateTS(BaseEstimator, TransformerMixin):
    """
    create ts
    """
    def __init__(self, geo_col, val_col):
        self.geo_col = geo_col
        self.val_col = val_col

    def fit(self, X):
        return self

    def transform(self, X):
        return X.pivot(index=self.geo_col, columns='date',values=self.val_col).fillna(0)

