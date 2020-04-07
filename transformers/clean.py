from sklearn.base import BaseEstimator, TransformerMixin

class DropNA(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y):
        return X.dropna(how='all', axis=1), y
