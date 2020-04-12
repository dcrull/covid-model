from sklearn.base import BaseEstimator, TransformerMixin

class DropNA(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.dropna(how='all', axis=1)

    def inverse_transform(self, X):
        return X
