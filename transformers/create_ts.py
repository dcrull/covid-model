from sklearn.base import BaseEstimator, TransformerMixin

class CreateTS(BaseEstimator, TransformerMixin):
    def __init__(self, response_var):
        self.response_var = response_var

    def fit(self, X):
        return self

    def transform(self, X):
        return X.pivot(index='geoid', columns='date', values=self.response_var).fillna(0)

    def inverse_transform(self, X):
        return X
