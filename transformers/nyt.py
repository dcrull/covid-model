from sklearn.base import BaseEstimator, TransformerMixin
from utils import string_padder
from config import IGNORE_TERRITORIES

class PrepNYT(BaseEstimator, TransformerMixin):
    def __init__(self, ignore_territories=IGNORE_TERRITORIES):
        self.ignore_territories = ignore_territories

    def address_exceptions(self, X):
        # ignore territories
        X = X.loc[~X['state'].isin(self.ignore_territories), :]
        return X

    def get_geoid(self, X):
        # nytimes makes certain exceptions so unique geoid is needed
        # https://github.com/nytimes/covid-19-data#geographic-exceptions
        if 'county' in X:
            X['geoid'] = X['state'] + ', ' + X['county']
        else:
            X['geoid'] = X['state']
        return X

    def convert_fips(self, X):
        X['fips'] = [f'{i:.0f}' for i in X['fips']]
        return string_padder(X, 'fips')

    def fit(self, X):
        return self

    def transform(self, X):
        X = self.address_exceptions(X)
        X = self.get_geoid(X)
        X = self.convert_fips(X)
        return X
