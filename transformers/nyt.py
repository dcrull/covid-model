from sklearn.base import BaseEstimator, TransformerMixin

class PrepNYT(BaseEstimator, TransformerMixin):
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
        max_len = max([len(i) for i in X['fips']])
        X['fips'] = [i.zfill(max_len) for i in X['fips']]
        return X

    def fit(self, X):
        return self

    def transform(self, X):
        X = self.get_geoid(X)
        X = self.convert_fips(X)
        return X
