from sklearn.pipeline import Pipeline
from transformers.create_ts import CreateTS
from transformers.model import Naive, SimpleARIMA, SimpleGBM, FBProph
from lib.cv19nyt import df_from_api, nyt_state_api, nyt_county_api

GEO = 'geoid'
TARGET = 'cases'
DATA_API = nyt_county_api
MODEL = Naive
MODEL_ID = 'naive'
GBM = False


STEPS = [('create_ts', CreateTS(GEO, TARGET))]

class CVPredict:
    def __init__(
        self,
        n_forecast,
        data_api=DATA_API,
        target=TARGET,
        steps=STEPS,
        model=MODEL,
        model_id=MODEL_ID,
    ):
        self.n_forecast = n_forecast
        self.data = df_from_api(data_api)
        self.target = target
        self.model = model
        self.model_id = model_id
        self.__train_holdout_split()
        self.pipeline = Pipeline(steps)

    def __train_holdout_split(self):
        self.train_data = self.data.iloc[:, :-self.n_forecast*2]
        self.test_data = self.data

    def cv_splitter(self, k):
        ncols = self.train_data.shape[1] - (self.n_forecast * 2)
        step = ncols // k

        for i in np.arange(step, ncols, step=step):
            train_X = self.train_data.columns[:i]
            train_y = self.train_data.columns[i:i+self.n_forecast]
            valid_X = self.train_data.columns[:i+self.n_forecast]
            valid_y = self.train_data.columns[i+self.n_forecast:i+2*self.n_forecast]
            yield train_X, train_y, valid_X, valid_y

    @staticmethod
    def mse(y, yhat):
        return np.mean((y - yhat) ** 2)

    @staticmethod
    def rmse(y, yhat):
        return np.sqrt(np.mean((y - yhat) ** 2))

    @staticmethod
    def mdpe(y, yhat):
        return np.nanmedian((y - yhat) * 100.0 / y)

    @staticmethod
    def mdape(y, yhat):
        return np.nanmedian(abs(y - yhat) * 100.0 / y)

    @staticmethod
    def loss_func(y, yhat, func):
        return func(y, yhat)

    def get_metrics(self, y, yhat, funcs):
        return pd.DataFrame.from_dict(
            {i[0]: self.loss_func(y, yhat, i[1]) for i in funcs},
            orient="index",
            columns=[f"{self.model_id}_test"],
        )

    def expanding_window_cv(self):
        folds = self.cv_splitter()
        train_kpis = []
        valid_kpis = []
        i = 0
        print("conducting expanding window cross-validation...")
        for train_X, train_y, valid_X, valid_y in tqdm(folds):
            train_y = self.train_data.loc[:, train_y]
            train_X, train_y = self.pipeline.fit_transform(X=self.train_data.loc[:, train_X], y=train_y)
            valid_y = self.train_data.loc[:, valid_y]
            valid_X, valid_y = self.pipeline.transform(X=self.train_data.loc[:, valid_X], y=valid_y)

            self.model.fit(train_X, train_y)
            train_yhat = self.model.predict(train_X)
            valid_yhat = self.model.predict(valid_X)

            train_foldkpis = self.get_metrics(
                train_y,
                train_yhat,
                (
                    ("mse", self.mse),
                    ("rmse", self.rmse),
                    ("mdpe", self.mdpe),
                    ("mdape", self.mdape),
                ),
            )
            valid_foldkpis = self.get_metrics(
                valid_y,
                valid_yhat,
                (
                    ("mse", self.mse),
                    ("rmse", self.rmse),
                    ("mdpe", self.mdpe),
                    ("mdape", self.mdape),
                ),
            )
            train_foldkpis.columns = [f"{col}_{i}" for col in train_foldkpis.columns]
            valid_foldkpis.columns = [f"{col}_{i}" for col in valid_foldkpis.columns]
            train_kpis.append(train_foldkpis)
            valid_kpis.append(valid_foldkpis)
            i += 1

        self.train_kpis = pd.concat(train_kpis, axis=1)
        self.valid_kpis = pd.concat(valid_kpis, axis=1)
        self.train_kpis["fold_mu"] = self.train_kpis.mean(axis=1)
        self.valid_kpis["fold_mu"] = self.valid_kpis.mean(axis=1)

    def final_predict(self):
        print('predicting on test data...')
        train_X = self.train_data.iloc[:, :-self.n_forecast]
        train_y = self.train_data.iloc[:, -self.n_forecast:]
        test_X = self.test_data.iloc[:, :-self.n_forecast]
        test_y = self.test_data.iloc[:, -self.n_forecast:]

        self.train_X, self.train_y = self.pipeline.fit_transform(train_X, train_y)
        self.test_X, self.test_y = self.pipeline.transform(test_X, test_y)

        self.model.fit(self.train_X, self.train_y)
        self.test_yhat = self.model.predict(self.test_X)

        self.test_kpis = self.get_metrics(self.test_y, self.test_yhat, (("mse", self.mse),
                                                                        ("rmse", self.rmse),
                                                                        ("mdpe", self.mdpe),
                                                                        ("mdape", self.mdape)))

    def save_obj(self, opath=f'{MODEL_ID}_artifact.dill'):
        # save the model, pipeline and outputs (but not data)
        obj = {'pipeline':self.pipeline,
               'model_id':self.model_id,
               'model':self.model,
               'y':self.test_y,
               'yhat':self.test_yhat,
               'kpis':self.test_kpis}
        with open(opath, 'wb') as f:
            dill.dump(obj, f)
        print(f'key model and data attributes and output from test saved at {opath}')

if __name__=='__main__':
    obj = TSPredict()
    obj.final_predict()
    obj.save_obj()

