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
        data_api=DATA_API,
        target=TARGET,
        steps=STEPS,
        model=MODEL,
        model_id=MODEL_ID,
        gbm=GBM,
    ):
        self.data = df_from_api(data_api)
        self.target = target
        self.model = model
        self.model_id = model_id
        self.__train_holdout_split()
        self.pipeline = Pipeline(steps)
        self.gbm = gbm

    def __train_holdout_split(self):
        self.train_data = self.data.iloc[:, :-1]
        self.test_data = self.data.iloc[:, -1]

    def cv_splitter(self, k=10):
        ncols = self.train_data.shape[1] - 1
        step = ncols // k

        for i in np.arange(step, ncols, step=step):
            train_idx = self.train_data.iloc[:, :i].columns
            test_idx = self.train_data.iloc[:, i].columns
            yield train_idx, test_idx

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

    # def get_actual(self, data):
    #     # assumes a n_stations x n_obs orientation (i.e. the output of the pipeline) unless self.reduce_test == True
    #     return pd.Series(data.sum().sum()) if self.reduce_test else data.sum(axis=1)

    def get_metrics(self, y, yhat, funcs):
        return pd.DataFrame.from_dict(
            {i[0]: self.loss_func(y, yhat, i[1]) for i in funcs},
            orient="index",
            columns=[f"{self.model_id}_test"],
        )

    def expanding_window_cv(self):
        folds = self.cv_splitter()
        kpis = []
        print("conducting expanding window cross-validation...")
        i = 0
        for train_idx, test_idx in tqdm(folds):
            if self.gbm and i == 0:
                i += 1
                continue
            train = self.pipeline.fit_transform(self.train_data.loc[:, train_idx])
            test = self.pipeline.transform(self.train_data.loc[:, test_idx])

            y = test
            self.model.fit(train)

            if self.gbm:
                yhat = self.model.predict(train)
            else:
                yhat = self.model.predict(test)

            foldkpis = self.get_metrics(
                y,
                yhat,
                (
                    ("mse", self.mse),
                    ("rmse", self.rmse),
                    ("mdpe", self.mdpe),
                    ("mdape", self.mdape),
                ),
            )
            foldkpis.columns = [f"{col}_{i}" for col in foldkpis.columns]
            kpis.append(foldkpis)
            i += 1

        self.kpis = pd.concat(kpis, axis=1)
        self.kpis["fold_mu"] = self.kpis.mean(axis=1)

    def final_predict(self):
        print('predicting on test data...')
        train = self.pipeline.fit_transform(self.train_data)
        test = self.pipeline.transform(self.test_data)
        self.test_y = self.get_actual(test)
        self.model.fit(train)
        if self.gbm:
            self.test_yhat = self.model.predict(train)
        else:
            self.test_yhat = self.model.predict(test)

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

