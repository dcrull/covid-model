from sklearn.pipeline import Pipeline
from transformers.counter_id import CounterID
from transformers.create_ts import CreateTS
from transformers.group import TimeGroup
from transformers.station_grp import StationGroup
from transformers.model import Naive, SimpleARIMA, SimpleGBM, FBProph
from utils import load_data
from datetime import timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import dill

## GLOBAL CONFIG VARIABLESS FOR VARIOUS MODELS

FREQUENCY = "D"
# MODEL = Naive(method=np.mean, kwargs={'axis': 1}, frequency=FREQUENCY)
# MODEL_ID = 'naive_model'
# REDUCE_TEST = False
# GBM = False

# MODEL = SimpleARIMA(lag_order=7, degree_of_diff=0, ma_window=0)
# MODEL_ID = 'simpleARIMA'
# REDUCE_TEST = True
# GBM = False

# MODEL = SimpleGBM(n_jobs=-1)
# MODEL_ID = 'XGB'
# REDUCE_TEST = False
# GBM = True

MODEL = FBProph()
MODEL_ID = "prophet"
REDUCE_TEST = True
GBM = False

STEPS = [
    ("get_counter", CounterID()),
    ("create_ts", CreateTS()),
    ("groupby", TimeGroup(frequency=FREQUENCY)),
    ("station_grp", StationGroup()),
]


class TSPredict:
    def __init__(
        self,
        model=MODEL,
        model_id=MODEL_ID,
        dpath=Path("data", "turnstiles_hw_2020.csv"),
        frequency=FREQUENCY,
        holdout_date_cutoff="2016-12-01",
        steps=STEPS,
        reduce_test=REDUCE_TEST,
        gbm=GBM,
    ):
        self.model = model
        self.model_id = model_id
        self.dpath = dpath
        self.frequency = frequency
        self.data = load_data(self.dpath)
        self.__train_holdout_split(holdout_date_cutoff, datecol="DATE_TIME")
        self.pipeline = Pipeline(steps)
        self.reduce_test = reduce_test
        self.gbm = gbm

    def __train_holdout_split(self, holdout_date_cutoff, datecol):
        self.train_data = self.data.loc[self.data[datecol] < holdout_date_cutoff, :]
        self.test_data = self.data.loc[self.data[datecol] >= holdout_date_cutoff, :]

    def cv_splitter(self, date_col="DATE_TIME"):
        date_range = pd.date_range(
            self.train_data[date_col].min(),
            self.train_data[date_col].max() - timedelta(days=30),
            freq="M",
            normalize=True,
        )
        for i in date_range:
            train_idx = self.train_data[date_col] <= i
            test_idx = (self.train_data[date_col] > i) & (
                self.train_data[date_col] <= (i + timedelta(days=30))
            )
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

    def get_actual(self, data):
        # assumes a n_stations x n_obs orientation (i.e. the output of the pipeline) unless self.reduce_test == True
        return pd.Series(data.sum().sum()) if self.reduce_test else data.sum(axis=1)

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
            train = self.pipeline.fit_transform(self.train_data.loc[train_idx, :])
            test = self.pipeline.transform(self.train_data.loc[test_idx, :])

            y = self.get_actual(test)
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