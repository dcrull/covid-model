import pandas as pd
from sklearn.pipeline import Pipeline
from config import NYT_COUNTY_URL, NYT_STATE_URL
from transformers.nyt import PrepNYT
from transformers.create_ts import CreateTS
from transformers.transform_ts import TSRate, GF

STEPS = [
    ('prepnyt', PrepNYT()),
    ('create_ts', CreateTS(response_var='cases')),
    ('first_diff', TSRate()),
    ('gf_s1.5', GF(sigma=1.5)),
]

class CVPredict:
    def __init__(self,
                 nyt_county_url=NYT_COUNTY_URL,
                 nyt_state_url=NYT_STATE_URL,
                 steps=STEPS):
        self.covid_us_county = self.__load_nyt(nyt_county_url)
        self.covid_us_state = self.__load_nyt(nyt_state_url)
        self.pipeline = Pipeline(steps)

    def __load_nyt(self, url):
        return pd.read_csv(url, parse_dates=['date'])

    def data_prep(self, data):
        return self.pipeline.fit_transform(data)
