import numpy as np
import pandas as pd

def exp_error(y, yhat, p=2):
    return pd.DataFrame((np.asarray(y) - np.asarray(yhat)) ** p, columns=y.columns, index=y.index)


def perc_error(y, yhat):
    return pd.DataFrame((np.asarray(y) - np.asarray(yhat)) * 100.0 / np.asarray(y), columns=y.columns, index=y.index)

def abs_perc_error(y, yhat):
    return abs(perc_error(y, yhat))