import numpy as np

def trim_leading(data, thresh):
    idx = 0
    for i in data:
        if i > thresh:
            break
        else:
            idx += 1
    return data[idx:]

def mse(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.mean((y - yhat) ** 2)

def rmse(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))

def mdpe(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.nanmedian((y - yhat) * 100.0 / y)

def mdape(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.nanmedian(abs(y - yhat) * 100.0 / y)