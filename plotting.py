import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import pyplot,colors
import pandas as pd

def cv_fold_error(cvresults):
    pass



def get_clusters(data, model, **kwargs):
    return model(**kwargs).fit(data)

def pandas_plot(data, sample_size=None, trim_thresh=None, **kwargs):
    if sample_size is not None: data = data.sample(sample_size)
    if trim_thresh is not None:
        data = data.values
        _ = [plt.plot(data[i, data[i, :] > trim_thresh], **kwargs) for i in range(len(data))]
        title = f'ct per day starting when count above {trim_thresh}'
    else:
        data.T.plot(**kwargs)
        title = 'ct per day'
    plt.title(title)
    plt.show()
    return

def trim_leading(data, thresh):
    idx = 0
    for i in data:
        if i > thresh:
            break
        else:
            idx += 1
    return data[idx:]

def heatmap(df, target, sort_col, norm=colors.LogNorm(vmin=1), **kwargs):
    pyplot.imshow(df.sort_values(sort_col, ascending=False),norm=norm, aspect='auto', **kwargs)
    pyplot.title(f'heat map of {target} sorted on {sort_col}')
    pyplot.show()

def plot_mean_ts(obj, target):
    X = pd.concat([obj.test_X, obj.test_y], axis=1)
    X2 = pd.concat([obj.test_X, obj.test_yhat], axis=1)
    X.columns = list(obj.test_X.columns) + list(obj.test_y.columns)
    X2.columns = X.columns

    X.mean().plot(label='actual;')
    X2.mean().plot(label='predicted')
    pyplot.title(f'mean {target} actual vs predicted')
    pyplot.legend(loc='best')
    pyplot.show()