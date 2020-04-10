import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib import pyplot,colors
import pandas as pd

# def get_clusters(data, model, **kwargs):
#     return model(**kwargs).fit(data)
#
# def pandas_plot(data, sample_size=None, trim_thresh=None, **kwargs):
#     if sample_size is not None: data = data.sample(sample_size)
#     if trim_thresh is not None:
#         data = data.values
#         _ = [plt.plot(data[i, data[i, :] > trim_thresh], **kwargs) for i in range(len(data))]
#         title = f'ct per day starting when count above {trim_thresh}'
#     else:
#         data.T.plot(**kwargs)
#         title = 'ct per day'
#     plt.title(title)
#     plt.show()
#     return

def trim_leading(data, thresh):
    idx = 0
    for i in data:
        if i > thresh:
            break
        else:
            idx += 1
    return data[idx:]

def heatmap(df, target, sort_col='2020-04-01', norm=colors.LogNorm(vmin=1), forecast_line=None, **kwargs):
    pyplot.imshow(df.sort_values(sort_col, ascending=False),norm=norm, aspect='auto', **kwargs)
    if forecast_line is not None: plt.axvline(df.shape[1] - forecast_line, c='r', lw=.5)
    pyplot.title(f'heat map of {target} sorted on {sort_col}')
    pyplot.show()

def plot_ts(data, idx=None, **kwargs):
    if idx is None:
        plot_data = data.mean()
    elif isinstance(idx, int):
        plot_data = data.sample(idx, random_state=32).T
    elif isinstance(idx, str):
        plot_data = data.loc[idx, :]
    plot_data.plot(**kwargs)
    return

def plot_forecast(X, yhat, idx=None, **kwargs):
    plot_ts(X, idx=idx, c='steelblue', lw=2, label='actual')
    plot_ts(yhat, idx=idx, c='indianred', ls='--', lw=2, label='forecast')
    plt.show()

def boxplot_error(errdata):
    errdata.columns = [i.date() for i in errdata.columns]
    sns.boxplot(x='variable', y='value', data=pd.melt(errdata))
    plt.show()
