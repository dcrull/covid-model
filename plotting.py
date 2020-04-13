import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot,colors
import pandas as pd

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
        plt.scatter(x=plot_data.index, y=plot_data, **kwargs)
    elif isinstance(idx, int):
        plot_data = data.sample(idx, random_state=32).T
        plot_data.plot(**kwargs)
    elif isinstance(idx, str):
        plot_data = data.loc[idx, :]
        plt.scatter(x=plot_data.index, y=plot_data, **kwargs)
    return

def plot_forecast(X, yhat, idx=None):
    plot_ts(X, idx=idx, c='steelblue', lw=2, label='actual')
    plot_ts(yhat, idx=idx, c='indianred', ls='--', lw=2, label='forecast')
    plt.legend()
    plt.show()

def boxplot_error(errdata):
    errdata.columns = [i.date() for i in errdata.columns]
    sns.boxplot(x='variable', y='value', data=pd.melt(errdata))
    plt.show()
