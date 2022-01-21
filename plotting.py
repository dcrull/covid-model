import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot,colors
import pandas as pd
import geopandas as gpd
from pathlib import Path
import contextily as ctx
import math

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

def single_ts(plot_data, ma=7, title=None, display_plot=True, **kwargs):
    fig, ax = plt.subplots(figsize=(15,10))
    ax.scatter(x=plot_data.index, y=plot_data, **kwargs)
    if ma is not None:
        ma_plot = plot_data.T.rolling(window=ma).mean().T
        ax.plot(ma_plot.index, ma_plot, c='indianred', label=f'{ma} day moving avg.')
    plt.legend(loc='best', fontsize=20)
    if title is not None:
        plt.savefig(Path("plots", f"{title}.png"), bbox_inches='tight')
        plt.title(title, fontsize=20)
    if display_plot: plt.show()
    return

def multi_ts(data, ma=7, title=None, display_plot=True, **kwargs):
    nrows= math.ceil(len(data)/2)
    fig, axes = plt.subplots(nrows, 2, figsize=(15,10))
    for i,idx in enumerate(data.index):
        n, m = i//2,i%2
        plot_data = data.loc[idx, :]
        axes[n,m].scatter(x=plot_data.index, y=plot_data, c='steelblue', **kwargs)
        axes[n, m].set_title(idx)
        if n != nrows-1: axes[n, m].xaxis.set_visible(False)
        if ma is not None:
            ma_plot = plot_data.T.rolling(window=ma).mean().T
            axes[n,m].plot(ma_plot.index, ma_plot, c='indianred', label=f'{ma} day moving avg.')
            if i == 0: axes[n,m].legend(loc='best', fontsize=10)
    if title is not None:
        plt.savefig(Path("plots", f"{title}.png"), bbox_inches='tight')
        fig.suptitle(title, fontsize=20)
    if display_plot: plt.show()
    return

def plot_forecast(X, yhat, idx=None):
    plot_ts(X, idx=idx, c='steelblue', lw=2, label='actual')
    plot_ts(yhat, idx=idx, c='indianred', ls='--', lw=2, label='forecast')
    plt.legend()
    plt.show()
    return

def boxplot_error(errdata):
    errdata.columns = [i.date() for i in errdata.columns]
    sns.boxplot(x='variable', y='value', data=pd.melt(errdata))
    plt.show()
    return

def get_window_sum(ts_data, win_size):
    return ts_data.iloc[:, -win_size:].sum(axis=1)

def get_win_on_win_growth_perc(ts_data, win_size):
    last_win = ts_data.iloc[:, -win_size:].sum(axis=1)
    prev_win = ts_data.iloc[:, -win_size*2:-win_size].sum(axis=1)
    return (last_win - prev_win)*100.0 / (prev_win + (prev_win==0)*1.0)

def merge_spatial(spatial_data, ref_data, target, multiplier):
    df = pd.concat([spatial_data[['geometry', 'POP']], ref_data], axis=1, sort=True)
    if multiplier is not None: df.loc[:, [target]] = (df.loc[:, [target]].astype(float) * multiplier).div(df['POP'].astype(float), axis=0)
    return df

def get_topX(ts_data, spatial_data, func, window, thresh, scale_col, multiplier=1000.0):
    if scale_col is not None:
        ts_data = (ts_data*multiplier).div(spatial_data[scale_col].astype(float), axis=0)
    return ts_data.loc[func(ts_data, window).sort_values(ascending=False)[:thresh].index, :]

def choro_plot(df, choro_col, title, display_plot, **plotkwargs):
    # just plots lower 48 states
    df = df.loc[df['geometry'].notnull(), :]
    fig, ax = plt.subplots(figsize=(15,15))
    gpd.GeoDataFrame(df, geometry='geometry', crs=4326).plot(column=choro_col, legend=True, ax=ax, **plotkwargs)
    # ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, zoom=2)
    ax.get_legend().set_bbox_to_anchor((1,0.5))
    ax.set_axis_off()
    plt.title(title, fontsize=20)
    plt.xlim([-127, -66])
    plt.ylim([23, 50])
    plt.savefig(Path("plots", f"{title}.png"), bbox_inches='tight')
    if display_plot: plt.show()
    return

def choro_cum(spatial_data, obs_data, choro_col, title, display_plot, multiplier=100000.0, **plotkwargs):
    last_date = obs_data['date'].max()
    df = obs_data.loc[obs_data['date'] == last_date, ['geoid', 'cases', 'deaths']].set_index('geoid')
    df = merge_spatial(spatial_data, df, choro_col, multiplier)
    title = f'{title} as of {last_date.date()}'
    return choro_plot(df, choro_col, title, display_plot, **plotkwargs)

def choro_window_plot(spatial_data, ts_data, window, func, choro_col, title, display_plot, multiplier=100000.0, **plotkwargs):
    df = func(ts_data, window)
    df.name = choro_col
    df = merge_spatial(spatial_data, df, choro_col, multiplier)
    return choro_plot(df, choro_col, title, display_plot, **plotkwargs)


def final_plots(obj, display_plot):
    state_obs = obj.load_and_prep(obj.nyt_state_url)
    state_spatial = obj.create_spatial_data(state_obs)
    state_cases_ts = obj.create_ts(state_obs, state_spatial, 'cases', get_diff=True, scale_data=False)
    state_deaths_ts = obj.create_ts(state_obs, state_spatial, 'deaths', get_diff=True, scale_data=False)
    county_obs = obj.load_and_prep(obj.nyt_county_url)
    county_spatial = obj.create_spatial_data(county_obs)
    county_cases_ts = obj.create_ts(county_obs, county_spatial, 'cases', get_diff=True, scale_data=False)
    county_deaths_ts = obj.create_ts(county_obs, county_spatial, 'deaths', get_diff=True, scale_data=False)

    # return state_obs, state_spatial, state_cases_ts, state_deaths_ts, county_obs, county_spatial, county_cases_ts, county_deaths_ts

    # cumulative cases/deaths by state/county
    choro_cum_maps = {'total cases per 100,000 by state': (state_spatial, state_obs, 'cases', 100000.0),
                      'total cases per 1,000 by county': (county_spatial, county_obs, 'cases', 1000.0),
                      'total deaths per 100,000 by state': (state_spatial, state_obs, 'deaths', 100000.0),
                      'total deaths per 1,000 by county': (county_spatial, county_obs, 'deaths', 1000.0),
                      }

    _ = [choro_cum(spatial_data=v[0],
                   obs_data=v[1],
                   choro_col=v[2],
                   title=k,
                   display_plot=display_plot,
                   multiplier=v[3],
                   cmap='inferno',
                   scheme='fisher_jenks',
                   missing_kwds={'color':'lightgray'}) for k,v in choro_cum_maps.items()]

    # last week total/growth of cases/deaths by state/county
    choro_window_maps = {'total cases last 7 days per 100,000 people by state': (state_spatial, state_cases_ts, get_window_sum, 'cases', 100000.0),
                         'total cases last 7 days per 1,000 people by county': (county_spatial, county_cases_ts, get_window_sum, 'cases', 1000.0),
                         'week on week % change in cases by state': (state_spatial, state_cases_ts, get_win_on_win_growth_perc, 'cases', None),
                         'week on week % change in cases by county': (county_spatial, county_cases_ts, get_win_on_win_growth_perc, 'cases', None),
                         'total deaths last 7 days per 100,000 people by state': (state_spatial, state_deaths_ts, get_window_sum, 'deaths', 100000.0),
                         'total deaths last 7 days per 1,000 people by county': (county_spatial, county_deaths_ts, get_window_sum, 'deaths', 1000.0),
                         'week on week % change in deaths by state': (state_spatial, state_deaths_ts, get_win_on_win_growth_perc, 'deaths', None),
                         'week on week % change in deaths by county': (county_spatial, county_deaths_ts, get_win_on_win_growth_perc, 'deaths', None),
                         }

    _ = [choro_window_plot(spatial_data=v[0],
                           ts_data=v[1],
                           window=7,
                           func=v[2],
                           choro_col=v[3],
                           title=k,
                           display_plot=display_plot,
                           multiplier=v[4],
                           cmap='inferno',
                           scheme='fisher_jenks',
                           missing_kwds={'color': 'lightgray'}) for k, v in choro_window_maps.items()]

    # ts of cases by county (mean)
    single_ts_plots = {'cases to date: mean of all counties': county_cases_ts.mean(),
                       'deaths to date: mean of all counties': county_deaths_ts.mean()}
    _ = [single_ts(v, ma=7, title=k, display_plot=display_plot) for k,v in single_ts_plots.items()]

    # top 10 counties
    top10_ts_plots = {'top 10 counties: total cases per 1,000 people last week': get_topX(county_cases_ts, county_spatial, get_window_sum, 7, 10, 'POP'),
                      'top 10 counties: total deaths per 1,000 people last week': get_topX(county_deaths_ts, county_spatial, get_window_sum, 7, 10, 'POP'),
                      'top 10 counties: week on week % change in cases': get_topX(county_cases_ts, county_spatial, get_win_on_win_growth_perc, 7, 10, 'POP'),
                      'top 10 counties: week on week % change in deaths': get_topX(county_deaths_ts, county_spatial, get_win_on_win_growth_perc, 7, 10, 'POP')}

    _ = [multi_ts(v, ma=7, title=k, display_plot=display_plot, s=5) for k,v in top10_ts_plots.items()]

#NOTES:
# TODO: top level counts
# TODO: numbers for legend (# deaths, % change, etc) for multi_ts
# TODO: distribution plots (by county for deaths/cases - total and growth week on week)
# TODO: animation of distribution change over time?

## Predictions
# Total cases (1,3,7, 30 days)
# Top 10 states/county (1,3,7,30 days)

# metrics @ diagnostics (states/counties cases/deaths)
# Error 1 vs t-1; 3 vs t-3, 7 vs t-7, 30 vs t-30
# top 10 best predictions states/counties (7 day)
# top 10 worst predictions states/counties (7 day)

# Maps
# choropleth map of total forecast cases/deaths state/county (7 days)
# choropleth of growth cases/deaths state/county (7 days)
# choropleth of error cases/deaths state/county (7 days)
# heatmap + forecast cases/deaths state/county