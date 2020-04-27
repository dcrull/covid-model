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

def single_ts(plot_data, ma=7, title=None, **kwargs):
    fig, ax = plt.subplots(figsize=(15,10))
    ax.scatter(x=plot_data.index, y=plot_data, **kwargs)
    if ma is not None:
        ma_plot = plot_data.T.rolling(window=ma).mean().T
        ax.plot(ma_plot.index, ma_plot, c='indianred', label=f'{ma} day moving avg.')
    plt.legend(loc='best', fontsize=20)
    if title is not None: plt.savefig(Path("plots", f"{title}.png"), bbox_inches='tight')
    plt.show()
    return

def multi_ts(data, ma=7, title=None, **kwargs):
    nrows= math.ceil(len(data)/2)
    fig, axes = plt.subplots(nrows, 2, figsize=(15,10))
    for i,idx in enumerate(data.index):
        n, m = i//2,i%2
        plot_data = data.loc[idx, :]
        axes[n,m].scatter(x=plot_data.index, y=plot_data, c='steelblue', label=idx, **kwargs)
        if ma is not None:
            ma_plot = plot_data.T.rolling(window=ma).mean().T
            axes[n,m].plot(ma_plot.index, ma_plot, c='indianred', label=f'{ma} day moving avg.')
        axes[n,m].legend(loc='best', fontsize=10)
    if title is not None: plt.savefig(Path("plots", f"{title}.png"), bbox_inches='tight')
    plt.show()
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

def choro_plot(df, choro_col, title, **plotkwargs):
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
    plt.show()

def choro_cum(spatial_data, obs_data, choro_col, title, multiplier=100000.0, **plotkwargs):
    last_date = obs_data['date'].max()
    df = obs_data.loc[obs_data['date'] == last_date, ['geoid', 'cases', 'deaths']].set_index('geoid')
    df = merge_spatial(spatial_data, df, choro_col, multiplier)
    title = f'{title} as of {last_date.date()}'
    return choro_plot(df, choro_col, title, **plotkwargs)

def choro_window_plot(spatial_data, ts_data, window, func, choro_col, title, multiplier=100000.0, **plotkwargs):
    df = func(ts_data, window)
    df.name = choro_col
    df = merge_spatial(spatial_data, df, choro_col, multiplier)
    return choro_plot(df, choro_col, title, **plotkwargs)


def final_plots(obj):
    state_obs = obj.load_and_prep(obj.nyt_state_url)
    state_spatial = obj.create_spatial_data(state_obs)
    state_cases_ts = obj.create_ts(state_obs, state_spatial, 'cases', get_diff=True, scale_data=False)
    state_deaths_ts = obj.create_ts(state_obs, state_spatial, 'deaths', get_diff=True, scale_data=False)
    county_obs = obj.load_and_prep(obj.nyt_county_url)
    county_spatial = obj.create_spatial_data(county_obs)
    county_cases_ts = obj.create_ts(county_obs, county_spatial, 'cases', get_diff=True, scale_data=False)
    county_deaths_ts = obj.create_ts(county_obs, county_spatial, 'deaths', get_diff=True, scale_data=False)

    # total by cases state
    choro_cum(spatial_data=state_spatial,
              obs_data=state_obs,
              choro_col='cases',
              title='total cases per 100,000 by state',
              cmap='inferno',
              schema='fisher_jenks',
              missing_kwds={'color':'lightgray'})

    # total by cases county
    choro_cum(spatial_data=county_spatial,
              obs_data=county_obs,
              choro_col='cases',
              title='total cases per 100,000 by county',
              cmap='inferno',
              schema='fisher_jenks',
              missing_kwds={'color': 'lightgray'})

    # total by deaths state
    choro_cum(spatial_data=state_spatial,
              obs_data=state_obs,
              choro_col='deaths',
              title='total deaths per 100,000 by state',
              cmap='inferno',
              schema='fisher_jenks',
              missing_kwds={'color': 'lightgray'})

    # total by deaths county
    choro_cum(spatial_data=county_spatial,
              obs_data=county_obs,
              choro_col='deaths',
              title='total deaths per 100,000 by county',
              cmap='inferno',
              schema='fisher_jenks',
              missing_kwds={'color': 'lightgray'})

    # last week total cases by state
    choro_window_plot(spatial_data=state_spatial,
                      ts_data=state_cases_ts,
                      window=7,
                      func=get_window_sum,
                      choro_col='cases',
                      title='total cases last 7 days per 100,000 people by state',
                      multiplier=100000.0,
                      cmap='inferno',
                      schema='fisher_jenks',
                      missing_kwds={'color':'lightgray'})

    # last week total cases by county
    choro_window_plot(spatial_data=county_spatial,
                      ts_data=county_cases_ts,
                      window=7,
                      func=get_window_sum,
                      choro_col='cases',
                      title='total cases last 7 days per 100,000 people by county',
                      multiplier=100000.0,
                      cmap='inferno',
                      schema='fisher_jenks',
                      missing_kwds={'color': 'lightgray'})

    # last week growth cases by state
    choro_window_plot(spatial_data=state_spatial,
                      ts_data=state_cases_ts,
                      window=7,
                      func=get_win_on_win_growth_perc,
                      choro_col='cases',
                      title='week on week % change in cases by state',
                      multiplier=1.0,
                      cmap='inferno',
                      schema='fisher_jenks',
                      missing_kwds={'color': 'lightgray'})

    # last week growth cases by county
    choro_window_plot(spatial_data=county_spatial,
                      ts_data=county_cases_ts,
                      window=7,
                      func=get_win_on_win_growth_perc,
                      choro_col='cases',
                      title='week on week % change in cases by county',
                      multiplier=1.0,
                      cmap='inferno',
                      schema='fisher_jenks',
                      missing_kwds={'color': 'lightgray'})

    # last week total deaths by state
    choro_window_plot(spatial_data=state_spatial,
                      ts_data=state_cases_ts,
                      window=7,
                      func=get_window_sum,
                      choro_col='deaths',
                      title='total deaths last 7 days per 100,000 people by state',
                      multiplier=100000.0,
                      cmap='inferno',
                      schema='fisher_jenks',
                      missing_kwds={'color': 'lightgray'})

    # last week total deaths by county
    choro_window_plot(spatial_data=county_spatial,
                      ts_data=county_cases_ts,
                      window=7,
                      func=get_window_sum,
                      choro_col='deaths',
                      title='total deaths last 7 days per 100,000 people by county',
                      multiplier=100000.0,
                      cmap='inferno',
                      schema='fisher_jenks',
                      missing_kwds={'color': 'lightgray'})

    # last week growth deaths by state
    choro_window_plot(spatial_data=state_spatial,
                      ts_data=state_cases_ts,
                      window=7,
                      func=get_win_on_win_growth_perc,
                      choro_col='deaths',
                      title='week on week % change in deaths by state',
                      multiplier=1.0,
                      cmap='inferno',
                      schema='fisher_jenks',
                      missing_kwds={'color': 'lightgray'})

    # last week growth deaths by county
    choro_window_plot(spatial_data=county_spatial,
                      ts_data=county_cases_ts,
                      window=7,
                      func=get_win_on_win_growth_perc,
                      choro_col='deaths',
                      title='week on week % change in deaths by county',
                      multiplier=1.0,
                      cmap='inferno',
                      schema='fisher_jenks',
                      missing_kwds={'color': 'lightgray'})

    def create_features(self, urlpath, target, get_diff, scale_data):
        obs_data = self.load_and_prep(urlpath)
        spatial_data = self.create_spatial_data(obs_data)
        features = self.create_ts(obs_data, spatial_data, target, get_diff, scale_data)
        return obs_data, spatial_data, features

#NOTES:
# - use 'inferno' for choropleth
# use scheme="fisher_jenks" for total cases state/county
# use missing_kwds={'color':'lightgray'} for missing values
# use per 1000.0 for county

# Plots and descriptive analytics
# choropleth map of total cases/deaths state/county (scaled by pop) [DONE]
# choropleth map of growth in cases/deaths state/county (scaled by pop in last week) [DONE]
# TS of total cases/deaths state/county (mean) [DONE]
# TS of total cases/deaths state/count (top 10 most in last week; fastest growth in last week)
# distribution of cases/deaths states/county
# distribution of growth rate cases/deaths state/county

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