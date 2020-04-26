import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot,colors
import pandas as pd
import geopandas as gpd
from pathlib import Path
import contextily as ctx

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

def choro_cum_ct(spatial_data, obs_data, choro_col, title, multiplier=100000.0, **plotkwargs):
    last_date = obs_data['date'].max()
    last_cum = obs_data.loc[obs_data['date']==last_date, ['geoid','cases','deaths']].set_index('geoid')
    df = pd.concat([spatial_data.set_index('geoid')[['geometry','POP']], last_cum], axis=1, sort=True)
    df.loc[:, ['cases','deaths']] = (df.loc[: ,['cases', 'deaths']].astype(float) * multiplier).div(df['POP'].astype(float), axis=0)
    title = f'{title} as of {last_date.date()}'
    return choro_plot(df, choro_col, title, **plotkwargs)


#NOTES:
# - use 'inferno' for choropleth
# use scheme="fisher_jenks" for total cases state/county

# Plots and descriptive analytics
# choropleth map of total cases/deaths state/county (scaled by pop)
# choropleth map of growth in cases/deaths state/county (scaled by pop in last week)
# TS of total cases/deaths state/county (mean)
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