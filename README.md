# covid-model

### load nytimes data:
- make certain immediate transformations (convert data types, align geoids)

### load census data:
- need geometries
- need pop and demo data

### enrich data:
- join NYT data with  census data and other records

### plots
- map
- heatmap
- time series
- interactive
- movies (of map)

#### utils

#### main
- build data pipeline
    - run TSA transformations: diff, smoothing, etc
    - run enrichment
    - run model(s) ensemble models
    - tune
    - eval
    
### objectives
- Predict instances of cases 1, 3, 7 days out
- Predict instances of death 1, 3, 7 days out
- Predict top 25 counties / 5 states by {cases, death} 1, 3, 7
- Predict local "apex" of cases/deaths

### tasks
1. build this all for State level first
2.  XGB, Prophet, bayesian inf, LSTM
3. enrich data
4. ensemble approach w/ advanced features (SA, etc)