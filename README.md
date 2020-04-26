# covid-model

Simple forecasting model for COVID-19 cases in the U.S. based on open data from the NYTimes: https://github.com/nytimes/covid-19-data

Built using Scikit-learn pipeline to support modular data preprocessing and model development.

Main forecast model driven by Facebook's open source Prophet library (https://facebook.github.io/prophet/), which in turn relies on STAN as the Bayesian inference engine.

[WIP]