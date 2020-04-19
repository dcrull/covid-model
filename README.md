# covid-model

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
2. Prophet (bayesian)
3. enrich data
4. ensemble approach w/ advanced features (SA, etc)
5. func approx w/ DL