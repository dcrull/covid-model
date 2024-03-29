from pathlib import Path
# CONFIG VARS

MODEL_ID = 'test'
Path('outputs').mkdir(exist_ok=True)
Path('plots').mkdir(exist_ok=True)

# DATA
NYT_STATE_URL = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
NYT_COUNTY_URL= 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'

# territories to ignore
IGNORE_TERRITORIES = ['Virgin Islands',
                      'Guam',
                      'American Samoa',
                      'Northern Mariana Islands']