# TODO: Substitue all strings with constants

# Column types (in context of data science)
TYPE = {
    'numerical',
    'categorical',
    'datetime'
}

# List of date time features available in pandas date time columns
DATE_TIME_FEATURES = {
    'NUMERICS':['year', 'month', 'day', 'hour', 'dayofyear', 'weekofyear', 'week', 'dayofweek',
               'quarter'],
    'BOOLEANS':['is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
                'is_year_start', 'is_year_end']
}

# Possible methods to fill nans in numerical columns
FILL_NAN_METHODS = {
    'MEAN':'mean','MEDIAN':'median'
}

# Possible methods to fill infs in numerical columns
FILL_INF_METHODS = {
    'MAXMIN':'maxmin','NAN':'nan'
}