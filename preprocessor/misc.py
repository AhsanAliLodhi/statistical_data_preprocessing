
import pandas as pd
import numpy as np
from tqdm import tqdm


def print_c(verbose=True, *argv):
    if verbose:
        print(*argv)

def hasnulls(df: pd.DataFrame, verbose: bool = False):
    nulls = df.isnull().sum().sum()
    print_c(verbose, "Number of nulls", nulls)
    if nulls > 0:
        print_c(verbose, df.isnull().sum().sort_values(ascending=False))
        return True
    else:
        return False

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    except TypeError:
        try:
            mask = [is_number(item) for item in s]
            return mask
        except ValueError:
            pass

def read_csv(df_name: str = 'input.csv',encoding=None, verbose = False):
    df = pd.read_csv(df_name,encoding = encoding,parse_dates=True)
    print_c(verbose,'Trying to read datetime columns')
    objs = df.select_dtypes(include ='object')
    cols = tqdm(objs.columns)
    cols = tqdm(cols)
    for col in cols:
        try:
            dates = pd.to_datetime(objs[col])
            #cols.set_description(col+" as Datetime")
        except Exception as e:
            err = str(e)
            if "Out of bounds" in err:
                dates = pd.to_datetime(objs[col],errors='coerce')
                #cols.set_description(col+" as Datetime")
            else:
                #cols.set_description(col+" Ignored")
                continue
        #cols.set_description('Found dates in '+str(col))
        uniquedates = len(dates.unique())
        if uniquedates > 1:
            #cols.set_description(str(col)+"saved as datetime")
            df[col] = dates
    df.columns = [col.lower() for col in df.columns]
    return df

def get_ml_type(col:pd.Series):
    dtype = str(col.dtype)
    if 'bool' in dtype or 'float' in dtype or 'int' in dtype or 'double' in dtype:
        return "numerical"

    elif 'time' in dtype or 'date' in dtype:
        return "datetime"
    else:
        return "categorical"
    return col.dtype

def make_conf(df:pd.DataFrame,verbose=None):
    conf = {}
    for col in df.columns:
        conf[col.lower()]={"type":get_ml_type(df[col])}
    return conf

