import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from tqdm import tqdm
from .misc import make_conf, print_c, is_number
from .constants import FILL_NAN_METHODS, FILL_INF_METHODS

def remove_single_value_features(df: pd.DataFrame = None , verbose: bool = False, include: list = []) -> pd.DataFrame:
    df = df.copy(deep = True)
    include = [item.lower() for item in include]

    # If include is empty, then include all columns
    if len(include) == 0:
        conf = make_conf(df)
        include = list(conf.keys())
        print_c(verbose,"Checking all columns")
    else:
        include = set(include) & set(df.columns)
        print_c(verbose,"Cheking columns provided in Include",len(include))

    print_c(verbose, "Gathering useless columns, columns with just one unique value")
    to_delete = []
    cols = tqdm(include)
    cols.set_description("Useless columns found 0 ")
    for col in cols:
        counts = df[col].unique()
        if len(counts) < 2:
            to_delete.append(col)
            cols.set_description("Useless columns found "+str(len(to_delete))+"")
    print_c(verbose, "Deleting..")
    df.drop(to_delete, axis=1, inplace=True)
    print_c(verbose, "Deletion complete")
    print_c(verbose, "Dataframe shape after deleting ", df.shape)
    return df

def remove_numericals_from_categories(df: pd.DataFrame = None , verbose: bool = False, include: list = [], by = None) -> pd.DataFrame:
    df = df.copy(deep = True)
    include = [item.lower() for item in include]
    
    conf = make_conf(df)
    conf = [col for col in list(conf.keys())  if conf[col]['type'] == 'categorical']

    # If include is empty, then include all columns
    if len(include) == 0:
        include = conf
        print_c(verbose,"Removing numericals from all categorical columns")
    else:
        include = set(include) & set(conf)
        print_c(verbose,"Removing numericals from categorical columns provided in Include",len(include))

    cols = tqdm(include)
    for col in cols:
        cols.set_description("Removing numericals from "+col)
        mask = is_number(df[col])
        df.loc[mask,col] = [by for item in df[col][mask]]
    return df


def fillnans(df: pd.DataFrame = None , verbose: bool = False, include: list = [], by = FILL_NAN_METHODS['MEDIAN']) -> pd.DataFrame:
    df = df.copy(deep = True)

    # Get all columns
    conf = make_conf(df)

    # Filter numerical columns
    conf = [col for col in list(conf.keys())  if conf[col]['type'] == 'numerical']

    include = [item.lower() for item in include]
    
    # If include is empty, then include all columns
    if len(include) == 0:
        include = conf
        print_c(verbose,"Filling nans in all columns")
    else:
        include = set(include) & set(conf)
        print_c(verbose,"Filling nans in the columns provided in Include",len(include))

    cols = tqdm(include)

    for col in cols:
        if by == FILL_NAN_METHODS['MEDIAN']:
            value = df[col].median()
        elif by == FILL_NAN_METHODS['MEAN']:
            value = df[col].mean()
        else:
            value = by
        cols.set_description("Filling nans in "+col+' with '+str(value))
        df[col] = df[col].fillna(value)
    return df

def fillinfs(df: pd.DataFrame = None , verbose: bool = False, include: list = [], by = FILL_INF_METHODS['MAXMIN'], max_min_factor: int = 100) -> pd.DataFrame:
    df = df.copy(deep = True)

    # Get all columns
    conf = make_conf(df)

    # Filter numerical columns
    conf = [col for col in list(conf.keys())  if conf[col]['type'] == 'numerical']

    include = [item.lower() for item in include]
    
    # If include is empty, then include all columns
    if len(include) == 0:
        include = conf
        print_c(verbose,"Filling infs in all columns")
    else:
        include = set(include) & set(conf)
        print_c(verbose,"Filling infs in the columns provided in Include",len(include))

    cols = tqdm(include)
    
    for col in cols:
        if by == FILL_INF_METHODS['MAXMIN']:
            mask = ~( np.isnan(df[col]) | np.isinf(df[col]) | np.isneginf(df[col]))
            valid_numbers = df[col][mask]
            col_min = max_min_factor * min(valid_numbers) # get the min of column
            col_max = max_min_factor * max(valid_numbers) # get the max of column
            cols.set_description("Filling infs in "+col+' with '+str(col_min)+', '+str(col_max))
            df[col].replace([np.inf],col_max)
            df[col].replace([-np.inf ],col_min)
        else:
            cols.set_description("Filling infs in "+col+' with '+str(by))
            df[col].replace([np.inf, -np.inf ],by)

    return df

def remove_datetimes(df: pd.DataFrame = None , verbose: bool = False, include: list = []) -> pd.DataFrame:
    df = df.copy(deep = True)
    include = [item.lower() for item in include]
    # Get all columns
    conf = make_conf(df)

    # Filter datetime columns
    conf = [col for col in list(conf.keys())  if conf[col]['type'] == 'datetime']

    # If include is empty, then include all columns
    if len(include) == 0:
        include = conf
        print_c(verbose,"Deleting all Datetime columns")
    else:
        include = set(include) & set(df.columns)
        print_c(verbose,"Deleting datetime columns provided in Include",len(include))

    df.drop(include, axis=1, inplace=True)

    return df
    

def normalize(df: pd.DataFrame = None , verbose: bool = False, include: list = [], scaler = None) -> pd.DataFrame:
    df = df.copy(deep = True)
    include = [item.lower() for item in include]
    
    if scaler is None:
        scaler = StandardScaler()
    # If include is empty, then include all columns
    if len(include) == 0:
        conf = make_conf(df)
        include = list(conf.keys())
        print_c(verbose,"Normalizing numbers from all columns")
    else:
        include = set(include) & set(df.columns)
        print_c(verbose,"Normalizing numbers from columns provided in Include",len(include))

    scaled_data = scaler.fit_transform(df)
    df = pd.DataFrame(data = scaled_data, columns = df.columns)
    return df

def collapse_category(df: pd.DataFrame, col: str, pbar = None, verbose: bool = True, cutoff_class: str = 'other', cutoff: int = 50) -> pd.DataFrame:
    """
    Merge values of a column with count less then cut off into one class
    :param df: the data
    :param col: the column name
    :param conf: the config dir
    :param pbar: tqdm progress bar
    :return: updated dataframe
    """
    df = df.copy(deep=True)
    uniques = df[col].value_counts()
    insignificants = uniques.index[uniques < cutoff]
    insignificants_percent = round(sum(uniques[insignificants])*100/len(df),2)
    msg = "Total uniques "+ str(len(uniques)) +", Insignificants " +str(len(insignificants)) + ", Insignificants "+str(insignificants_percent)+"%"
    if pbar is None:
        print_c(verbose,msg)
    else:
        pbar.set_description(msg)
    df[col] = np.where(df[col].isin(insignificants), cutoff_class, df[col])
    return df