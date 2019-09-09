
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyod.models.hbos import HBOS
from .misc import print_c, is_number, make_conf
from .imputer import remove_single_value_features, collapse_category
from .constants import DATE_TIME_FEATURES

def extract_numericals(df: pd.DataFrame,col: str, pbar :tqdm = None, verbose: bool = True):
    df = df.copy(deep = True)
    msg = 'creating '+col+'_numerical'
    if pbar is None:
        print_c(verbose,msg)
    else:
        pbar.set_description(msg)
    df[col+'_numerical'] = np.NaN
    df.loc[is_number(df[col]) ,col+'_numerical'] = [float(item) for item in df[col][is_number(df[col])]]
    return df

def extract_numericals_forall(df: pd.DataFrame, verbose: bool = True):
    df = df.copy(deep=True)
    df.columns = [column.lower() for column in df.columns]
    conf = make_conf(df)
    conf = [col for col in list(conf.keys())  if conf[col]['type'] == 'categorical']
    cols = tqdm(conf)
    new_cols = []
    for col in cols:
        df = extract_numericals(df,col,cols,verbose)
        new_cols.append(col+'_numerical')
    df = remove_single_value_features(df,verbose,include = new_cols)
    return df

def extract_is_nan(df: pd.DataFrame, col: str, pbar: tqdm = None, verbose: bool = True) -> pd.DataFrame:
    """
    Create a null column
    :param df: the data
    :param col: the column name
    :param conf: the config dir
    :param pbar: tqdm progress bar
    :param verbose: verbosity
    :return:
    """
    df = df.copy(deep=True)
    nulls = df[col].isnull()
    if nulls.sum() == 0:
        return df

    msg = "Adding "+col+'_'+'isnull column'
    if pbar is None:
        print_c(verbose, msg)
    else:
        pbar.set_description(msg)
    df[col+'_'+'isnull'] = df[col].isnull()
    return df

def extract_is_nan_forall(df: pd.DataFrame, verbose: bool = True):
    df = df.copy(deep=True)
    df.columns = [column.lower() for column in df.columns]
    conf = make_conf(df)
    conf = [col for col in list(conf.keys())  if conf[col]['type'] == 'numerical']
    cols = tqdm(conf)
    new_cols = []
    for col in cols:
        df = extract_is_nan(df,col,cols,verbose)
        new_cols.append(col+'_isnull')
    df = remove_single_value_features(df,verbose,include = new_cols)
    return df


def extract_is_inf(df: pd.DataFrame, col: str, pbar = None, verbose: bool = True, seperate_ninf: bool = False) -> pd.DataFrame:
    """
    Create an is_inf column
    :param df: the data
    :param col: the column name
    :param conf: the config dir
    :param pbar: tqdm progress bar
    :return:
    """
    df = df.copy(deep=True)

    msg = "Adding "+col+'_'+'isinf column'
    if pbar is None:
        print_c(verbose,msg)
    else:
        pbar.set_description(msg)
    df[col+'_'+'isinf'] = 0
    df.loc[np.isinf(df[col]) ,col+'_'+'isinf'] = 1
    if seperate_ninf:
        df.loc[np.isneginf(df[col]) ,col+'_'+'isinf'] = -1
    else:
        df.loc[np.isneginf(df[col]) ,col+'_'+'isinf'] = 1
    return df

def extract_is_inf_forall(df: pd.DataFrame, verbose: bool = True, seperate_ninf: bool = False):
    df = df.copy(deep=True)
    df.columns = [column.lower() for column in df.columns]
    conf = make_conf(df)
    conf = [col for col in list(conf.keys())  if conf[col]['type'] == 'numerical']
    cols = tqdm(conf)
    new_cols = []
    for col in cols:
        df = extract_is_inf(df,col,cols,verbose,seperate_ninf=seperate_ninf)
        new_cols.append(col+'_isinf')
    df = remove_single_value_features(df,verbose,include = new_cols)
    return df

def extract_datetime_features(df: pd.DataFrame, col: str, pbar: tqdm = None, verbose: bool = True)\
        -> pd.DataFrame:
    """
    Process a datetime column for machine learning
    :param df: the data
    :param col: the column
    :param conf: the config dir
    :param pbar: tqdm progress bar
    :param verbose: verbosity
    :return: updated dataframe
    """
    df = df.copy(deep=True)
    numbers = DATE_TIME_FEATURES['NUMERICS']
    booleans = DATE_TIME_FEATURES['BOOLEANS']

    msg = "Extracting 15 datetime features from "+str(col)
    if pbar is None:
        print_c(verbose, msg)
    else:
        pbar.set_description(msg)

    mask = ~df[col].isnull()
    for feature in numbers:
        df[col+'_'+feature] = 0
        df.loc[mask ,col+'_'+feature] =  [getattr(x, feature) for x in df[col][mask]]

    for feature in booleans:
        df[col+'_'+feature] = -1
        df.loc[mask ,col+'_'+feature] =  [getattr(x, feature) for x in df[col][mask]]
        df[col+'_'+feature] = df[col+'_'+feature].astype('int32')

    return df

def extract_datetime_features_forall(df: pd.DataFrame, verbose: bool = True):
    df = df.copy(deep=True)
    df.columns = [column.lower() for column in df.columns]
    conf = make_conf(df)
    conf = [col for col in list(conf.keys())  if conf[col]['type'] == 'datetime']
    cols = tqdm(conf)
    features = DATE_TIME_FEATURES['NUMERICS'] + DATE_TIME_FEATURES['BOOLEANS']
    new_cols = []
    for col in cols:
        df = extract_datetime_features(df,col,cols,verbose)
        for feature in features:
            new_cols.append(col+'_'+feature)
    df = remove_single_value_features(df,verbose,include = new_cols)
    return df


def onehot_encode(df: pd.DataFrame, col: str, pbar: tqdm = None, verbose: bool = True, replace = True, 
        allow_na: bool = True ,cutoff: int = 50, cutoff_class: str = 'other' ) -> pd.DataFrame:
    """
    One hot encode
    :param df: the data
    :param col: the column name
    :param conf: the config dir
    :param pbar: tqdm progress bar
    :param verbose: verbosity
    :return:
    """
    df = df.copy(deep=True)
    msg = "One hot encoding "+ str(col)
    
    if pbar is None:
        print_c(verbose, msg)
    else:
        pbar.set_description(msg)

    if cutoff > 0:
        df = collapse_category(df = df, col = col, pbar = pbar, verbose = verbose, cutoff_class = cutoff_class, cutoff = cutoff)
    one_hot = pd.get_dummies(df[col], prefix=col+'_onehot_', dummy_na = allow_na)

    try:
        df = df.join(one_hot)
    except Exception as e:
        msg = "Ignoring "+str(col)
        if pbar is None:
            print_c(verbose, msg)
        else:
            pbar.set_description(msg)
    if replace:
        df.drop(col, axis=1, inplace=True)
        msg = "Dropping "+col
        if pbar is None:
            print_c(verbose, msg)
        else:
            pbar.set_description(msg)
    return df

def onehot_encode_all(df: pd.DataFrame, verbose: bool = True, allow_na: bool = True, cutoff: int = 50, 
        cutoff_class: str = 'other') -> pd.DataFrame:
    df = df.copy(deep=True)
    df.columns = [column.lower() for column in df.columns]
    conf = make_conf(df)
    # Filter categorical columns
    conf = [col for col in list(conf.keys())  if conf[col]['type'] == 'categorical']
    cols = tqdm(conf)

    for col in cols:
        df = onehot_encode(df,col,cols,verbose,allow_na = allow_na,cutoff = cutoff, cutoff_class = cutoff_class)
    
    new_cols = [col for col in df.columns if '_onehot_' in col]
    df = remove_single_value_features(df,verbose,include = new_cols)

    return df


def extract_is_outlier(df: pd.DataFrame, col: str, pbar = None, verbose: bool = True, model = None, 
        outliers_fraction: float = 0.05, replace_with = None) -> pd.DataFrame:
    """
    Create an is_outlier column
    :param df: the data
    :param col: the column name
    :param conf: the config dir
    :param pbar: tqdm progress bar
    :return:
    """
    df = df.copy(deep=True)

    msg = "Trying to find outliers in "+str(col)
    if pbar is None:
        print_c(verbose,msg)
    else:
        pbar.set_description(msg)

    if model is None:
        model = HBOS(contamination=outliers_fraction)
    X = df[col].astype(np.float32)
    mask = ~( np.isnan(X) | np.isinf(X) | np.isneginf(X))
    model.fit(X[mask].to_frame())
    preds = model.predict(X[mask].to_frame())
    df[col+'_'+'isoutlier'] = 0
    df.loc[mask,col+'_'+'isoutlier'] = preds
    
    
    if replace_with is not None:
        msg = "Replacing outliers in "+str(col)+" with "+str(replace_with)
        if pbar is None:
            print_c(verbose,msg)
        else:
            pbar.set_description(msg)
        df.loc[df[col+'_'+'isoutlier'] == 1, col ] = replace_with
  
    return df

def extract_is_outlier_forall(df: pd.DataFrame, verbose: bool = True, model = None, 
        outliers_fraction: float = 0.05, replace_with = None) -> pd.DataFrame:
    df = df.copy(deep=True)
    df.columns = [column.lower() for column in df.columns]
    conf = make_conf(df)
    
    if model is None:
        model = HBOS(contamination=outliers_fraction)

    # Filter numerical columns
    conf = [col for col in list(conf.keys())  if conf[col]['type'] == 'numerical']
    cols = tqdm(conf)

    for col in cols:
        df = extract_is_outlier(df, col, cols, verbose, model = model, 
            outliers_fraction = outliers_fraction, replace_with = replace_with)
    
    new_cols = [col for col in df.columns if '_isoutlier' in col]
    df = remove_single_value_features(df,verbose,include = new_cols)

    return df