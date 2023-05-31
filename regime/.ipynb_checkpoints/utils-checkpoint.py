import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

"""
checked functions

def to_monthly_period_index(df, col_dt, format = None):

def compound_daily_return_to_other_freq(ret_ser, freq = 'M'):

def insert_cols(df, loc_names, new_colnames, value):


"""



# checked
def to_monthly_period_index(df, col_dt, format = None):
    """
    convert a column <col_dt> of dates, in <df>, into monthly Period dtype, and set it as the index of <df>.
    The original column is dropped.
    
    Parameters:
    ---------------------------------
    df: DataFrame

    col_dt: string
        column name of <df>. The column is a date series to be converted to monthly Periods.
    format:
        usd in parsing datetime type pd.to_datetime(df[col_dt], format = format)
        
    Returns:
    ---------------------------------
    a DataFrame with monthly Period as index.
    """
    # extract date column
    date_col = df[col_dt]
    # drop it 
    df_new = df.drop(columns = [col_dt])
    # change to monthly period
    df_new['month'] = pd.to_datetime(date_col, format = format).dt.to_period('M')
    # set as index
    return df_new.set_index('month')

# checked
def compound_daily_return_to_other_freq(ret_ser, freq = 'M'):
    """
    Compound a daily return series, with date as index, into a monthly/weekly return series.
    """
    ret_ser_copy = ret_ser.copy()
    ret_ser_copy.index = pd.PeriodIndex(ret_ser_copy.index, freq='d')
    ret_ser_compound = ret_ser_copy.resample(freq).apply(lambda x: x.add(1).prod()-1)
    return ret_ser_compound


# checked
def _combine_ser_of_ser_into_df(ser_of_ser, use_new_cols=None):
    """
    combine a series of series into dataframe.
    
    single na in the values of <ser_of_ser> is allowed.
    """
    ser_of_ser_dropna = ser_of_ser.dropna()
    if use_new_cols is None:   # don't use new column names
        # list a series
        return pd.DataFrame(list(ser_of_ser_dropna.values), index=ser_of_ser_dropna.index).reindex(ser_of_ser.index)
    else:        
        # use new column names, a list of list
        return pd.DataFrame(list(ser_of_ser_dropna.map(lambda x: x.values).values),
                            index=ser_of_ser_dropna.index,
                           columns=use_new_cols).reindex(ser_of_ser.index)


def apply_func_to_ser_df(ser, func, *args, return_as_df = False, use_new_cols=None, **kwargs):
    """
    apply function to each value in a series, or each row (as a series) in a DataFrame.
        
    The output can be either a DataFrame or series, by parameter <return_as_df>. 
    If a DataFrame is desired, it is assumed the output of the function is a Series whose index is the same for every row.
    Otherwise it is recommended to output as a series, and further process out of this function.
    
    The function has the same functionality as `ser.map()` or `df.apply(axis=1)`, but you can monitor progress by tqdm in this function.
    
    NA is allowed.
    
    Parameters:
    --------------------------------
    ser: Series or DataFrame
        If DataFrame, apply function to each row.
    func:
    
    args:
        for <func>
    return_as_df: boolean, default False
        return as a DataFrame if True.
    kwargs:
        for <func>
        
    Returns:
    -------------------------------------
    Series or DataFrame
    
    """
    ret_lst_dropna = []
    # index without NA values
    index_dropna = ser.index[ser.notnull()] if isinstance(ser, pd.Series) else ser.index[ser.notnull().any(axis=1)]
#     return index_notna
    for ind in tqdm(index_dropna):
        ret_lst_dropna.append(func(ser.loc[ind], *args, **kwargs))
    # series of series
    ret_ser_of_ser_dropna = pd.Series(ret_lst_dropna, index=index_dropna)
    
    if not return_as_df: # return as series
        return ret_ser_of_ser_dropna.reindex(ser.index)
    else:                # return as df
        ret_df_dropna = _combine_ser_of_ser_into_df(ret_ser_of_ser_dropna, use_new_cols)
        return ret_df_dropna.reindex(ser.index) 
    
# checked    
def insert_cols(df, loc_names, new_colnames=None, value=None):
    """
    insert columns to <df> in place. 
    insert each column of <value> at the location in <loc_names> (as column name), with new column name in <new_colnames>
    
    <loc_names>, <new_colnames> are strings, value is a Series; 
    or <loc_names>, <new_colnames> are lists of strings, value is a DataFrame.
    
    Examples:
    ------------------------------------
    >>> df = pd.DataFrame([[1, 2], [3, 4]], index=['Mon', 'Tues'], columns=['a', 'b'])
    >>> df
          a  b
    Mon   1  2
    Tues  3  4
    >>> insert_cols(df, 'b', 'new_col', pd.Series([5,6], index=['Mon', 'Tues']))
    >>> df
          a  new_col  b
    Mon   1        5  2
    Tues  3        6  4
    >>> insert_cols(df, ['new_col', 'b'], ['new_col_1', 'new_col_2'], pd.DataFrame([[7, 8], [9, 10]], index=['Mon', 'Tues']))
    >>> df
          a  new_col_1  new_col  new_col_2  b
    Mon   1          7        5          8  2
    Tues  3          9        6         10  4
    """
    if new_colnames is None:
        new_colnames = value.name if isinstance(value, pd.Series) else value.columns
    if isinstance(loc_names, str):
        df.insert(df.columns.get_loc(loc_names), new_colnames, value)
    else:
        for i, (loc_name, new_colname) in enumerate(zip(loc_names, new_colnames)):
            df.insert(df.columns.get_loc(loc_name), new_colname, value.iloc[:, i])
    return 

    


def load_CPI_index():
    """
    load CPI index
    """
    html = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    cpi_ser = to_monthly_period_index(pd.read_csv(html), "DATE").squeeze()
    return cpi_ser
    
# def load_processed_mkt_data(path):
#     filepath_tgt = f"{path}/data/raw/mkt_data_tgt_processed.pickle"
#     with open(filepath_tgt, 'rb') as handle:
#         market_data_tgt = pickle.load(handle)
        
#     filepath_acq = f"{path}/data/raw/mkt_data_acq_processed.pickle"
#     with open(filepath_acq, 'rb') as handle:
#         market_data_acq = pickle.load(handle)    

#     return market_data_tgt, market_data_acq    
         
    
    
    
# def print_and_save_string(string, txt_file):
#     print(string)
#     _ = txt_file.write(string+"\n")
#     return

def _replace_None_by_nan(ser):
    """
    if ser is all None, replace None with nans.
    """
    if ser.isna().all():
        return pd.Series(np.nan, index = ser.index)
    return ser

def isclose_any_col_df_to_ser(df, ser, thres_percent = .05, thres_abs = 2., logic = 'or'):
    """
    for each row, determine whether any column in <df> is close to the value in <ser>.
    Closedness is measured by:
    - percentage diff lower than <thres_percent>
    - and/or absolute diff lower than <thres_abs>.
    And/or is indicated by <logic>
    
    
    Parameters:
    ---------------------------
    df: DataFrame
    
    ser: Series
    
    thres_percent: float
    
    thres_abs: float
    
    logic: 'and' / 'or'
    """
    diff = abs(df.sub(ser, axis=0))
    TF_abs = diff.le(thres_abs)
    
    diff_percent = diff.div(abs(ser), axis=0)
    TF_percent = diff_percent.le(thres_percent)
    
    if logic == 'or':
        return (TF_percent|TF_abs).any(axis=1)
    else:
        return (TF_percent&TF_abs).any(axis=1)    
    
def isclose_any_col_df_to_df(df_left, df_right, thres_percent = .05, thres_abs = 2., logic = 'or'):
    result = pd.Series(False, index=df_left.index)
    for col in df_right.columns:
        result = result | isclose_any_col_df_to_ser(df_left, df_right[col], thres_percent = thres_percent, thres_abs = thres_abs)
    return result
    

    
