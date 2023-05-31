######################
## market calendar
######################

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from datetime import timedelta
from .preprocessing import convert_date_str_ser_to_datetime, convert_singe_date_str_to_datetime

"""
all checked

def get_trading_day_range(start_date = '1900-01-01', end_date = '2030-01-01', return_as_series = False):
def get_trading_day_offset(ser, offset):
def get_num_trading_days_between(left_date_ser, right_date_ser):
"""

# checked
def get_trading_day_range(start_date = '1900-01-01', end_date = '2030-01-01', return_as_series = False):
    """
    return the valid trading days (of NYSE) within the date range. 
    return either a Series or an array.
    
    Parameters:
    -----------------------------------------
    start_date: 
    
    end_date: 
    
    return_as_series: boolean
        whether to return as Series. Otherwise return as array.
    
    Returns:
    -----------------------------------------
    Series
    """
    
    nyse = mcal.get_calendar('NYSE')
    days = nyse.valid_days(start_date, end_date).date
    if return_as_series:
        return pd.Series(days)
    return days


# checked
def get_trading_day_offset(ser, offset):
    """
    get the trading days with a certain offset from the input dates.
    offset=0 is the nearest next trading day, including the day itself. 
    
    Parameters:
    -----------------------------------------
    ser: Series of, or a single `datetime.date` or string `YY-mm-dd`
        NA allowed
    offset: int
        number of days in the offset.
    
    
    Returns:
    -----------------------------------------
    a series of, or a single `datetime.date`
    
    
    Examples:
    ----------------------------------------------
    input: ser = datetime.date(2022, 9, 4) or datetime.date(2022, 9, 5) or datetime.date(2022, 9, 6), offset = 0
    output: datetime.date(2022, 9, 6).
    """
    if isinstance(ser, pd.Series):
        if ser.empty:   # edge case
            return pd.Series(dtype=object)
        if ser.isna().all():   # all missing
            return pd.Series(dtype=object, index=ser.index)

        # change to series of `datetime.date` objects
        ser_new = convert_date_str_ser_to_datetime(ser.dropna())         #pd.to_datetime().dt.date
        # get date range
        start, end = ser_new.min(), ser_new.max()
    else: # supposedly ser is a single datetime-like object
        if pd.isna(ser):     # edge case
            return np.nan
        # convert to `datetime.date`
        ser_new = convert_singe_date_str_to_datetime(ser)  #pd.to_datetime(ser).date()
        # get date range
        start, end = ser_new, ser_new
    
    # some tolerance    
    start -= timedelta(days=2*abs(offset) + 5)
    end += timedelta(days=2*abs(offset) + 5)
#     return start, end
    
    # get `NYSE` calendar
    trading_days = get_trading_day_range(start, end)

    # find index in the series of all trading days
    ind_arr = np.searchsorted(trading_days, ser_new)
    days_with_offset = trading_days[ind_arr + offset]
    
    if isinstance(ser, pd.Series):  # fill back those missing numbers
        return pd.Series(days_with_offset, index=ser_new.index).reindex(ser.index)
    return days_with_offset

# checked
def get_num_trading_days_between(left_date_ser, right_date_ser):
    """
    calculate the number of trading days between two series or two single dates.
    NA is allowed.
    """
    
    if isinstance(left_date_ser, pd.Series) and isinstance(right_date_ser, pd.Series):
        # dropna
        dates_combined = pd.concat([convert_date_str_ser_to_datetime(left_date_ser), convert_date_str_ser_to_datetime(right_date_ser)], axis=1).dropna()
        if dates_combined.empty:
            return pd.Series(np.nan, index=left_date_ser.index)
        # get the ser without na
        left, right = dates_combined.iloc[:, 0], dates_combined.iloc[:, 1]
        # start and end date for pulling trading days
        start, end = left.min(), right.max()
    else:    # two single date-likes
        if pd.isna(left_date_ser) or pd.isna(right_date_ser):
            return np.nan
        # get the ser without na
        left, right = convert_singe_date_str_to_datetime(left_date_ser), convert_singe_date_str_to_datetime(right_date_ser) #pd.to_datetime(left_date_ser).date(), pd.to_datetime(right_date_ser).date()
        # start and end date for pulling trading days
        start, end = left, right
    
    start -= timedelta(5)
    end += timedelta(5)
        
    trading_days = get_trading_day_range(start, end)
    # difference of indice of left and right
    days_diff = np.searchsorted(trading_days, right) - np.searchsorted(trading_days, left)
    
    if isinstance(left_date_ser, pd.Series):
        return pd.Series(days_diff, index=left.index).reindex(left_date_ser.index)
    return days_diff
