#####################
## CRSP helpers
#####################

import pandas as pd
import numpy as np
from .utils import compound_daily_return_to_other_freq, to_monthly_period_index, _replace_None_by_nan
from sklearn.utils import _is_arraylike_not_scalar

"""
checked functions

# get stock/fund identifier
def get_stocknames_CRSP(id_no, id_type='permno', date = None, 
                              cols=['permno', 'permco', 'ticker', 'comnam', 'namedt', 'nameenddt', 'cusip', 'ncusip'], 
                              db=None):
                              
def get_fundnames_CRSP(id_no, id_type='crsp_fundno', date = None, 
                              cols=['crsp_fundno', 'ticker', 'fund_name',  'et_flag', 'index_fund_flag', 'm_fund', 'first_offer_dt'], 
                              db=None):
                              
def _convert_stocknames_to_permno(stocknames, return_names=False):                              
def _convert_fundnames_to_crsp_fundno(fundnames, return_names=False):

def get_stock_permno_CRSP(id_no, id_type='ticker', date = None, return_names=False, db=None):
def get_fund_crsp_fundno_CRSP(id_no, id_type='ticker', date = None, return_names=False, db=None):

def get_stock_permno_by_ticker_and_cusip_CRSP(ticker, cusip, date=None, return_names=False, db=None):

# market data
## total market index
def get_total_mkt_return_daily_CRSP(start_date = '1900-01-01', end_date = '2030-01-01', db=None):
def get_total_mkt_return_monthly_CRSP(start_month = '1900-01', end_month = '2030-01', db=None):
def get_total_mkt_return_weekly_CRSP(start_date = '1900-01-01', end_date = '2030-01-01', db=None):


## funds
def get_fund_daily_return_CRSP(id_no, id_type='crsp_fundno', start_date='1900-01-01', end_date='2030-01-01', db=None):
def get_fund_monthly_return_CRSP(id_no, id_type='crsp_fundno', start_month='1900-01', end_month='2030-01', db=None):

"""







################################
## stock & fund identifier
################################

# checked
def get_stocknames_CRSP(id_no, id_type='permno', date = None, 
                              cols=['permno', 'permco', 'ticker', 'comnam', 'namedt', 'nameenddt', 'cusip', 'ncusip'], 
                              db=None):
    """
    get (raw) stocknames file of a stock from the `crsp_m_stock.stocknames`, by its permno / ticker / cusip.
    
    all the available columns are 
    ['permno', 'namedt', 'nameenddt', 'shrcd', 'exchcd', 'siccd', 'ncusip',
       'ticker', 'comnam', 'shrcls', 'permco', 'hexcd', 'cusip', 'st_date',
       'end_date', 'namedum']
    
    can add the filter that the company is listed on a given date.
       
    Parameters:
    -----------------------------
    id_no: supports permno / ticker / cusip.
    
    id_type: {'permno', 'ticker', 'cusip'}
    
    date:
        The stock is listed on this date.
    cols: list
        
    db: 
    
    Returns:
    -----------------------------
    DataFrame
    
    """
    command = f"select {', '.join(cols)} from crsp_m_stock.stocknames where "
    
    if id_type == 'permno':
        command += f" permno = {id_no}"
    elif id_type == 'ticker':
        command += f" ticker = '{id_no}'"
    elif id_type == 'cusip':
        command += f" (substring(cusip, 1, {len(id_no)}) = '{id_no}' or substring(ncusip, 1, {len(id_no)}) = '{id_no}')"
    else:
        raise Exception("Unrecognized identifier!")
        # print('')
        # return None
    
    if pd.notna(date):
        command += f" and namedt <= '{str(date)}' and nameenddt >= '{str(date)}'"
#     return command    
    return db.raw_sql(command)

# checked
def get_fundnames_CRSP(id_no, id_type='crsp_fundno', date = None, 
                              cols=['crsp_fundno', 'ticker', 'fund_name',  'et_flag', 'index_fund_flag', 'm_fund', 'first_offer_dt'], 
                              db=None):
    """
    get fundnames file of a mutual fund from the `crsp_q_mutualfunds.fund_names`, by its ticker / crsp_fundno.
    
    all the available columns are 
    ['cusip8', 'crsp_fundno', 'chgdt', 'chgenddt', 'crsp_portno',
       'crsp_cl_grp', 'fund_name', 'ticker', 'ncusip', 'mgmt_name', 'mgmt_cd',
       'mgr_name', 'mgr_dt', 'adv_name', 'open_to_inv', 'retail_fund',
       'inst_fund', 'm_fund', 'index_fund_flag', 'vau_fund', 'et_flag',
       'delist_cd', 'header', 'first_offer_dt', 'end_dt', 'dead_flag',
       'merge_fundno']
    
    can add the filter that the fund is listed on a given date.
       
    Parameters:
    -----------------------------
    id_no: supports ticker / crsp_fundno.
    
    id_type: {'ticker', 'crsp_fundno'}
    
    date:
        The fund is listed on this date.
    cols: list
        
    db: 
    
    Returns:
    -----------------------------
    DataFrame
    
    """
    command = f"select {', '.join(cols)} from crsp_q_mutualfunds.fund_names where {id_type} = "
    if id_type == 'crsp_fundno':
        command += f" {id_no}"
    elif id_type == 'ticker':
        command += f" '{id_no}'"
    else:
        raise Exception("Unrecognized identifier!")
        # print('Unsupported identifier.')
        # return None
    
    if pd.notna(date):
        command += f" and chgdt <= '{str(date)}' and chgenddt >= '{str(date)}'"
    return db.raw_sql(command)



# checked
def _convert_stocknames_to_permno(stocknames, return_names=False):
    """
    extract permno and stock information from a stocknames file.
    
    Pay attention to 'Multiple permnos'. 
    If there are multiple permnos, output this string if return a value, or the string appears in comnam if return a series.
    
    Parameters:
    ------------------------------------
    stocknames:
    
    return_names: boolean, default False
        if True return a series of ['permno', 'ticker', 'cusip', 'comnam'], otherwise just the permno number
    
    Returns:
    ------------------------------------
    float or Series
    
    """
    index = ['permno', 'ticker', 'cusip', 'comnam']
    if len(stocknames.permno.unique()) == 1:  # unique permno
        if return_names:
            ser = stocknames.iloc[-1][index]
            ser.name = None
            return ser
        return stocknames.iloc[-1]['permno']
    # multiple or no permno
    if len(stocknames.permno.unique()) > 1:
        return pd.Series(['Multiple permnos', np.nan, np.nan, np.nan], index=index) if return_names else 'Multiple permnos'
    return pd.Series(dtype=object, index=index) if return_names else np.nan
    
# checked
def _convert_fundnames_to_crsp_fundno(fundnames, return_names=False):
    """
    extract crsp_fundno and fund information from a fundnames file.
    
    Pay attention to 'Multiple permnos'. 
    Output this string if return a value, and the string appears in comnam if return a series.
    
    Parameters:
    ------------------------------------
    fundnames:
    
    return_names: boolean, default False
        if True return a series of ['crsp_fundno', 'ticker', 'fund_name',  'et_flag', 'index_fund_flag', 'm_fund', 'first_offer_dt'], otherwise just the crsp_fundno number
    
    Returns:
    ------------------------------------
    float or Series
    """
    
    index = ['crsp_fundno', 'ticker', 'fund_name',  'et_flag', 'index_fund_flag', 'm_fund', 'first_offer_dt']
    if len(fundnames.crsp_fundno.unique()) == 1:  # unique permno
        if return_names:
            ser = fundnames.iloc[-1][index]
            ser.name = None
            return ser
        return fundnames.iloc[-1]['crsp_fundno']
    # multiple or no permno
    if len(fundnames.crsp_fundno.unique()) > 1:
        return pd.Series(['Multiple fundnos', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], index=index) if return_names else 'Multiple fundnos'
    return pd.Series(dtype=object, index=index) if return_names else np.nan


    
# checked    
def get_stock_permno_CRSP(id_no, id_type='ticker', date = None, return_names=False, db=None):
    """
    get the permno of a stock by its ticker or cusip.
    
    Parameters:
    ------------------------------------
    id_no: support ticker / cusip
    
    id_type: {'ticker', 'cusip'}
    
    date: 
        The stock is listed on this date.
    return_names: boolean, default False
        if True return a series of ['permno', 'ticker', 'cusip', 'comnam'], otherwise just the permno number
    db:
    
    
    Returns:
    ------------------------------------
    float or Series
    """
    stocknames = get_stocknames_CRSP(id_no, id_type=id_type, date = date, db=db)
    return _convert_stocknames_to_permno(stocknames, return_names)


def get_fund_crsp_fundno_CRSP(id_no, id_type='ticker', date = None, return_names=False, db=None):
    """
    get the crsp_fundno of a fund by its ticker.
    
    Parameters:
    ------------------------------------
    id_no: support ticker
    
    id_type: {'ticker'}
    
    date: 
        The fund is listed on this date.
    return_names: boolean, default False
        if True return a series of ['crsp_fundno', 'ticker', 'fund_name',  'et_flag', 'index_fund_flag', 'm_fund', 'first_offer_dt'], otherwise just the crsp_fundno number
    db:
    
    
    Returns:
    ------------------------------------
    float or Series
    """
    fundnames = get_fundnames_CRSP(id_no, id_type=id_type, date = date, db=db)
    return _convert_fundnames_to_crsp_fundno(fundnames, return_names)




# checked
def get_stock_permno_by_ticker_and_cusip_CRSP(ticker, cusip, date=None, return_names=False, db=None):
    """
    get the permno of a stock by its ticker, and then cusip.
    
    Returns:
    ------------------------------------
    permno if return_id is False, else a series of (permno, ticker, cusip, comnam)
    """
    def check_permno_notnull(permno):
        p = permno if not _is_arraylike_not_scalar(permno) else permno.permno
        if isinstance(p, float) and pd.notna(p):
            return True
        return False
    
    # search by ticker first
    if pd.notna(ticker):
        permno = get_stock_permno_CRSP(ticker, 'ticker', date, return_names, db)
        if check_permno_notnull(permno):
            return permno
    # by cusip then
    if pd.notna(cusip):
        permno = get_stock_permno_CRSP(cusip, 'cusip', date, return_names, db)
        if check_permno_notnull(permno):
            return permno

    # cannot match by either ticker or cusip
    index = ['permno', 'ticker', 'cusip', 'comnam']
    return pd.Series(dtype=object, index=index) if return_names else np.nan






###############################
## Market data: delisting
###############################

# checked
def get_delisting_information(permno, 
                              cols=['dlstcd', 'dlstdt', 'dlpdt', 'dlamt', 'dlret'],
                              db=None):
    """
    search for delisting information in crsp_m_stock.dsedelist database, by permno of the stock.
    
    All the columns are 
    ['permno', 'last_trade_date', 'delist_code', 'nwperm', 'nwcomp',
       'nextdt', 'delist_amount', 'dlretx', 'dlprc', 'delist_date',
       'delist_return', 'permco', 'compno', 'issuno', 'hexcd', 'hsiccd',
       'cusip', 'acperm', 'accomp']
       
    Return a series.
       
    NA is allowed.
    
    Parameters:
    ----------------------------------------
    permno: a single permno number. can be NA.
    
    cols: list of column names
    
    db:
    
    Returns:
    ----------------------------------------
    Series.
    """
    cols_lst = ['dlstcd', 'dlstdt', 'dlpdt', 'dlamt', 'dlret']
    name_lst = ['delist_code', 'last_trade_date', 'delist_date', 'delist_amount', 'delist_return']
    rename_dict = dict(zip(cols_lst, name_lst))

    if pd.isna(permno):  # missing
        return pd.Series(dtype=object, index=cols).rename(index=rename_dict)
    # not missing permno
    command = f"select {', '.join(cols)} from crsp_m_stock.dsedelist where permno = {permno}"   
    df_delist = db.raw_sql(command)
#     return df_delist
    if len(df_delist)!= 1:
        return pd.Series(dtype=object, index=cols).rename(index=rename_dict)
    df_delist = df_delist.iloc[-1]
    df_delist.name = None
    return df_delist.rename(index=rename_dict)





###############################
## Market data: fund
###############################
# checked
def get_fund_daily_return_CRSP(id_no, id_type='crsp_fundno', start_date='1900-01-01', end_date='2030-01-01', db=None):
    """
    get the daily return series of a mutual fund, by its permno or ticker.
    
    Parameters:
    -----------------------------
    id_no: supports permno or ticker.
    
    id_type: {'crsp_fundno', 'ticker'}, default 'crsp_fundno'
    
    start_date: datetime.date, or string. default '1900-01-01'.
    
    end_date: datetime.date, or string. default '2030-01-01'. 

    db: 
    
    Returns:
    -----------------------------
    Series
    """
    # convert ticker to crsp_fundno
    if id_type == 'ticker':
        date = start_date if start_date != '1900-01-01' else None
        crsp_fundno = get_fund_crsp_fundno_CRSP(id_no, id_type='ticker', date=date, db=db)
        if pd.isna(crsp_fundno):
            print("cannot find crsp_fundno.")
            return pd.DataFrame(dtype=float, columns=['dret'])
    elif id_type == 'crsp_fundno':
        crsp_fundno = id_no
    else:
        raise Exception("Unrecognized identifier!")
#         print('Unrecognized identifier.')
#         return pd.DataFrame(dtype=float, columns=['dret'])
    
    #
    command = f"select caldt, dret from crsp_q_mutualfunds.daily_returns where crsp_fundno = {crsp_fundno} and caldt >= '{start_date}' and caldt <='{end_date}'"
    
    #
    market_data = db.raw_sql(command).dropna().rename(columns={'caldt':'date'}).set_index('date').sort_values('date').dret#.set_index('date').sort_index() 
    return market_data

# checked
def _convert_start_end_month_to_date(start_month, end_month):
    """
    e.g. '1900-01', '2030-01' -> '1900-01-01', '2030-02-01'
    Used to query monthly data in crsp where date is the end of each month.
    """
    return str(start_month) + '-01', str(pd.Period(end_month, 'M') + 1) + '-01'

# checked
def get_fund_monthly_return_CRSP(id_no, id_type='crsp_fundno', start_month='1900-01', end_month='2030-01', db=None):
    """
    get the monthly return series of a mutual fund, by its fundno or ticker.
    
    Parameters:
    -----------------------------
    id_no: supports permno or ticker.
    
    id_type: {'crsp_fundno', 'ticker'}, default 'crsp_fundno'
    
    start_month: pd.Period, or string. default '1900-01'.
    
    end_month: pd.Period, or string. default '2030-01'. 

    db: 
    
    Returns:
    -----------------------------
    Series
    """
    start, end = pd.Period(start_month, 'M'), pd.Period(end_month, 'M')
    
    # convert ticker to crsp_fundno
    if id_type == 'ticker':
        date = str(start+1)+'-01' if start_month != '1900-01' else None
        crsp_fundno = get_fund_crsp_fundno_CRSP(id_no, id_type='ticker', date=date, db=db)
        if pd.isna(crsp_fundno):
            print("cannot find crsp_fundno.")
            return pd.DataFrame(dtype=float, columns=['mret'])
    elif id_type == 'crsp_fundno':
        crsp_fundno = id_no
    else:
        raise Exception("Unrecognized identifier!")
        # print('Unrecognized identifier.')
        # return pd.DataFrame(dtype=float, columns=['mret'])
    
    #    
    start_date, end_date = _convert_start_end_month_to_date(start, end)      #str(start)+'-01', str(end+1)+'-01'
    
    #
    command = f"select caldt, mret from crsp_q_mutualfunds.monthly_returns where crsp_fundno = {crsp_fundno} and caldt >= '{start_date}' and caldt <='{end_date}'"
    
    #
    market_data = db.raw_sql(command).dropna().sort_values('caldt')#.set_index('date').sort_index() 
    return to_monthly_period_index(market_data, 'caldt').mret






###############################
## Market data: stock
###############################
# def _replace_None_by_nan(ser):
#     """
#     if ser is all None, replace None with nans.
#     """
#     if ser.isna().all():
#         return pd.Series(np.nan, index = ser.index)
#     return ser

def clean_prc_ret(values):
    """
    clean price and returns:
    - For all None prices and returns in the period, replace None by nans.
    - negative price: take absolute value.  
    """
    # missing price
    if 'prc' in values.columns:
        values.prc = abs(_replace_None_by_nan(values.prc))
    if 'ret' in values.columns:
        values.ret = _replace_None_by_nan(values.ret)           
    return values


# checked
def get_stock_market_data_daily_CRSP(id_no, id_type='permno', start_date='1900-01-01', end_date='2030-01-01', 
                                cols=['permno', 'prc', 'ret', 'vol', 'shrout', 'cfacpr', 'cfacshr'], 
                                db=None):
    """
    get the daily time series of equity market data from the `crsp_m_stock.dsf` database, by its permno or ticker.
    
    all the available columns are 
    ['cusip', 'permno', 'permco', 'issuno', 'hexcd', 'hsiccd', 'bidlo',
       'askhi', 'prc', 'vol', 'ret', 'bid', 'ask', 'shrout', 'cfacpr',
       'cfacshr', 'openprc', 'numtrd', 'retx']
    
    Parameters:
    -----------------------------
    id_no: supports permno or ticker.
    
    id_type: {'permno', 'ticker'}, default 'permno'
    
    start_date: datetime.date, or string. default '1900-01-01'.
    
    end_date: datetime.date, or string. default '2030-01-01'. 
    
    cols: list
        ['*'] or any combination. Would automatically add 'date' if it isn't included.
    db: 
    
    Returns:
    -----------------------------
    DataFrame or Series, depending on whether there is only one column.
    
    """

    # convert ticker to permno
    if pd.isna(id_no):
        return np.nan
    if id_type == 'ticker':
        date = start_date if start_date != '1900-01-01' else None
        permno = get_stock_permno_CRSP(id_no, id_type='ticker', date=date, db=db)
        if pd.isna(permno):
            print("cannot find permno.")
            return pd.DataFrame(dtype=float, columns=cols)
    elif id_type == 'permno':
        permno = id_no
    else:
        print('Unrecognized identifier.')
        return pd.DataFrame(dtype=float, columns=cols)
    
    # add 'date'
    if 'date' not in cols and '*' not in cols:
        cols += ['date']
    #    
    start_date, end_date = str(start_date), str(end_date)
    
    #
    command = f"select {', '.join(cols)} from crsp_m_stock.dsf where permno = " + \
    f"{str(permno)} and date >= '{start_date}' and date <= '{end_date}'"
    
    #
    market_data = clean_prc_ret(db.raw_sql(command).set_index('date').sort_index())
    return market_data if len(market_data.columns) > 1 else market_data.iloc[:, 0]



#######################
## Market Data: index
#######################
# checked




def get_total_mkt_return_daily_CRSP(start_date = '1900-01-01', end_date = '2030-01-01', db=None):
    """
    return the CRSP total market daily return (w/ dividends) series.
    
    Parameters:
    --------------------------
    start_date:
    
    end_date:
    
    db: 
    

    Returns:
    --------------------------
    Series
    """
    command = f"select date, vwretd from crsp_m_stock.dsi where date >= '{str(start_date)}' and date <= '{str(end_date)}'"
    df = db.raw_sql(command).dropna().set_index('date')
    return df.vwretd


    
# checked
def get_total_mkt_return_monthly_CRSP(start_month = '1900-01', end_month = '2030-01', db=None):
    """
    return the CRSP total market monthly return (w/ dividends) series.
    
    Parameters:
    --------------------------
    start_date:
    
    end_date:
    
    db: 
    

    Returns:
    --------------------------
    Series
    """
    start_date, end_date = _convert_start_end_month_to_date(start_month, end_month)
    command = f"select date, vwretd from crsp_m_stock.msi where date >= '{start_date}' and date <= '{end_date}'"
    df = db.raw_sql(command).dropna()
    return to_monthly_period_index(df, 'date').squeeze()

# checked
def get_total_mkt_return_weekly_CRSP(start_date = '1900-01-01', end_date = '2030-01-01', db=None):
    """
    return the CRSP total market weekly return (w/ dividends) series.
    
    Parameters:
    --------------------------
    start_date:
    
    end_date:
    
    db: 
    

    Returns:
    --------------------------
    Series
    """
    ret_ser = get_total_mkt_return_daily_CRSP(start_date, end_date, db)
    return compound_daily_return_to_other_freq(ret_ser, 'W-FRI')
