import pandas as pd
import numpy as np
from os.path import expanduser
from tqdm import tqdm
import re

import locale
from locale import atof, setlocale

setlocale(locale.LC_ALL, 'en_US')

from .utils import load_CPI_index

def print_shape(df):
    """
    Print the shape of a dataset.
    """
    print(f'The dataset is of size {df.shape}.')

def extract_colnames_from_report_file(filepath):
    """
    extract the column names of the dataset from the report file, as a list of lowercase strings.
    Each line after 'Custom Report Contents:\n' is a column name.
    """
    # open file as a list of strings of each line 
    lines = open(expanduser(filepath), 'r').readlines()

    # only need lines after "Custom Report Contents:\n"
    ind = 0
    while not lines[ind].startswith('Custom Report Contents:'):
        ind += 1
    # discard newline character. return lower cases.
    return list(map(lambda x: x.strip().lower(), lines[ind+1:]))



# checked
def convert_date_str_ser_to_datetime(ser):
    """
    convert a series of date-like strings to datetime.date objects.
    NA is allowed.
    
    Parameters:
    ---------------------
    ser: Series
        a series of date-like strings.
        
    Returns:
    ---------------------
    a series of datetime.date.
    """
    return pd.to_datetime(ser).dt.date

# checked
def convert_singe_date_str_to_datetime(date):
    """
    convert a single date-like string to datetime.date object.
    
    Parameters:
    ---------------------
    date:
        one date-like strings.
        
    Returns:
    ---------------------
    datetime.date.
    """
    return pd.to_datetime(date).date()


# checked
def convert_num_str_ser_to_float(ser):
    """
    convert a series of numeric-like strings (e.g. '1,000') to floats. NA is allowed in <ser>.
    
    Parameters:
    ---------------------
    ser: Series
        a series of numeric-like strings.
        
    Returns:
    ---------------------
    a series of floats.
    """
    return ser.map(atof, na_action='ignore')



##############################
## correct dataset
##############################

def get_delete_index(df):
    """
    get the index of wrong data entries to be deleted manually.
    """
    index_del = [274214020, 227680020, 243448020] + [25826020, 103764020]
    return df.index.intersection(index_del)



def correct_consid(dff):
    """
    fix data error manually. <dff> should contain columns <consido> and <consid>.
    """ 
    # create copy
    df = dff.copy()
    
    # 
    if 1040793020 in df.index:
        df.loc[1040793020, ['consido', 'consid']] = ['Cash Only', '$7 cash/sh com']
    
    # fix `consid` manually
    index = [1018291020,
 1752953020,
 1871799020,
 2474162020,
 2684890020,
 2770926020,
 2942217020,
 2634022020,
 3037577020,
 3090980020,
 3098396020,
 3121972020,
 3238285020,
 3416138020,
 3453292020,
 3473708020,
 3664650020,
 3711080020,
 3700733020,
 3728695020,
 3761157020,
 3306644020,
 3846599020,
1889692020,
        1610526020,
        1425273020,
        1594235020,
        2724775020,
        2530559020,
        2309478020,
        3100226020,
        3731057020,
        1181437020,
        1581335020,
        2333741020,
        1074016020,
        2952899020,
        2732707020,
        3213676020,
        2701755020,
        2229022020,
        2291770020,
        2919836020,
        934761020,
            569766020,
            170805020,
            938046020,
            166546020,
            197336020,
            238913020,
            325340020,
            330520020,
            338719020,
            344267020,
            389787020,
            414062020,
            405440020,
            420869020,
            436959020,
            692819020,
            679983020,
            616158020,
            565173020,
            553658020,
            542491020,
            490778020,
            483432020,
            476416020,
            2382542020] +\
    [125504020, 146736020, 147638020, 163620020, 164842020, 174041020, 179091020, 180234020, 754797020, 806241020, 806318020, 1259809020,
    911864020, 1460560020, 2047839020, 2118413020, 2737006020, 3132719020,
    3944361020, 3992461020,
    24445020, 25826020, 1684872020, 3859429020, 3670772020, 3002849020, 2719314020, 2183939020,
    976280020, 830630020, 858307020,
    3953827020, 3763912020, 3617548020, 3066310020, 1616418020,
    379449020, 25022020]
    
    correction_consid = ['.105 shs com/sh com',
 '$6 cash and $2.5 com/sh com',
 '$40 cash plus $35 com/sh com',
 '$25 cash plus 0.6531 shs com/sh com',
 '$1.26 cash plus 0.5846 shs com/sh com',
 '$62.93 cash plus 0.6019 shs com/sh com',
 '$6.45 cash plus 0.75 shs com/sh com',
 '$34.1 cash and $27.9 com/sh com',
 '$1.91 cash plus 0.14864 shs com/sh com',
 '$95.63 cash/sh com',
 '$35 cash plus 0.23 shs com/sh com',
 '$3.78 cash plus 0.7884 shs com/sh com',
 '$2.3 cash plus 0.738 shs com/sh com',
 '$6.28 cash plus 0.528 shs com/sh com',
 '$6.8 cash and .7275 shs com/sh com',
 '$2.5 cash plus 0.8 shs com/sh com',
 '$26.790 cash plus 0.0776 shs com/sh com',
 '$1.39 cash plus 0.42 shs com/sh com',
 '$41.75 cash plus 0.907 shs com/sh com',
 '$2.89 cash plus 1.408 shs com/sh com',
 '0.4 shs com/sh com',
 '$1.46 cash and .297 shs com/sh com',
 '$133 cash plus .4506 shs com/sh com',
             '$12.5 cash/sh com',
             '$1.50 cash plus $13.875 com/sh com',
             '$17.5 cash/sh com',
             '$11.375 cash plus .2917 shs com/sh com',
             '$6.25 cash plus 0.3521 shs com/sh com',
             '$83 cash plus $49.5 shs com/sh com',
             '$5 cash plus 0.588 shs com/sh com',
             '1.123 shs com/sh com',
             '.124 shs com/sh com',
             '1.14 shs com/sh com',
             '0.89 shs com/sh com',
             '0.46 shs com/sh com',
             '1.7 shs com/sh com',
             '1.63 shs com/sh com',
             '0.73 shs com/sh com',
             '0.27 shs com/sh com',
             '0.2413 shs com/sh com',
             '0.93 shs com/sh com',
             '1.02 shs com/sh com',
                 '4.4 shs com/sh com',
                 '.4626 shs com/sh com',
                 '$65 cash/sh com',
                 '1.752 shs com/sh com',
                 '.322 shs com/sh com',
                 '1.12 shs com/sh com',
                 '.384 shs com/sh com',
                  '$30.25 com/sh com',
                 '$17.5 com/sh com',
                 '$20 cash plus .45 shs com/sh com',
                 '2.13 shs com/sh com',
                 '1 shs com/sh com',
                 '.2 shs com/sh com',
                 '.6803 shs com/sh com',
                 '.438 shs com/sh com',
                 '$5.875 cash plus $6.125 com/sh com',
                 '$16 cash/sh com',
                 '$28.85 cash/sh com',
                 '.55 shs com/sh com',
                 '.65 shs com/sh com',
                 '$12.75 com/sh com',
                 '.933 shs com/sh com',
                 '$8 cash and $32 com/sh com',
                 '1.05 shs com/sh com',
                 '.53 shs com/sh com',
                 '.845 shs com/sh com',
                 '$21.75 cash/sh com'] + \
    ['$17 cash/sh com', '$17 cash plus 1.07195 shs com/sh com', '$45 cash/sh com', '1.3889 shs com/sh com', '$16.2 cash/sh com', '$9.25 cash/sh com', '.2 shs com/sh com', '1 shs com/sh com',
'$2.23 cash plus 1 shs com/sh com', '.34 shs com/sh com', '1.347 shs com/sh com', '$2.21 cash/sh com',
    "1.77 shs com/sh com", "$.39 cash plus .791 com shs/sh com", "$3.45 cash/sh com", "$.055 cash/sh com", "$2.35 cash/sh com", "1.32 shs com/sh com",
    "$39 cash/sh com", "$58 cash/sh com",
    "$27.25 cash/sh com", "$21 cash/sh com", "$43.9 cash/sh com", "$3.5 cash plus .0406 shs com/sh com", "$58.5 cash/sh com", "$22.34 cash/sh com", "$44.25 cash/sh com", "$4.5 cash plus 1.0856 sh ord/sh com",
    "$21 cash/sh com", "1.83 shs com/sh com", ".7933 shs com/sh com",
    "$10.5 cash plus .0483 shs com/sh com", ".1562 shs com/sh com", "$4.24 cash plus 0.592 shs com/com", "$21.75 cash plus .2675 shs com/sh com", "$3.5 cash plus .552 shs com/sh com",
    ".67 shs com/sh com", ".6 shs com/sh com"]
    
    # create the mapping series
    correction_ser = pd.Series(correction_consid, index=index, name='consid')
    
    # find the indices also in <df>
    index_intersection = df.index.intersection(index)
    # do the correction
    df.consid[index_intersection] = correction_ser[index_intersection]
    return df


###############################
## process market data
###############################

def add_delisting_prc_ret(mkt_data_tgt_input, df):
    """
    add the delisting amount and return on the delisting day for deals that are:
    - completed
    - CRSP delisting code is due to MA reason
    - delisting return is not missing
    """
    mkt_data_tgt = mkt_data_tgt_input.copy()
    # filter deals
    ind_set = df.index[df.statc.eq("C") & df.delist_code.between(200, 300, "left") & df.delist_return.notna()]

    for i in tqdm(ind_set):
        mkt_data_df = mkt_data_tgt[i]
        if len(mkt_data_df) >= 1 and mkt_data_df.index[-1]==df.last_trade_date.loc[i]:
            delist_date = df.delist_date[i]
            # add a new row
            mkt_data_df.loc[delist_date] = np.nan
            # copy these numbers from last row
            cols = ['permno', 'shrout', 'cfacpr', 'cfacshr']
            mkt_data_df.iloc[-1][cols] = mkt_data_df.iloc[-2][cols]
            # add the delisting amount and return
            mkt_data_df.iloc[-1][['prc', 'ret']]=[df.delist_amount[i], df.delist_return[i]]   
    return mkt_data_tgt


def fill_na_prc_ret(mkt_data_df):
    """
    fill missing price and return, taking adjusting factor into consideration.
    """
    if 'prc' in mkt_data_df.columns and mkt_data_df.prc.notna().all():  # no missing price, (thus no missing ret hopefully)
        return mkt_data_df
    elif 'prc' in mkt_data_df.columns:
        # have missing price
        if "cfacpr" not in mkt_data_df.columns or len(np.unique(mkt_data_df.cfacpr)) == 1: # no need to consider adjust factor
            mkt_data_df.prc = mkt_data_df.prc.fillna(method='ffill')
        elif len(np.unique(mkt_data_df.cfacpr)) >= 2: # need to adjust
            ind_set = np.where(mkt_data_df.prc.isna())[0]
            if ind_set[0]==0:
                ind_set = ind_set[1:]
            for i in ind_set:
                prev_prc, prev_cfacpr = mkt_data_df.iloc[i-1][['prc', 'cfacpr']]
                mkt_data_df.prc.iloc[i] = prev_prc/prev_cfacpr*mkt_data_df.iloc[i].cfacpr
    # price cleaning completed
    if "ret" not in mkt_data_df.columns: # no ret column
        return mkt_data_df
    elif "prc" not in mkt_data_df.columns:  # have ret but no prc.
        mkt_data_df.ret = mkt_data_df.ret.fillna(0.)
        return mkt_data_df
    # have missing ret and prc
    if "cfacpr" not in mkt_data_df.columns:
        prc_adj = mkt_data_df.prc
    else:
        prc_adj = mkt_data_df.prc.div(mkt_data_df.cfacpr)

    ind_set = np.where(mkt_data_df.ret.isna())[0]
    if ind_set[0]==0:
        ind_set = ind_set[1:]
    if len(ind_set)>=1:
        prc_adj_prev = prc_adj.iloc[ind_set-1].values
        mkt_data_df.ret.iloc[ind_set] = prc_adj.iloc[ind_set].sub(prc_adj_prev).div(prc_adj_prev)
    return mkt_data_df

def calculate_mktcap(mkt_data_df):
    """
    given market data, calculate mktcap, and shifted (previous day) mktcap
    """
    mkt_data_df['mktcap'] = mkt_data_df.prc.mul(mkt_data_df.shrout)
    mkt_data_df['mktcap_prev'] = mkt_data_df.mktcap.shift()
    return mkt_data_df



def adjust_price(mkt_data_df, cfacpr):
    """
    adjust price by accumulative factor.
    """
    if pd.notna(cfacpr):
        mkt_data_df['prc_adj'] = mkt_data_df.prc.div(mkt_data_df.cfacpr)*cfacpr
    else:
        mkt_data_df['prc_adj'] = np.nan
    mkt_data_df['prc_adj_prev'] = mkt_data_df.prc_adj.shift()
    return mkt_data_df 

def adjust_price_for_mkt_data_ser(mkt_data, date_ser):
    ind_set = mkt_data.dropna().index
    for i in tqdm(ind_set):
        date = date_ser[i]
        if date in mkt_data[i].index:
            cfacpr = mkt_data[i].loc[date, 'cfacpr']
        else:
            cfacpr = np.nan
        mkt_data[i] = adjust_price(mkt_data[i], cfacpr)
    return mkt_data


def compute_trailing_20_ave_prc_adj(mkt_data_df):
    """
    given market data, calculate trailing 20 days average of adjusted price
    """
    mkt_data_df['prc_adj_trail_20_ave'] = mkt_data_df.prc_adj.rolling(20, min_periods=1).mean()
    return mkt_data_df




def extract_col_at_date_from_mkt_data_ser(mkt_data_ser, date_ser, col):
    """
    extract the value of a column at a certain date, for a series of market data.
    """
    idx_all = mkt_data_ser.index
    idx = idx_all[mkt_data_ser.notna()]
    res = []
    for i in idx:
        mkt_data, date = mkt_data_ser[i], date_ser[i]
        try:
            res.append(mkt_data.loc[date, col])
        except:
            res.append(np.nan)
    return pd.Series(res, index=idx).reindex(mkt_data_ser.index)

def calculate_deal_price_for_ser(cash_term, stock_term, payment_type, acq_prc):
    # diff cash and stock
    cash_payment = ['Cash', "Common Stock, fixed dollar", "Cash and Common Stock, fixed dollar"]
    stock_payment = ["Common Stock", "Cash and Common Stock"]
    idx_cash = payment_type.isin(cash_payment)
    idx_stock = payment_type.isin(stock_payment)
    #
    pr = pd.Series(np.nan, index=cash_term.index)
    pr[idx_cash] = cash_term[idx_cash] + stock_term[idx_cash]
    pr[idx_stock] = cash_term[idx_stock] + stock_term[idx_stock] * acq_prc[idx_stock] 
    return pr

##############################
##
##############################
def check_list_of_keys_in_list(key_lst, lst):
    """
    check whether any of the key in <key_lst> is contained in the list <lst>.
    """
    for key in key_lst:
        if key in lst:
            return True
    return False

def create_choice(df):
    """
    create the column of whether the deal term consists of a choice. <df> should contain columns <consid> and <synop>
    
    Returns:
    ----------------------------------------
    a Series of 0/1 indicating whether whether the deal term consists of a choice.
    """
    choice_lst = ['choice', 'Choice', 'choose', 'Choose']    # keys of amendment to search in synopsis
    choice_synop = df.synop.map(lambda x: check_list_of_keys_in_list(choice_lst, x))
    choice_consid = df.consid.map(lambda x: check_list_of_keys_in_list(choice_lst, x), na_action='ignore').fillna(False)
    return (choice_consid | choice_synop).astype(int)


def create_amend(df):
    """
    create the column of whether the deal is amended.
    
    Returns:
    ----------------------------------------
    a Series of 0/1 indicating whether the deal is amended or not. 
    """
    return (df.valamend.notna()|df.pr.ne(df.pr_initial)).astype(int)
    # amend_lst = ['sweet', 'amend']    # keys of amendment to search in synopsis
    # amend_lst_more = amend_lst + ['Original', 'original', 'previous', 'Previous']   # keys of amendment to search in consid
    # amend_synop = df.synop.map(lambda x: check_list_of_keys_in_list(amend_lst, x), na_action='ignore').fillna(False)
    # amend_consid = df.consid.map(lambda x: check_list_of_keys_in_list(amend_lst_more, x), na_action='ignore').fillna(False)
    # return (amend_consid | amend_synop).astype(int)  

###################
## competing 
###################

def create_compete_group_no(df):
    """
    create competing deal group number. <df> should include columns <ttic> <cha> <competecode>.
    deals in the same group must have the same target company.
    
    Returns:
    -----------------------------------
    a series with the competing group numbers.
    """
    compete_group_no = pd.Series(np.nan, index=df.index, name='compete_group_no')
    no = 0
    for i in df.index[df.competecode.notna()]:
        if pd.notna(compete_group_no[i]): # has been assigned a group
            continue
        code_lst = [int(code) for code in df.competecode[i].split('\n')]
        # competing deal in the dataset, and the same target
        code_lst = [code for code in code_lst if code in df.index and df.ttic[i]==df.ttic[code]]
        if len(code_lst)==0:
            continue
        code_lst.append(i)   # the compeing group
        if len(compete_group_no[code_lst].value_counts())==0: # all the deals in the group has not been assigned no
            compete_group_no[code_lst] = no
            no += 1
        elif len(compete_group_no[code_lst].value_counts())==1: # some deals in the group has been assigned no
            compete_group_no[code_lst] = compete_group_no[code_lst].dropna().iloc[0]
        else:    # deals in the same group are assigned different group numbers. error
            compete_group_no[code_lst] = -1
    return compete_group_no


def create_compete_status_code_single_group(statc_series, dw_series):
    """
    assign competing deal status codes to one single competing group.
    
    compete status codes:
    - 0: winner in a competition and completes the deal
    - 1: winner in a competition and not completes the deal
    - 2: loser in a competition
    - 3: winner in a competition, still pending
    - 4: still competing
    - 9: error code
    
    Parameters:
    ----------------------------------------
    statc_series: Series
        a series of the status of the deals in a group
    dw_series: Series
        a series of the withdrawl date of the deals in a group
    
    Returns:
    ----------------------------------------
    a series of competing deal status codes in a group.
    """
    
    if 'C' in statc_series.values and statc_series.value_counts().loc['C'] >= 2: # more than one completed deals in a competing group. error
        return pd.Series(9, index=statc_series.index)
    elif 'C' in statc_series.values and statc_series.value_counts().loc['C'] == 1: # just one completed deal in the group
        return statc_series.replace({'C':0,'W':2, 'P':2})
    elif 'P' in statc_series.values and statc_series.value_counts().loc['P'] >= 2: # no completed and more than 2 pending. still compeing
        return statc_series.replace({'P':4,'W':2})
    elif 'P' in statc_series.values and  statc_series.value_counts().loc['P'] == 1: # no completed and only one pending.
        return statc_series.replace({'P':3,'W':2})
    else:       # statc_series.value_counts().loc['P'] == 0: # no completed and no pending, all withdrawn
        result = pd.Series(2, index=statc_series.index)
        result.iloc[np.argmax(dw_series)] = 1    # the last withdrawn deal is the winner of the competition.
        return result
    return pd.Series(9, index=statc_series.index)

def create_compete_status_code(df):
    """
    create competing deal status codes for the dataset. <df> should include columns <compete_group_no> <dw> <statc>.
    
    compete status codes:
    - 0: winner in a competition and completes the deal
    - 1: winner in a competition and not completes the deal
    - 2: loser in a competition
    - 3: winner in a competition, still pending
    - 4: still competing
    - 9: error code
    
    Returns:
    -----------------------------------
    a series with the competing deal status codes.
    """
    compete_statc_code = pd.Series(np.nan, index=df.index, name='compete_statc_code')
    # map from group no to the indices of deals in that group
    no_to_index = df.groupby('compete_group_no').apply(lambda x:x.index)
    # exclude groups with only one deal
    count = df.compete_group_no.value_counts()
    count = count[count.ge(2)]
    for group_no in count.index: # each group
        index_group = no_to_index[group_no]
        compete_statc_code[index_group] = \
        create_compete_status_code_single_group(df.statc[index_group], df.dw[index_group])
    return compete_statc_code



def adjust_value_by_CPI(value_ser, date_ser):
    """
    adjust a series of values by cpi index (inflation). 
    If the last available month for cpi index is ealier than the last month of values, ffill the last available CPI to all the months later.
    Baseline is set to be the last month of CPI.
    """
    # to monthly period
    month_ser = date_ser.map(lambda x: pd.Period(f'{x.year}-{x.month}', 'M'))
    cpi_ser = load_CPI_index()
    if cpi_ser.index[-1] < month_ser.max(): # need to add months, use the last available CPI index value
        index_need = pd.date_range(start=str(cpi_ser.index[-1]+1), end=str(month_ser.max()+1), freq="M").to_period("M")
        cpi_ser = pd.concat([cpi_ser, pd.Series(cpi_ser.iloc[-1], index=index_need)])
    adj_factor = cpi_ser.iloc[-1] / cpi_ser
    # return adj_factor, cpi_ser
    return value_ser.mul(adj_factor[month_ser].values)


#####################################
## payment type
#####################################

def extract_all_payment_types(ser):
    """
    returns all the possible payment types in the dataset.
    each element in <ser> is a string of payment types, delimited by \newline
    """
    return np.unique(ser.map(lambda x: x.split('\n'), na_action='ignore').dropna().sum())

def transform_payment_str(string, lst_cash, lst_stock):
    """
    categorize one `consido` string into four groups:
    - 'Cash'
    - 'Common Stock'
    - 'Cash and Common Stock'
    - 'No Cash or Common Stock'.

    strings indicating cash or stock payments are included in <lst_cash> and <lst_stock>
    """
    str_lst = string.split('\n')
    if check_list_of_keys_in_list(lst_cash, str_lst):   # cash payment is included
        if check_list_of_keys_in_list(lst_stock, str_lst):  # stock payment is also included
            return 'Cash and Common Stock'
        else:    # only cash
            return 'Cash'
    elif check_list_of_keys_in_list(lst_stock, str_lst):   # no cash, have stock payment
        return 'Common Stock'
    else:      # no cash or stock
        return 'No Cash or Stock'

    
    
##################################
## consideration
##################################
def convert_consid_to_readable(string):
    """
    convert the consid string to more readable format.
    
    - replace any whitespace by one space
    - replace '/ ' and '/  ' by '/'. 
    - replace '. ' by '.'
    - replace ', ' by ' '
    - replace 'USD ' by '$'
    """
    string = " ".join(string.split())
    return string.replace('/ ', '/').replace('/  ', '/').replace('. ', '.').replace(', ', ' ').replace('USD ', '$')

def convert_consid_single_to_easy_to_parse(consid_single):
    """
    convert a single consideration string to an easy-to-parse format
    """
    consid_single = consid_single.strip()
    consid_single = consid_single.replace(' sh ', ' shs ')
    consid_single = consid_single.replace(' sh/', ' shs com/')
    consid_single = consid_single.replace(' and ', ' plus ')
    consid_single = consid_single.replace(' ord ', ' com ')
    consid_single = consid_single.replace(' ord/', ' com/')
    consid_single = consid_single.replace(' com sh/', ' shs com/')
    consid_single = consid_single.replace(' com shs/', ' shs com/')
    consid_single = consid_single.replace(' ADRs', ' shs com')
    consid_single = consid_single.replace(' ADR', ' shs com')
    consid_single = consid_single.replace(' American depositary shares', ' shs com')
    consid_single = consid_single.replace(' American depositary share', ' shs com')
    consid_single = consid_single.replace('Cl A ', '')
    consid_single = consid_single.replace(' Series A ', '')
    consid_single = consid_single.replace('An estimated ', '')
    consid_single = consid_single.replace('Class A ', '')
    consid_single = consid_single.replace(' in ', ' ')
#     consid_single = consid_single.replace('UAI ', '')
    consid_single = consid_single.replace(' per share ', ' ')
    consid_single = consid_single.replace(' per ', ' ')
    consid_single = consid_single.removeprefix("plus ")
    consid_single = consid_single.replace(' newly issued ', ' ')
    consid_single = consid_single.replace(' newly-issued ', ' ')
    consid_single = consid_single.replace(' new ', ' ')
    consid_single = consid_single.replace(' co ', ' ')
    consid_single = consid_single.replace(' of ', ' ')
    consid_single = consid_single.replace(' US ', ' ')
    consid_single = consid_single.replace(' sh comA/', ' shs com/')
    consid_single = consid_single.replace(' including ', ' plus ')
    consid_single = consid_single.replace(' A', '')
    consid_single = consid_single.replace(' B', '')
    consid_single = consid_single.replace(' Class', '')
    consid_single = consid_single.replace(' class', '')
    
#     consid_single = consid_single.replace(' C', '')
    consid_single_lst = consid_single.split()
    if len(consid_single_lst) >= 2 and consid_single_lst[1]=='cash' and consid_single_lst[0][0]!='$':
        consid_single = '$' + consid_single
        
    if ' plus the assumption' in consid_single:
        consid_single = consid_single.split(" plus the assumption")[0]
    return consid_single


def extract_substr_before_key(key, string):
    """
    if <key> is contained in <string>, returns the substring before the first <key> appearance.
    """
    if key in string:
        return string[:string.find(key)]
    return False

def extract_substr_before_list_of_keys(key_lst, string):
    """
    for the first key in <key_lst> that is contained in the <string>, returns  the substring before the first key appearance.
    """
    for key in key_lst:
        temp = extract_substr_before_key(key, string)
        if temp:
            return temp
    return False


key_list = ['/sh com','/ sh com', '/shs com', '/ shs com', '/com', '/ com', '/sh', '/sh ', '/coma', '/sh ord', '/Class A sh com', '/ Class A sh com'] # '/sh com,',  '/sh com A;', '/ com;',, '/sh ,'  '/sh com A','/sh comA', '/com A',
def extract_terms_from_list_of_consids(key_list, consid_list):
    """
    for a list of considerations, extract the terms
    """
    terms_list = []
    for consid_single in consid_list:
        temp = extract_substr_before_list_of_keys(key_list, consid_single)
        if temp:
            terms_list.append(convert_consid_single_to_easy_to_parse(temp))
    return terms_list


def extract_term_from_consid(consid_ser):
    """
    extract easy-to-parse term string from consid
    """
    consid_lst_ser = consid_ser.str.split(";")
    terms_lst_ser = consid_lst_ser.map(lambda x: extract_terms_from_list_of_consids(key_list, x), na_action='ignore')
    terms_ser = terms_lst_ser.map(lambda x: x[0] if len(x)>0 else np.nan, na_action='ignore')
    return terms_ser


def extract_cash_stock_from_term(term):
    """
    extract cash and stock terms from term string
    """
    index = ['cash_term', 'stock_term', 'payment_type']
    if pd.isna(term):
        return pd.Series([np.nan, np.nan, np.nan], index=index)
    # cash only, "$2.2 cash"
    if len(term.split()) <= 2 and re.search("^\$.* cash$", term) != None:
        cash = term.removeprefix('$').removesuffix(' cash')
        try:
            return pd.Series([float(cash), 0, 'Cash'], index=index)
        except:
            pass
        
    # cash only, '$21'
    if (re.search('\s', term) == None) and (term[0]=='$'):
        cash = term[1:]
        try:
            return pd.Series([float(cash), 0, 'Cash'], index=index)
        except:
            pass
        
    # stock only, '2.1 shs com', '$10 shs com'
    if len(term.split()) <= 3 and (re.search('.* shs com$', term) != None or re.search('.* com$', term) != None or re.search('.* shs$', term) != None):
        stock = term.removesuffix(" shs com").removesuffix(" com").removesuffix(" shs")
        try:
            return pd.Series([0, float(stock), 'Common Stock'], index=index) if stock[0]!='$' \
        else pd.Series([0, float(stock[1:]), 'Common Stock, fixed dollar'], index=index)
        except:
            pass

    # combination, '$8.5 cash plus .85 shs (or shs com, or com)'
    if (re.search('^\$.* cash plus .* shs$', term) != None) or (re.search('^\$.* cash plus .* shs com$', term) != None) or (re.search('^\$.* cash plus .* com$', term) != None):
        term_new = term.removeprefix('$').removesuffix(" shs com").removesuffix(" com").removesuffix(" shs")
        cash = term_new.split()[0]
        stock = term_new.split()[-1]
        try:
            return pd.Series([float(cash), float(stock), 'Cash and Common Stock'], index=index) if stock[0]!='$' \
        else pd.Series([float(cash), float(stock[1:]), 'Cash and Common Stock, fixed dollar'], index=index)
        except:
            pass  
        
    # combination, '0.2109 shs com plus $9 cash', '$12.546 com plus $12.054 cash' (fixed dollar)
    if (re.search('.* com plus \$.* cash$', term) != None) or (re.search('.* shs com plus \$.* cash$', term) != None) or (re.search('.* shs plus \$.* cash$', term) != None):
        term_new = term.removesuffix(" cash")
        cash = term_new.split()[-1][1:]
        stock = term_new.split()[0]
        try:
            return pd.Series([float(cash), float(stock), 'Cash and Common Stock'], index=index) if stock[0]!='$' \
        else pd.Series([float(cash), float(stock[1:]), 'Cash and Common Stock, fixed dollar'], index=index)
        except:
            pass   
        
    # combination, fixed dollar, '$15 cash plus com'
    if (re.search('^\$.* cash plus com$', term) != None) or (re.search('^\$.* cash plus sh com$', term) != None) or (re.search('^\$.* cash plus shs com$', term) != None):
        cash = term.split()[0][1:]
        try:
            return pd.Series([float(cash), 0, 'Cash and Common Stock, fixed dollar'], index=index)# if stock[0]!='$' else (float(cash), float(stock[1:]), 'Cash and Common Stock, fixed dollar')
        except:
            pass
        
    return pd.Series([np.nan, np.nan, 'parse failed'], index=index)

