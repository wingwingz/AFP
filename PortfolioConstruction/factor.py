import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

def construct_factor(df_factor_measure, df_return_all, ascending, q=1/3):
    '''
    + Inputs: df_factor_measure is the factor df; df_return_all is the returns df
    + Outputs: df_factor_ret is the returns on the factor portfolio; df_port is the long-short positions
    '''
    
    df_rank = df_factor_measure.rank(axis=1, ascending=ascending, pct=True)

    df_pos = df_rank.applymap(lambda x:1 if x <= q else 0)
    df_neg = df_rank.applymap(lambda x:-1 if x > (1-q) else 0)
    
    df_return = df_return_all.loc[df_factor_measure.index, df_factor_measure.columns]

    df_pos_ret_country = df_pos.multiply(df_return).div(df_pos.sum(axis=1), axis=0)
    df_pos_ret = df_pos_ret_country.sum(axis=1)
    
    df_neg_ret_country = df_neg.multiply(df_return).div(-df_neg.sum(axis=1), axis=0)
    df_neg_ret = df_neg_ret_country.sum(axis=1)
    
    df_factor_ret = df_pos_ret + df_neg_ret
    df_port = df_pos + df_neg
    
    #df_factor_cum_ret = np.exp(df_factor_ret.cumsum()) - 1
    
    return df_factor_ret, df_port


def calc_momentum_ret(df, window=52):
    '''
    Takes input a dataframe, containing weekly equity etf returns, and finds cumulative returns for trailing year
    + Input: df: equity_rets_w
    + Output: mom_df
    '''
    rolling_sum = df.rolling(window, closed='left').sum().dropna()
    rolling_ret_mom = np.exp(rolling_sum) - 1
    
    return rolling_ret_mom


def calc_beta_ret(df, market_port_ret, window=52):
    # Find country beta's through rolling regression
    y = market_port_ret
    rolling_betas = {}
    for c in df.columns:
        X = sm.add_constant(df[c])
        model = RollingOLS(y, X, window)
        rolling_res = model.fit(params_only=True)
        rolling_betas[c] = rolling_res.params.dropna()
    
    # Put all beta's for every country and every date in a dataframe
    out_df = pd.DataFrame()
    for key, value in rolling_betas.items():
        col = pd.DataFrame(value[key])
        if out_df.empty:
            out_df = out_df.append(col)
        else:
            out_df = pd.concat([out_df, col], axis=1)
    
    return out_df

