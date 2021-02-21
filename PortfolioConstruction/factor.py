import pandas as pd

def construct_factor(df_factor_measure, df_return_all, ascending):
    '''
    + Inputs: df_factor_measure is the factor df; df_return_all is the returns df
    + Outputs: df_factor_ret is the returns on the factor portfolio; df_port is the long-short positions
    '''
    
    df_rank = df_factor_measure.rank(axis=1, ascending=ascending, pct=True)

    df_pos = df_rank.applymap(lambda x:1 if x <= 1/3 else 0)
    df_neg = df_rank.applymap(lambda x:-1 if x > 2/3 else 0)
    
    df_return = df_return_all.loc[df_factor_measure.index, df_factor_measure.columns]

    df_pos_ret_country = df_pos.multiply(df_return).div(df_pos.sum(axis=1), axis=0)
    df_pos_ret = df_pos_ret_country.sum(axis=1)
    
    df_neg_ret_country = df_neg.multiply(df_return).div(-df_neg.sum(axis=1), axis=0)
    df_neg_ret = df_neg_ret_country.sum(axis=1)
    
    df_factor_ret = df_pos_ret + df_neg_ret
    df_port = df_pos + df_neg
    
    #df_factor_cum_ret = np.exp(df_factor_ret.cumsum()) - 1
    
    return df_factor_ret, df_port