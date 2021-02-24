import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate cumulative returns
def calc_cum_returns(df, log_ret = True):
    '''
    Computes cumulative returns either from simple returns or log returns
    + Input: df dataframe containing return series
    '''
    if log_ret == False:
        df = np.log(df + 1)
    
    df_ = df.copy()
    df_ = df_.dropna()
    df_ = np.exp(df_.cumsum()) - 1
    df_.columns = df_.columns.map(lambda x : x+'_cum_ret')
    return df_

# Generate performance assessment statistics
def calc_performance(rets_series, log_ret = True, time = 52):
    '''
    Generates performance metrics for portfolio or market
    + Input: return series (in log returns)
    + Output: mean annualized returns, SR, volatility
    '''
    if log_ret == False:
        rets_series = np.log(rets_series + 1)
    
    perf_df = pd.DataFrame(rets_series)
    perf_df.columns = ['returns']
    
    # Compute mean return, annualized
    def calc_mean_return(df, column):
        mean_log_return = np.mean(df[column]) * time
        # Convert back to simple returns
        mean_return_annualized = np.exp(mean_log_return) - 1
        return mean_return_annualized
    
    # Compute volatility, annualized
    def calc_vol(df, column):
        simple_returns = np.exp(df[column]) - 1
        std_annualized = simple_returns.std() * np.sqrt(time)
        return std_annualized
    
    # Compute SR, annualized
    def calc_sharpe_ratio(df, column):
        mean_ret = calc_mean_return(df, column)
        std_ = calc_vol(df, column)
        sr_annualized = mean_ret/std_
        return sr_annualized
    
    # Compute skewness and kurtosis
    def calc_skewness_kurtosis(df, column):
        skewness = df[column].skew()
        kurtosis = df[column].kurtosis()
        return skewness, kurtosis

    # Compute max drawdown 
    def calc_max_drawdown(df, column):
        # Calculate using simple returns
        returns = np.exp(df[column])-1
        cum_returns = (1 + returns).cumprod()
        drawdown = 1 - cum_returns.div(cum_returns.cummax())
        max_drawdown = np.max(drawdown.expanding().max())
        max_drawdown_date = (drawdown.expanding().max().idxmax()).strftime('%Y-%m-%d')
        return max_drawdown, max_drawdown_date
    
    # Compute all metrics for portfolio
    mean_return_ann = calc_mean_return(perf_df, 'returns')
    std_ann = calc_vol(perf_df, 'returns')
    sr_ann = calc_sharpe_ratio(perf_df, 'returns')
    skewness, kurtosis = calc_skewness_kurtosis(perf_df, 'returns')
    max_drawdown, max_drawdown_date = calc_max_drawdown(perf_df, 'returns')
    
    all_ = [round(mean_return_ann*100,2), round(std_ann*100,2), round(sr_ann, 2), \
            round(skewness, 2), round(kurtosis, 2), round(max_drawdown*100, 2), max_drawdown_date]
    
    return all_

def plot_corr(factor_returns_df, title):
    '''
    Takes in factor returns dataframe and outputs factor correlations matrix
    '''
    # Construct correlation matrix 
    corr = factor_returns_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask = mask, annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
    plt.title(title)
    plt.show()


# Generate graphs of returns
def plot_perf(perf, perf_bm, label, title, benchmark_label = 'Market-RF'):
    fig, ax = plt.subplots(nrows=1, figsize=(12,6))
    if type(perf) is not list:
        perf = [perf]
        label = [label]
    for i in range(len(perf)):
        ax.plot(perf[i], linewidth=1, label=label[i])
    ax.plot(perf_bm, linewidth=1, linestyle='--', c='black', label=benchmark_label)
    ax.set_ylabel('Cumulative Returns')
    ax.set_title(title)
    ax.legend(loc='best')