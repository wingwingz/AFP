class VolTargetRebalance:
    
    def __init__(self, 
                 aum, 
                 asset_names, # list of asset class names
                 returns, # dataframe of log returns with assets in same order with asset_classes
                 time, # daily return:252, weekly return:52 ...
                 target_vol, # float, in percentage, usually 10 - 15
                 burn, # number of observations to drop to smooth out impact of initial vol estimation
                 barrier=1.5, # risk control - when to switch to short vol
                 lam_short=0.98, # lambda for EWMA portfolio vol / correlation calculation
                 lam_long=0.995, # lambda for EWMA vol weight rebalance calculation
                 leverage_cap=1.5# float, only in effect for vol target funds i.e. nasset = 1 for naive rebalance
                ):
        self.aum = aum
        self.asset_names = asset_names
        self.returns = returns
        self.time = time
        self.target_vol = target_vol
        self.leverage_cap = leverage_cap
        self.lam_short = lam_short
        self.lam_long = lam_long
        self.burn = burn
        self.barrier = barrier
        
        self.nassets = len(asset_names)
        self.nt = len(returns) - burn
        if self.nassets == 1:
            self.r = returns.values.reshape(-1, 1)
        else:
            self.r = returns.values

        vols = np.empty(self.r.shape)
        for i in range(self.nassets):
            vols[:,i] = self.calc_EWMA_vol(self.r[:,i], lam_long, dataframe=False, verbose=False)
        self.vols = vols
        
    def calc_EWMA_vol(self, returns, lam, dataframe=True, verbose=True):
        r_sq = returns ** 2
        if dataframe:
            sigma_sq = pd.Series(index=r_sq.index, dtype='float64')
        else:
            sigma_sq = np.empty(len(returns))
        sigma_sq[0] =r_sq[0]
        for i in range(1, len(r_sq)):
            sigma_sq[i] = (1 - lam) * r_sq[i] + lam * sigma_sq[i - 1]
        sigma = np.sqrt(sigma_sq) * np.sqrt(self.time) * 100
        if verbose:
            print("half-life:", -np.log(2)/np.log(lam))
            print("avg annual volatility:", sigma.mean())
        return sigma

        
    def get_weights(self):
        raise NotImplementedError
        
    def get_flow(self, threshold_small=0.01, threshold_big=0.2):
        target_weights = self.get_weights()
        RP_k = np.empty(target_weights[:-1].shape)
        RP_fund = np.empty(RP_k.shape[0])
        trade = np.empty(target_weights[:-1].shape)
        NAV = np.empty((self.nt, 1))
        NAV[0] = self.aum
        RP_k[0] = np.multiply(target_weights[0], self.r[self.burn + 1])
        RP_fund[0] = RP_k[0].sum()
        real_weights = target_weights.copy()

        for i in range(1, self.nt - 1):

            NAV[i] = NAV[i - 1] * (1 + RP_fund[i - 1])
            postTradePos = target_weights[i] * NAV[i]
            preTradePos = real_weights[i - 1] * NAV[i - 1] * (1 + self.r[self.burn + i])
            trade[i - 1] = postTradePos - preTradePos
            trade_weight = trade[i - 1] / (NAV[i] * target_weights[i])

            if np.all(np.abs(trade_weight) <= threshold_small):
                real_weights[i] = target_weights[i - 1] * (1 + self.r[self.burn + i])
                trade[i - 1] = 0
            elif np.any(np.abs(trade_weight) >= threshold_big):
                trade[i - 1] = trade[i - 1] / (max(np.abs(trade_weight)) / 0.1)
                real_weights[i] = (trade[i - 1] + preTradePos) / NAV[i]

            RP_k[i] = np.multiply(real_weights[i], self.r[self.burn + i + 1])
            RP_fund[i] = RP_k[i].sum()
        
        NAV[-1] = NAV[-2] * (1 + RP_fund[-1])
        postTradePos_last = target_weights[-1] * NAV[-1]
        preTradePos_last = real_weights[-2] * NAV[-2] * (1 + self.r[-1])
        trade[-1] = postTradePos_last - preTradePos_last
        if np.all(np.abs(trade[-1] / NAV[-1]) <= threshold_small):
            real_weights[-1] = target_weights[-2] * (1 + self.r[-1])
            trade[-1] = 0
        
        df_trade = pd.DataFrame(data=trade, index=self.returns[self.burn+1:].index, columns=self.asset_names)
        df_trade_pct = df_trade / NAV[1:]
        df_weights = pd.DataFrame(data=np.c_[target_weights[:-1], real_weights[:-1]], index=self.returns[self.burn+1:].index, columns=[x + y for x in ['target_', 'real_'] for y in self.asset_names])
        df_returns = pd.DataFrame(data=RP_fund, index=self.returns[self.burn+1:].index, columns=['fund_return'])
 
        return df_trade, df_trade_pct, df_weights, df_returns
        
class NaiveRebalance(VolTargetRebalance):
    
    def get_weights(self):
        weights = self.target_vol / self.nassets / self.vols[self.burn:]
        weights = np.atleast_2d(weights)

        realized_vol = np.empty(self.nt)
        ratio = np.empty(self.nt)
        for i in range(self.nt):
            portfolio_returns = self.r[:self.burn + i + 1] @ weights[i, :]
            short_vol = self.calc_EWMA_vol(portfolio_returns, self.lam_short, dataframe=False, verbose=False)[-1]
            long_vol = self.calc_EWMA_vol(portfolio_returns, self.lam_long, dataframe=False, verbose=False)[-1]
            ratio[i] = short_vol / long_vol
            realized_vol[i] = long_vol if ratio[i] < self.barrier else short_vol

        plt.plot(self.returns[self.burn:].index, ratio, label=r'$\lambda^s$={}, $\lambda^l$={}'.format(self.lam_short, self.lam_long))
        plt.legend()                             
        plt.title('Ratio between EWMA Portfolio Vol with $\lambda$ = {} vs EWMA Portfolio Vol with $\lambda$ = {}'.format(self.lam_short, self.lam_long))
        
        leverage = self.target_vol / realized_vol
        weights_rescaled = np.multiply(weights, leverage.reshape(-1, 1))
        if self.nassets == 1:
            weights_rescaled[weights_rescaled > self.leverage_cap] = self.leverage_cap
        return weights_rescaled
    
class HistCorrRebalance(VolTargetRebalance):
    
    def _get_portfolio_risk(self, weights, cov):
        weights = np.matrix(weights)
        return np.sqrt((weights * cov * weights.T))[0, 0]

    def _get_risk_contribution(self, weights, cov):
        weights = np.matrix(weights)
        portfolio_risk = self._get_portfolio_risk(weights, cov)
        return np.multiply(weights.T, cov * weights.T) / portfolio_risk ** 2

    def _risk_budget_objective_error(self, weights, args):
        cov = args[0]
        assets_risk_budget = args[1]
        weights = np.matrix(weights)
        portfolio_risk = self._get_portfolio_risk(weights, cov)
        assets_risk_contribution = self._get_risk_contribution(weights, cov)
        assets_risk_target = np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))
        return sum(np.square(assets_risk_contribution - assets_risk_target.T))[0, 0]

    def _get_risk_parity_weights(self, cov, assets_risk_budget, initial_weights):
        constraints = ({'type': 'eq', 'fun': lambda x: self._get_portfolio_risk(x, cov) - self.target_vol / 100},
                       {'type': 'ineq', 'fun': lambda x: x})
        optimize_result = minimize(fun=self._risk_budget_objective_error, x0=initial_weights, args=[cov, assets_risk_budget], \
                                   method='SLSQP', tol=1e-10, constraints=constraints, options={'disp': False})
        weights = optimize_result.x
        return weights

    def get_weights(self):
        weights = np.empty((self.nt, self.nassets))
        for i in range(self.nt):
            weights[i,:] = self._get_risk_parity_weights(self.returns[i:self.burn+i].cov().values * self.time, [0.25,0.25,0.25,0.25], [0.25,0.25,0.25,0.25])
        return weights