# StockClassification module for automatic stock selection
# For supervised learning

__author__ = 'Paul JP paulpbmo@gmail.com'

import sys, os, re, subprocess, importlib
required_modules={#'google.colab':'google.colab',
                  'yahooquery':'yahooquery',
                  #'numba':'numba',
                  #'openpyxl':'openpyxl',
                }
default_installation_path='/usr/local/lib/python3.7/dist-packages' # on linux VM
 
# for modname, mod in required_modules.items():
#     try:
#         importlib.import_module(modname)
#         print(f"imported {modname}")
#     except ImportError as e:
#         print(f'{e}, installing {modname}...')
#         cmd=f'pip install {mod} --target {default_installation_path} --upgrade'
#         process=subprocess.Popen(cmd.split(),stdout=subprocess.PIPE)
#         print('result:\n',process.communicate())
# print('finished importing')

# from numba import jit
import pandas as pd
import yahooquery as yq
from datetime import date, datetime, timedelta
import numpy as np
from autostockselection.AlphaVantage import AlphaVantage

class stockClassifier:
    def __init__(self, ticker, end = datetime.today(), engine = 'yq', lookback='5y', benchmark = '^IXIC'):
        self._benchmark = benchmark
        self._ticker = ticker
        if engine == 'yq':
            self._engine = 'yq'
            self.yqobj = yq.Ticker([ticker, benchmark])
        elif engine == 'blp':
            self._engine = 'blp'
        self.lookup = {'7d':7,
                       '1m':30,
                       '3m':90,
                       '1y':None,
                       '3y':None,
                       '5y':None,
                       '10y':None
                       }
        self.end = end
        self.today = datetime.today()
        if lookback in self.lookup.keys():
            if self.lookup[lookback]:
                self.start = datetime.today() - timedelta(self.lookup[lookback])
            else:
                self.start = datetime(self.today.year-int(lookback.replace('y','')),self.today.month,self.today.day)
        else:
            raise ValueError("lookback invalid - can only be 7d,1m,3m,1y,3y,5y, or 10y!")
    
    @property
    def ticker(self):
        return self._ticker
    @ticker.setter
    def ticker(self, newticker):
        self._ticker = newticker
    
    @property
    def benchmark(self):
        return self._benchmark
    @benchmark.setter
    def benchmark(self, newbm):
        self._benchmark = newbm
    
    # @jit(nopython=False, parallel=True)
    def historicPrice(self, adj=True):
        if self._engine == 'yq':    
            df = self.yqobj.history(start=self.start-timedelta(days=365), end=self.end, adj_ohlc=adj)
        elif self._engine == 'blp':
            df = blp.bdh(ticker=[self.ticker, self.benchmark],
                         flds='px_last',
                         start_date = self.start,
                         end_date = self.end,
                         adjust = 'all')
            df.columns = df.columns.get_level_values(0) # get rid of the second level index 
        return df
    def calReturn(self):
        df = self.historicPrice().close.unstack().T
        df.loc[:,"ticker_cum_return"]=[(p/df[self.ticker].values[0])**(1/((d-df.index[0]).days/365.25))-1 \
                                if not d==df.index[0] else np.nan \
                                for d,p in df[self.ticker].iteritems()
                               ]
        df.loc[:,"bm_cum_return"]=[(p/df[self.benchmark].values[0])**(1/((d-df.index[0]).days/365.25))-1 \
                                if not d==df.index[0] else np.nan \
                                for d,p in df[self.benchmark].iteritems()
                               ]
        return df[df.index>=self.start.date()]

    def activeReturnDF(self):
        close_v_bm = self.calReturn()
        close_v_bm['activeReturn'] = close_v_bm.ticker_cum_return - close_v_bm.bm_cum_return
        return close_v_bm
    
    def activeReturn(self):
        return self.activeReturnDF().activeReturn[-1]

    def avgactiveReturn(self):
        return self.activeReturnDF().activeReturn.mean()

    def activeRisk(self):
        return self.activeReturnDF().activeReturn.std()

    def informationRatio(self):
        return self.activeReturn()/self.activeRisk() if self.activeRisk() else np.nan

    def avgInfoRatio(self):
        return self.avgactiveReturn()/self.activeRisk() if self.activeRisk() else np.nan


class preProcessing(stockClassifier):
    '''Class for creating dataset of historic financial data for classification'''

    def __init__(self, ticker, end, lookback, benchmark, engine='blp'):  # , ticker, end):
        super().__init__(ticker, end, lookback, benchmark, engine)

    def _blpFeaturize(self, lookforward='3y') -> pd.DataFrame:  # TODO - build lookforward str->int conversion
        '''generate vector of features for a given (ticker, startdate) pair with blp engine'''
        ticker = self._ticker
        start = self.start
        end = self.end
        df = pd.DataFrame(dtype=object,
                          index=pd.MultiIndex.from_tuples([(ticker, end)])
                          )
        df.index.names = ['ticker', 'start']

        # features
        # income statement features
        IS_flds = ['sales_growth', 'sales_5yr_avg_gr', 'free_cash_flow_5_year_growth', 'ebitda_growth',
                   'ebitda_growth_adjsuted_5yr_cagr',
                   'gross_margin', 'ebitda_margin', 'ebit_margin', 'rd_expend_to_net_sales', 'd&a_to_sales',
                   'T12_sg&a_expn_%_t12_sales', 'int_exp_to_net-sales', 'net_income_margin_adjustsed',
                   'eps_growth', 'eff_tax_rate', 'eps_surprise_last_qtr', 'eps_surprise_last_annual']
        BS_flds = ['cash_ratio', 'quick_ratio', 'cur_ratio',
                   'invent_days', 'accounts_payable_turnover_days', 'days_accounts_receivable',
                   'cash_conversion_cycle', 'fixed_charge_coverage_ratio', 'interest_coverage_ratio',
                   'asset_to_eqy', 'net_debt_to_shrhldr_eqty', 'tot_debt_to_tot_eqy']
        CF_flds = ['cap_expend_ratio', 'free_cash_flow_margin', 'free_cash_flow_margin_3yr_avg', 'dvd_payout_ratio',
                   'total_payout_ratio', 'cfo_to_sales']
        VAL_flds = ['cash_%_current_market_cap', 'px_to_cash_flow', 'best_pe_ratio', 'best_cur_ev_to_ebitda',
                    'px_to_sales_ratio', 'eqy_dvd_yld_ind', 'px_to_tang_bv_per_sh', 'market_capitalization_to_bv']
        return_flds = ['operating_roic', 'normalized_roe', 'oper_roe', 'return_on_asset', 'net_fixed_asset_turn',
                       'asset_turnover']
        size_flds = ['historical_market_cap', 'sales_rev_turn', 'total_equity']
        market_flds = ['volatility_90d', 'hist_trr_prev_1yr']
        other_flds = ['cntry_of_risk',
                      'employee_turnover_pct']  # TODO add IPO date, years of listing and years since founding

        # current valuation
        tmp = blp.bdh(ticker, ticker, flds=VAL_flds, start_date=self.end - timedelta(days=5), end_date=self.end,
                      currency='USD')  # previous 5 days in case end date is a holiday
        for i in VAL_flds:
            df.loc[(ticker, end), i + '_cur'] = temp[(ticker), i][-1]

        # hist relative return
        df.loc[(ticker, end), 'H_activeReturn'] = self.activeReturn()
        df.loc[(ticker, end), 'H_avgactiveReturn'] = self.avgactiveReturn()
        df.loc[(ticker, end), 'H_activeRisk'] = self.activeRisk()
        df.loc[(ticker, end), 'H_IR'] = self.informationRatio()
        df.loc[(ticker, end), 'H_avgIR'] = self.avgInfoRatio()

        if df.loc[(ticker, end), 'H_activeReturn'] > 0.05:  # 5% or more relative outperf
            x = int(2)
        elif 0 < df.loc[(ticker, end), 'H_activeReturn'] <= 0.05:
            x = int(1)
        elif -0.05 < df.loc[(ticker, end), 'H_activeReturn'] <= 0:
            x = int(-1)
        else:
            x = int(-2)

        if df.loc[(ticker, end), 'H_avgIR'] > 2:
            y = int(2)
        elif 1 < df.loc[(ticker, end), 'H_avgIR'] <= 2:
            y = int(1)
        elif 0 < df.loc[(ticker, end), 'H_avgIR'] <= 1:
            y = int(-1)
        else:
            y = int(-2)

        df.loc[(ticker, end), 'H_category'] = str((x, y))

        # Target variables
        T = stockClassifier(ticker,
                            end=datetime(self.end.year + int(lookforward[:-1]), self.end.month, self.end.day),
                            lookback=lookforward, benchmark=self.benchmark, engine=self._engine
                            )

        df.loc[(ticker, end), 'Target_activeReturn'] = T.activeReturn()
        df.loc[(ticker, end), 'Target_avgactiveReturn'] = T.avgactiveReturn()
        df.loc[(ticker, end), 'Target_activeRisk'] = T.activeRisk()
        df.loc[(ticker, end), 'Target_IR'] = T.informationRatio()
        df.loc[(ticker, end), 'Target_avgIR'] = T.avgInfoRatio()

        if df.loc[(ticker, end), 'Target_activeReturn'] > 0.05:  # 5% or more relative outperf
            x = int(2)
        elif 0 < df.loc[(ticker, end), 'Target_activeReturn'] <= 0.05:
            x = int(1)
        elif -0.05 < df.loc[(ticker, end), 'Target_activeReturn'] <= 0:
            x = int(-1)
        else:
            x = int(-2)

        if df.loc[(ticker, end), 'Target_avgIR'] > 2:
            y = int(2)
        elif 1 < df.loc[(ticker, end), 'Target_avgIR'] <= 2:
            y = int(1)
        elif 0 < df.loc[(ticker, end), 'Target_avgIR'] <= 1:
            y = int(-1)
        else:
            y = int(-2)

        df.loc[(ticker, end), 'Target_category'] = str((x, y))

        for flds in [IS_flds, BS_flds, CF_flds, VAL_flds, market_flds]:
            tmp = blp.bdh(ticker, flds=flds,
                          start_date=self.start,  # FIXME - lookback period reference
                          end_date=self.end,
                          currency='USD')
            tmp_avg = tmp.mean(axis='index')[ticker]
            tmp_avg.index = [i + '_avg' for i in tmp_avg.index]
            tmp_avg.name = (ticker, end)
            df = df.iloc[0, :].append(tmp_avg).to_frame().T
            #             print(df)
            tmp_std = tmp.std(axis='index')[ticker]
            tmp_std.index = [i + '_std' for i in tmp_std.index]
            tmp_std.name = (ticker, end)
            df = df.iloc[0, :].append(tmp_std).to_frame().T

        # size flds - no need to get std
        tmp = blp.bdh(ticker, flds=size_flds,
                      start_date=self.start,  # FIXME - lookback period reference
                      end_date=self.end,
                      currency='USD')
        tmp_avg = tmp.mean(axis='index')[ticker]
        tmp_avg.index = [i + '_avg' for i in tmp_avg.index]
        df = df.iloc[0, :].append(tmp_avg).to_frame().T

        return df

    def _avFeaturize(self, AlphaVantageKey, lookback='3y'): # FIXME - lookback period for child class
        '''generate vector of features for a given (ticker, startdate) pair'''
        ticker = self._ticker
        start = self.start
        df = pd.DataFrame(dtype=object,
                          index=pd.MultiIndex.from_tuples([(ticker, start)])
                          )
        df.index.names = ['ticker', 'start']
        # Target variables
        df.loc[(ticker, start),'Target_activeReturn'] = self.activeReturn()
        df.loc[(ticker, start),'Target_avgactiveReturn'] = self.avgactiveReturn()
        df.loc[(ticker, start),'Target_activeRisk'] = self.activeRisk()
        df.loc[(ticker, start), 'Target_IR'] = self.informationRatio()
        df.loc[(ticker, start), 'Target_avgIR'] = self.avgInfoRatio()
        if df.loc[(ticker, start),'Target_activeReturn'] > 0.05: # 5% or more relative outperf
            x = int(2)
        elif 0 < df.loc[(ticker, start),'Target_activeReturn'] <= 0.05:
            x = int(1)
        elif -0.05 < df.loc[(ticker, start),'Target_activeReturn'] <= 0:
            x = int(-1)
        else:
            x = int(-2)

        if df.loc[(ticker, start),'Target_avgIR'] > 2:
            y = int(2)
        elif 1 < df.loc[(ticker, start),'Target_avgIR'] <= 2:
            y = int(1)
        elif 0 < df.loc[(ticker, start), 'Target_avgIR'] <= 1:
            y = int(-1)
        else:
            y = int(-2)

        df.loc[(ticker, start), 'category'] = str((x, y))

        # historic active return
        # hist_end = start
        # hist = stockClassifier(ticker=ticker, end=hist_end, lookback=lookback)
        # print(hist_end, lookback, hist.historicPrice())
        # df.loc[(ticker, start), 'H_activeReturn'] = hist.activeReturn()
        # df.loc[(ticker, start), 'H_avgactiveReturn'] = hist.avgactiveReturn()
        # df.loc[(ticker, start), 'H_activeRisk'] = hist.activeRisk()
        # df.loc[(ticker, start), 'H_IR'] = hist.informationRatio()
        # df.loc[(ticker, start), 'H_avgIR'] = hist.avgInfoRatio()

        # fundamental data
        av = AlphaVantage(key = AlphaVantageKey) # FIXME - remove key when uploading

        # trailing EPS data
        df_EPSA = av.getEPS(ticker).T.reset_index(0).reportedEPS # default annual frequency
        df_EPSA = df_EPSA.to_frame() #.query(f'year<{start.year}')
        df_EPSA.index = df_EPSA.index.astype(int)
        df_EPSQ = av.getEPS(ticker, 'Q').T.reset_index(0).surprisePercentage.astype(float).groupby('year').mean()
        df_EPSQ = df_EPSQ.to_frame() #.query(f'year<{start.year}')

        Pend = int(start.year - 1)
        for i in [1, 2, 3, 5, 7, 10]:
            Pstart = int(Pend - i - 1)
            if Pstart in df_EPSA.index:
                df.loc[(ticker, start),f'T{i}yEPSg'] = (float(df_EPSA.loc[Pend])/float(df_EPSA.loc[Pstart]))**(1 / int(i)) - 1 \
                    if float(df_EPSA.loc[Pstart]) else np.nan
            if Pstart in df_EPSQ.index:
                df.loc[(ticker, start), f'T{i}yEPSSurprise'] = df_EPSQ.query(f"year>={Pstart}&year<={Pend}").mean().values / 100
        df_IS = av.getIncomeStatement(ticker,'Q').T
        df_IS = df_IS[df_IS.index.get_level_values(0) < start]
        df_BS = av.getBalanceSheet(ticker, 'Q').T
        df_BS = df_BS[df_BS.index.get_level_values(0) < start]
        df_CF = av.getCashFlow(ticker, 'Q').T
        df_CF = df_CF[df_CF.index.get_level_values(0) < start]
        # print(df_IS)

        # income statement to features

        df.loc[(ticker, start),'avgGM'] = (df_IS.grossProfit.astype(float) / df_IS.totalRevenue.astype(float)).mean()
        df.loc[(ticker, start), 'avgOPM'] = (df_IS.ebit.astype(float) / df_IS.totalRevenue.astype(float)).mean()
        df.loc[(ticker, start), 'avgRD'] = (
                df_IS.researchAndDevelopment.astype(float) / df_IS.totalRevenue).astype(float).mean()
        df.loc[(ticker, start), 'avgDnA'] = (
                df_IS.depreciationAndAmortization.astype(float) / df_IS.totalRevenue.astype(float)).mean()
        df.loc[(ticker, start), 'avgInt'] = (
                df_IS.interestAndDebtExpense.astype(float) / df_IS.totalRevenue.astype(float)).mean()
        df.loc[(ticker, start), 'avgEBITDA'] = (df_IS.ebitda.astype(float) / df_IS.totalRevenue.astype(float)).mean()
        df.loc[(ticker, start), 'avgNM'] = (df_IS.netIncome.astype(float) / df_IS.totalRevenue.astype(float)).mean()

        # balance sheet features
        df.loc[(ticker, start), 'avgCashR'] = (
                df_BS.cashAndCashEquivalentsAtCarryingValue.astype(float) / df_BS.totalCurrentLiabilities.astype(float)).mean()
        df.loc[(ticker, start), 'avgCurR'] = (
                    df_BS.totalCurrentAssets.astype(float) / df_BS.totalCurrentLiabilities.astype(float)).mean()
        df.loc[(ticker, start), 'avgQuickR'] = (
                (df_BS.totalCurrentAssets.astype(float) - df_BS.inventory.astype(float)) / df_BS.totalCurrentLiabilities.astype(float)).mean()
        df.loc[(ticker, start), 'avgTTGearing'] = (
                (df_BS.shortTermDebt.astype(float) +
                 df_BS.longTermDebt.astype(float)) / (df_BS.totalShareholderEquity.astype(float))).mean()
        df.loc[(ticker, start), 'avgNetGearing'] = (
                (df_BS.shortTermDebt.astype(float) + df_BS.longTermDebt.astype(float)
                  - df_BS.cashAndCashEquivalentsAtCarryingValue.astype(float)) /
                (df_BS.totalShareholderEquity.astype(float))).mean()
        df.loc[(ticker, start), 'avgInvDays'] = (df_BS.inventory.astype(float) / df_IS.costOfRevenue.astype(float)).mean() * 91.25
        df.loc[(ticker, start), 'avgARDays'] = (df_BS.currentNetReceivables.astype(float) / df_IS.totalRevenue.astype(float)).mean() * 91.25
        df.loc[(ticker, start), 'avgAPDays'] = (df_BS.currentAccountsPayable.astype(float) / df_IS.costOfRevenue.astype(float)).mean() * 91.25
        df.loc[(ticker, start), 'avgWK'] = ((df_BS.inventory.astype(float) + df_BS.currentNetReceivables.astype(float)
                - df_BS.currentAccountsPayable.astype(float)) / (df_IS.totalRevenue)).mean()

        # cash flow features
        df.loc[(ticker, start), 'CFOtoEBITDA'] = (df_CF.operatingCashflow.astype(float) / df_IS.ebitda.astype(float)).mean()
        df.loc[(ticker, start), 'CFOmargin'] = (df_CF.operatingCashflow.astype(float) / df_IS.totalRevenue.astype(float)).mean()
        df.loc[(ticker, start), 'DVDpayout'] = (df_CF.dividendPayout.astype(float) / df_IS.netIncome.astype(float)).mean()
        df.loc[(ticker, start), 'shRepoR'] = (df_CF.paymentsForRepurchaseOfEquity.astype(float) / df_IS.netIncome.astype(float)).mean()
        df.loc[(ticker, start), 'TTredisttoSHtoNI'] = ((df_CF.dividendPayout.astype(float) +
                                                  df_CF.paymentsForRepurchaseOfEquity.astype(float)) / df_IS.netIncome.astype(float)).mean()
        df.loc[(ticker, start), 'capEx'] = (df_CF.capitalExpenditures.astype(float) / df_IS.totalRevenue.astype(float)).mean()
        df.loc[(ticker, start), 'FCFtoRev'] = ((df_CF.operatingCashflow.astype(float) + df_IS.interestAndDebtExpense.astype(float) * (1 -
                                                                        df_IS.incomeTaxExpense.astype(float) / df_IS.incomeBeforeTax.astype(float))
                                                - df_CF.capitalExpenditures.astype(float)) / df_IS.totalRevenue.astype(float)).mean()
        # Valuation features
        T = yq.Ticker(ticker)
        hist_px = T.history(start=self.start-timedelta(days=365*3), end=self.start, adj_ohlc=True)

        if type(hist_px) is pd.DataFrame:
            curP = hist_px.loc[ticker, 'close'][-1]
            hist_px.reset_index(0, inplace=True)
            hist_px.drop('symbol', axis='columns', inplace=True)
            hist_px['year'] = [d.year for d in hist_px.index]
            hist_px = hist_px.groupby('year').mean().close.to_frame()
            # print(hist_px.loc['BABA','close'])
            curTPE = curP / float(df_EPSA.loc[Pend])
            PE = []
            for year, row in hist_px.iterrows():
                PE.append(row.close/float(df_EPSA.loc[year]) if year in df_EPSA.index else None)

            avghistTPE = sum(PE) / sum([1 for i in PE if i]) if sum([1 for i in PE if i])  else np.nan

            df.loc[(ticker, start), 'curTPE'] = curTPE
            df.loc[(ticker, start), 'avghistTPE'] = avghistTPE

        return df

    def featurize(self, lookforward='3y', AlphaVantageKey=None):
        if self._engine == 'yq':
            return self._avFeaturize(AlphaVantageKey)
        elif self._engine == 'blp':
            return self._blpFeaturize(lookforward)