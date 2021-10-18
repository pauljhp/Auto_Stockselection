# module for getting ticker level fundamental data
# advantage vs yahooquery is AlphaVantage allows 5 years of historical data
# at quarterly frequency
#

import requests, json, os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class AlphaVantage:
    '''Class for collecting data from AlphaVantage'''
    def __init__(self, key):
        self._key = key
        self.__all__ = ['getIncomeStatement', 'getCashFlow', 'getBalanceSheet', 'getEPS',
                        'settings',
                        'getPrice', 'getEarningsCal', 'getFX', 'getCPI', 'getRetailSales']
        self._url = "https://www.alphavantage.co/query?" # function=INCOME_STATEMENT&symbol=IBM&apikey=demo"

    def _getEndPoint(self, **kwargs):
        url = self._url
        for key, val in kwargs.items():
            url += str(key) + "=" + str(val) + "&"
        url += f"apikey={self._key}"
        return url

    def _genericFetcher(self, ticker, field, frequency):
        '''Generic data fetcher for ticker level data'''
        url = self._getEndPoint(function=field, symbol=str(ticker))
        print(url)
        res = requests.get(url).json()
        freqLUT = {'A': 'annual',
                  'Q': 'quarterly'}
        if field == 'EARNINGS':
            Key = 'Earnings'
        else:
            Key = 'Reports'

        df = pd.DataFrame({datetime.strptime(d['fiscalDateEnding'], "%Y-%m-%d"):
                               {key: float(val) if val.isdigit() else val for key, val in d.items()}
             for d in res[f'{freqLUT[frequency]}{Key}']}
        )
        df.columns = pd.MultiIndex.from_tuples([(dt, dt.year) for dt in df.columns])
        df.columns.names = ['datetime','year']
        df.replace('None',np.nan, inplace=True)
        return df

    def getIncomeStatement(self, ticker, frequency = 'A'):
        '''Get Income Statements for the past 5 years from Alpha Vantage'''
        df = self._genericFetcher(ticker, field = 'INCOME_STATEMENT', frequency = frequency)
        return df

    def getBalanceSheet(self, ticker, frequency = 'A'):
        '''Get Balance Sheet for the past 5 years from Alpha Vantage'''
        df = self._genericFetcher(ticker, field='BALANCE_SHEET', frequency=frequency)
        return df

    def getCashFlow(self, ticker, frequency = 'A'):
        '''Get Cashflow statement for the past 5 years from Alpha Vantage'''
        df = self._genericFetcher(ticker, field='CASH_FLOW', frequency=frequency)
        return df

    def getEPS(self, ticker, frequency = 'A'):
        '''Get EPS for the past 5 years from Alpha Vantage'''
        df = self._genericFetcher(ticker, field='EARNINGS',
                                  frequency=frequency)
        return df