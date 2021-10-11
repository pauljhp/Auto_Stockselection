# StockClassification module for automatic stock selection
# For supervised learning

__author__ = 'Paul JP paulpbmo@bmo.com'

import sys, os, re, subprocess, importlib
required_modules={'google.colab':'google.colab',
                  'yahooquery':'yahooquery',
                  'numba':'numba',
                  'openpyxl':'openpyxl',
                }
default_installation_path='/usr/local/lib/python3.7/dist-packages' # on linux VM
 
for modname, mod in required_modules.items():
    try:
        importlib.import_module(modname)
        print(f"imported {modname}")
    except ImportError as e:
        print(f'{e}, installing {modname}...')
        cmd=f'pip install {mod} --target {default_installation_path} --upgrade'
        process=subprocess.Popen(cmd.split(),stdout=subprocess.PIPE)
        print('result:\n',process.communicate())
print('finished importing')

# from numba import jit
import pandas as pd
import yahooquery as yq
from datetime import date, datetime, timedelta
import numpy as np

class stockClassifier:
    def __init__(self, ticker, end = datetime.today(), engine = 'yq', lookback='5y', benchmark = '^IXIC'):
        self._benchmark = benchmark
        self._ticker = ticker
        if engine == 'yq':
            self._engine = 'yq'
            self.yqobj = yq.Ticker([ticker,benchmark])
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
            df = self.yqobj.history(start=self.start-timedelta(days=365),end=self.end,adj_ohlc=adj)
        elif self._engine == 'blp':
            df = blp.bdh(ticker=[self.ticker,self.benchmark],
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