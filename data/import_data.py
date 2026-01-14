from futu import *
import alpaca_trade_api as alpaca
import datetime as dt
import pandas as pd
# import yfinance as yf
# from curl_cffi import requests
from enum import Enum

class AlpacaConfig(Enum):
    API_KEY = 'PKO1ARBTPTA3QL7M5BF3'
    API_SECRET = 'bvSSDPdWcQkTNgtO2Fxdbznf1a8tqylrfQCPKanA'
    HEADERS = {
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': API_SECRET
    }
    BARS_URL = 'https://data.alpaca.markets/v2/stocks' 
   

def download_alpaca_daily_data(symbol, start: dt.date, end: dt.date) -> pd.DataFrame:
    print(f"downloading {symbol}")
    api = alpaca.REST(AlpacaConfig.API_KEY.value, AlpacaConfig.API_SECRET.value, base_url='https://paper-api.alpaca.markets')
    if end < dt.date.today() - dt.timedelta(days=2):
        bars = api.get_bars(symbol, alpaca.TimeFrame.Day, start, end).df
        bars.reset_index(inplace=True)
        bars.rename(columns={"timestamp":"date"},inplace=True)
        bars['date'] = bars.date.dt.date
    else:
        bars = api.get_bars(symbol, alpaca.TimeFrame.Day, start).df
        bars.reset_index(inplace=True)
        bars.rename(columns={"timestamp":"date"},inplace=True)
        bars['date'] = bars.date.dt.date
        bars = bars[(bars.date>=start)&(bars.date<=end)]
    return bars[['date','open','high','low','close','volume']]


class Market(Enum):
    HK = 'HK'
    US = 'US'
    CRYPTO = 'CRYPTO'

    def code_list(self):
        with open(f'./data/code_pool_{self.value}.txt','r') as fp:
            list = [line.rstrip() for line in fp]
        return list


def get_sector(code_list):
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    ret, data = quote_ctx.get_owner_plate(code_list)
    if ret != RET_OK:
        print('error:', data)
    quote_ctx.close()
    sector_mapping = pd.read_csv("./data/sector_mapping.csv")
    df = data.set_index('plate_type').loc['INDUSTRY']
    sector_df = df.merge(sector_mapping, on='plate_name', how='inner')
    return sector_df[['code','name','sector']]

# def download_yf_data(code_list, start_date=None, end_date=None):
#     """
#     Download stock data from Yahoo Finance for a list of stocks
#     Args:
#         code_list: list of stock codes
#         start_date: start date (datetime.date or string YYYY-MM-DD)
#         end_date: end date (datetime.date or string YYYY-MM-DD), defaults to today if None
#     Returns:
#         data: DataFrame with stock data from yfinance
#     """
#     session = requests.Session(impersonate="chrome")
#     # Handle date parameters
#     if isinstance(start_date, str):
#         start_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
#     if isinstance(end_date, str):
#         end_date = dt.datetime.strptime(end_date, '%Y-%m-%d').date()
#     if end_date is None:
#         end_date = dt.date.today()
        
#     # Batch download data using yfinance
#     try:
#         data = yf.download(
#             tickers=code_list,
#             start=start_date,
#             end=end_date + dt.timedelta(days=1),
#             group_by='ticker',
#             progress=False,
#             session=session
#         )
#         return data
#     except Exception as e:
#         print(f"Error downloading batch data: {e}")
#         return None
    

def download_futu_historical_daily_data(ticker, start:dt.date, end:dt.date)-> pd.DataFrame:
    '''
    ticker: "HK.00700" or "US.AAPL"
    output: a dataframe with columns: date, open, high, low, close, volume
    '''
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111) 
    if ticker == 'HK.800000':
        ret, df, _ = quote_ctx.request_history_kline(ticker, 
                                                    start=str(start), 
                                                    end= str(end), 
                                                    max_count= None, 
                                                    fields=[KL_FIELD.DATE_TIME,
                                                            KL_FIELD.OPEN,
                                                            KL_FIELD.HIGH,
                                                            KL_FIELD.LOW,
                                                            KL_FIELD.CLOSE,
                                                            KL_FIELD.TRADE_VAL])
    else:
        ret, df, _ = quote_ctx.request_history_kline(ticker, 
                                                        start=str(start), 
                                                        end= str(end), 
                                                        max_count= None, 
                                                        fields=[KL_FIELD.DATE_TIME,
                                                                KL_FIELD.OPEN,
                                                                KL_FIELD.HIGH,
                                                                KL_FIELD.LOW,
                                                                KL_FIELD.CLOSE,
                                                                KL_FIELD.TRADE_VOL])
    if ret !=RET_OK:
        print('error: ', df)
        df = pd.DataFrame()
    else:
        df = df[df.columns[-6:]].rename(columns={"time_key":"date"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        if ticker == 'HK.800000':
            df['volume'] = df.turnover/df.close
            df.drop('turnover', axis=1, inplace=True)
    quote_ctx.close()
    return df