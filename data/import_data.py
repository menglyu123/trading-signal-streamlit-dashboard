from futu import *
import datetime as dt
import pandas as pd
import yfinance as yf
from curl_cffi import requests


with open('./data/code_pool_hk.txt','r') as fp:
    HK_CODE_LIST = [line.rstrip() for line in fp]

def get_sector(code_list):
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    with open('./data/code_pool_hk.txt','r') as fp:
        code_pool = [line.rstrip() for line in fp]
    code_list = ['HK.0'+code for code in code_pool]
    ret, data = quote_ctx.get_owner_plate(code_list)
    if ret != RET_OK:
        print('error:', data)
    quote_ctx.close()
    sector_mapping = pd.read_csv("./data/sector_mapping.csv")
    df = data.set_index('plate_type').loc['INDUSTRY']
    sector_df = df.merge(sector_mapping, on='plate_name', how='inner')
    sector_df['code'] = sector_df.code.apply(lambda x: x[4:])
    return sector_df[['code','sector']]

def download_yf_data(code_list, start_date=None, end_date=None):
    """
    Download stock data from Yahoo Finance for a list of stocks
    Args:
        code_list: list of stock codes
        start_date: start date (datetime.date or string YYYY-MM-DD)
        end_date: end date (datetime.date or string YYYY-MM-DD), defaults to today if None
    Returns:
        data: DataFrame with stock data from yfinance
    """
    session = requests.Session(impersonate="chrome")
    # Handle date parameters
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%d').date()
    if end_date is None:
        end_date = dt.date.today()
        
    # Batch download data using yfinance
    try:
        data = yf.download(
            tickers=code_list,
            start=start_date,
            end=end_date + dt.timedelta(days=1),
            group_by='ticker',
            progress=False,
            session=session
        )
        return data
    except Exception as e:
        print(f"Error downloading batch data: {e}")
        return None
    

def download_futu_historical_daily_data(ticker, start:dt.date, end:dt.date)-> pd.DataFrame:
    '''
    ticker: "HK.00700" or "US.AAPL"
    output: a dataframe with columns: date, open, high, low, close, volume
    '''
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111) 
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
    quote_ctx.close()
    return df