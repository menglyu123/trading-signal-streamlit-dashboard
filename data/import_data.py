import pandas as pd
import datetime as dt, time
import yfinance as yf
from futu import *

with open('./data/code_pool_hk.txt','r') as fp:
    CODE_LIST = [line.rstrip() for line in fp]



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
    from curl_cffi import requests
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
  
def download_today_capital_flow(tickers):
    data = []
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    for ticker in tickers:
        ret, tmp = quote_ctx.get_capital_flow('HK.0'+ticker, period_type = PeriodType.INTRADAY) # DAY, WEEK, MONTH
        if ret == RET_OK:
            tmp = tmp[['capital_flow_item_time','in_flow','super_in_flow','big_in_flow','mid_in_flow','sml_in_flow']]
            tmp.rename(columns={'capital_flow_item_time':'time'}, inplace=True)
            tmp['time'] = pd.to_datetime(tmp.time)
            data.append(tmp)   
            time.sleep(1)  # Interface limitations: A maximum of 30 requests per 30 seconds
        else:
            print("error: ",tmp)
    quote_ctx.close() # After using the connection, remember to close it to prevent the number of connections from running out
    return data