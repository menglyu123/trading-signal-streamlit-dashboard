import datetime as dt
import yfinance as yf
import requests

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
            progress=False
        )
        return data
    except Exception as e:
        print(f"Error downloading batch data: {e}")
        return None  