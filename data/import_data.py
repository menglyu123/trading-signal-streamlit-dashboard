import datetime as dt
import yfinance as yf
import requests
import time
import random
import pandas as pd

with open('./data/code_pool_hk.txt','r') as fp:
    CODE_LIST = [line.rstrip() for line in fp]

def download_with_retry(tickers, start_date, end_date, max_retries=3):
    """
    Download data with retry logic and exponential backoff
    """
    for attempt in range(max_retries):
        try:
            # Let Yahoo Finance handle session management
            data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date + dt.timedelta(days=1),
                group_by='ticker',
                progress=False
            )
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff: wait 2^attempt seconds
                wait_time = 2 ** attempt + random.uniform(0, 1)
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Final attempt failed: {e}")
                raise e

def download_yf_data(code_list, start_date=None, end_date=None, chunk_size=5):
    """
    Download stock data from Yahoo Finance for a list of stocks with chunked processing
    Args:
        code_list: list of stock codes
        start_date: start date (datetime.date or string YYYY-MM-DD)
        end_date: end date (datetime.date or string YYYY-MM-DD), defaults to today if None
        chunk_size: number of tickers to download at once (default: 5)
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
    
    # Split code_list into chunks to avoid rate limiting
    chunks = [code_list[i:i + chunk_size] for i in range(0, len(code_list), chunk_size)]
    
    all_data = []
    
    for i, chunk in enumerate(chunks):
        try:
            print(f"Downloading chunk {i+1}/{len(chunks)} ({len(chunk)} tickers)")
            
            # Download data for this chunk with retry logic
            data = download_with_retry(chunk, start_date, end_date)
            
            if data is not None and not data.empty:
                all_data.append(data)
                print(i, chunk, data)
                print(f"✓ Successfully downloaded chunk {i+1}")
            else:
                print(f"⚠ Warning: Chunk {i+1} returned empty data")
            
            # Add random delay between chunks (1-3 seconds)
            if i < len(chunks) - 1:  # Don't delay after the last chunk
                delay = random.uniform(1.0, 3.0)
                print(f"Waiting {delay:.1f} seconds before next chunk...")
                time.sleep(delay)
                
        except Exception as e:
            print(f"✗ Error downloading chunk {i+1}: {e}")
            # Continue with next chunk instead of failing completely
            continue
    
    # Combine all downloaded data
    if all_data:
        try:
            combined_data = pd.concat(all_data, axis=1)
            print(f"✓ Successfully combined data from {len(all_data)} chunks")
            return combined_data
        except Exception as e:
            print(f"✗ Error combining data: {e}")
            return None
    else:
        print("✗ No data was successfully downloaded")
        return None  