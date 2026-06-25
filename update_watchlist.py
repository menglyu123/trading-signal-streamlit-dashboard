#!/usr/bin/env python3
"""
Standalone script to update watchlist with buy signals.
Can be run independently from the dashboard.

Usage:
    python update_watchlist.py --market HK
    python update_watchlist.py --market US
"""

import os
import sys
import argparse
import pandas as pd
from auto_trade import auto_trade
from data.import_data import Market, download_futu_historical_daily_data, download_alpaca_daily_data
import datetime as dt

WATCHLIST_DIR = "./data"

def get_watchlist_path(market: str) -> str:
    """Build a market-specific watchlist path."""
    filename = f"watchlist_{market.lower()}.txt"
    return os.path.join(WATCHLIST_DIR, filename)

def load_watchlist(market: str):
    """Load existing watchlist."""
    path = get_watchlist_path(market)
    if not os.path.exists(path):
        return [], path
    with open(path, "r") as fh:
        tickers = [line.strip() for line in fh.readlines() if line.strip()]
    return tickers, path

def save_watchlist(market: str, tickers):
    """Save watchlist to disk."""
    path = get_watchlist_path(market)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(tickers))
    print(f"✓ Watchlist saved: {path}")
    print(f"  Total stocks: {len(tickers)}")
    return path

def update_watchlist_with_buy_signals(trader, predictions_pool, predictions_df, market):
    """Update watchlist with all stocks that have buy signals."""
    buy_signal_codes = []
    
    if predictions_df is None or predictions_df.empty:
        print("No predictions available")
        return buy_signal_codes
    
    # Get buy signals based on multiple criteria
    for ticker in predictions_df.index:
        try:
            ticker_pred_df = predictions_pool.get(ticker)
            if ticker_pred_df is None or ticker_pred_df.empty:
                continue
            
            # Get latest data point
            latest = ticker_pred_df.iloc[-1]
            
            # Criteria 1: prediction > 0.5
            has_positive_prediction = latest.get('prediction', 0) > 0.5
            
            # Criteria 2: down_deep condition (EMA_60 / close > 1.1)
            ema_60 = latest.get('EMA_60', 0)
            close_price = latest.get('close', 0)
            is_down_deep = (close_price > 0) and (ema_60 / close_price > 1.1)
            
            # Criteria 3: Strong uptrend (up_strength >= 0.8 and uptrend = True)
            up_strength = latest.get('up_strength', 0)
            uptrend = latest.get('uptrend', False)
            has_strong_uptrend = (up_strength >= 0.8) and (uptrend == True)
            
            # Combine criteria: any strong signal qualifies
            if has_positive_prediction or is_down_deep or has_strong_uptrend:
                buy_signal_codes.append(ticker)
        except Exception as e:
            continue
    
    # Remove duplicates while preserving order
    buy_signal_codes = list(dict.fromkeys(buy_signal_codes))
    
    return buy_signal_codes

def get_batch_predictions(trader):
    """Get predictions for all stocks."""
    from_date = pd.Timestamp.now().date()
    current_date = from_date
    start_date = current_date - dt.timedelta(days=trader.winlen+60+20+500)
    
    data = {}
    
    print(f"Downloading data for {len(trader.code_list)} stocks...")
    
    # Download US market data
    if trader.market == Market.US.name:
        for code in trader.code_list:
            data[code] = download_alpaca_daily_data(code, start_date, current_date)
    
    # Download HK market data
    if trader.market == Market.HK.name:
        import time
        for i, ticker in enumerate(trader.code_list):
            download_ticker = 'HK.0' + ticker
            if i % 60 == 0:
                tick = time.time()
            data[ticker] = download_futu_historical_daily_data(download_ticker, start_date, current_date)
            time_cost = time.time() - tick
            if ((i + 1) % 60 == 0) and (time_cost <= 30):
                time.sleep(31 - time_cost)
            if (i + 1) % 10 == 0:
                print(f"  Downloaded {i + 1}/{len(trader.code_list)} stocks")
    
    preds_df_comb = {}
    for ticker in trader.code_list:
        df = data[ticker].copy()
        pred_df = trader.add_signal_cols(df)
        if len(pred_df) < 3:
            continue
        preds_df_comb[ticker] = pred_df
    
    return preds_df_comb

def get_last_n_trading_days_predictions(n, preds_set):
    """Get predictions for last n trading days."""
    preds_df_comb = []
    for ticker, pred_df in preds_set.items():
        pred_df['code'] = ticker
        preds_df_comb.append(pred_df)
    
    if not preds_df_comb:
        return [], []
    
    comb = pd.concat(preds_df_comb, axis=0)
    predictions_list = []
    dates_list = []
    for date in comb.date.unique():
        pred_df = comb[comb.date == date]
        pred_df.set_index('code', inplace=True)
        predictions_list.append(pred_df)
        dates_list.append(date)
    
    return predictions_list[-n:], dates_list[-n:]

def main():
    parser = argparse.ArgumentParser(description='Update watchlist with buy signals')
    parser.add_argument('--market', choices=['HK', 'US'], default='HK', help='Market to update (HK or US)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Updating {args.market} Stock Watchlist with Buy Signals")
    print(f"{'='*60}\n")
    
    # Initialize trader
    trader = auto_trade(args.market)
    print(f"Total stocks in {args.market} market: {len(trader.code_list)}")
    
    # Get predictions
    print("\nFetching predictions for all stocks...")
    preds_pool = get_batch_predictions(trader)
    print(f"✓ Got predictions for {len(preds_pool)} stocks")
    
    # Get latest predictions
    predictions_list, dates_list = get_last_n_trading_days_predictions(1, preds_pool)
    
    if not predictions_list:
        print("✗ Could not fetch predictions")
        return
    
    predictions_df = predictions_list[0]
    print(f"Latest prediction date: {dates_list[0]}")
    
    # Update watchlist
    print("\nIdentifying buy signals...")
    buy_signals = update_watchlist_with_buy_signals(trader, preds_pool, predictions_df, args.market)
    
    if buy_signals:
        print(f"✓ Found {len(buy_signals)} stocks with buy signals")
        print(f"\nBuy Signal Stocks ({args.market}):")
        for code in buy_signals:
            print(f"  - {code}")
        
        # Save watchlist
        save_watchlist(args.market, buy_signals)
        print(f"\n{'='*60}")
        print("✓ Watchlist updated successfully!")
        print(f"{'='*60}\n")
    else:
        print("✗ No buy signals found")

if __name__ == '__main__':
    main()
