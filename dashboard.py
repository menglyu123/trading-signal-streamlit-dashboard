import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
from auto_trade import auto_trade
import streamlit as st
# Must be the first Streamlit command
st.set_page_config(layout="wide", page_title="HK Stock Market Predictions")
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import io, time
import numpy as np
from data.import_data import download_futu_historical_daily_data, download_yf_data
from util import get_sector


def get_last_n_trading_days_predictions(n, trader, from_date=None):
    """
    Get predictions for the last n trading days
    Returns a tuple of (predictions_list, dates_list)
    """
    if from_date is None:
        from_date = pd.Timestamp.now().date()
    
    preds_df_comb = []
    current_date = pd.Timestamp(from_date)

    # Download data once for all dates with enough history
    start_date = current_date - dt.timedelta(days=trader.winlen+60+90+n+40)

    data = {}

    # Download US market data
    if trader.market == 'US':      
        rs = download_yf_data(trader.code_list, start_date, current_date)
        tickers = set(rs.columns.get_level_values(0))
        for ticker in tickers:
            df = rs[ticker].reset_index()
            df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close', 'Volume':'volume'},inplace=True)
            data[ticker] = df
        
    # Download HK market data
    if trader.market == 'HK':
        for i, ticker in enumerate(trader.code_list):
            # download_ticker = ticker
            # if trader.market == 'HK':
            download_ticker = 'HK.0'+ticker
            # if trader.market == 'US':
            #     download_ticker = 'US.'+ticker
            if i%60 == 0:
                tick = time.time()
            data[ticker] = download_futu_historical_daily_data(download_ticker, start_date, current_date)
            time_cost = time.time()-tick
            if ((i+1)%60==0) & (time_cost<=30):
                time.sleep(31-time_cost)
    if data is None:
        return [], []
    
    for ticker in trader.code_list:
        # Get data up to current date
        print(f"predict {ticker}")
        df = data[ticker].copy()
        pred_df = trader.add_signal_cols(df)
        if len(pred_df) < 3:
            print("error ticker", ticker)
            continue
        pred_df['code'] = ticker
        preds_df_comb.append(pred_df)
    
    comb = pd.concat(preds_df_comb, axis=0)
    predictions_list = []
    dates_list = []
    for date in comb.date.unique():
        pred_df = comb[comb.date==date]
        pred_df.set_index('code', inplace=True)
        predictions_list.append(pred_df)
        dates_list.append(date)
    return predictions_list[-n:], dates_list[-n:]

def plot_single(code, bdf):
    """
    Create a plot and return it as a bytes buffer
    """
    plt.figure(figsize=(10,4))
    plt.plot(bdf.date, bdf.close, label='Close')
    plt.plot(bdf.date, bdf.EMA_10, label='EMA_10')
    plt.plot(bdf.date, bdf.EMA_20, label='EMA_20')
    plt.plot(bdf.date, bdf.EMA_60, label='EMA_60')
    bdf['accel'] = np.exp(bdf.prediction)
    bdf['accel'] = bdf.accel.pct_change()
    bdf.accel.fillna(0, inplace=True)

    pred_mask = [True if p>0 else False for p in bdf.prediction]
    accel_mask = [True if p>0 else False for p in bdf.accel]
    plt.scatter(bdf[pred_mask]['date'], bdf[pred_mask]['close'], s=30*bdf[pred_mask]['prediction'], c='blue', marker='o', label='Predict')
    plt.scatter(bdf[accel_mask]['date'], bdf[accel_mask]['close'], s=40*bdf[accel_mask]['accel'], c='red', marker='+', label='accel')
 
    plt.title(f'{code}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf

def plot_prediction_histogram(predictions_list, dates_list):
    """
    Plot histograms of predictions for the last 3 trading days
    Args:
        predictions_list: list of prediction DataFrames
        dates_list: list of corresponding dates
    Returns:
        buf: image buffer containing the plot
    """
    if len(predictions_list) < 3 or len(dates_list) < 3:
        return None
        
    # Get the global min and max for consistent bins
    all_preds = []
    for pred_df in predictions_list[:3]:  # Use only the first 3 days
        all_preds.extend(pred_df['prediction'].values)
    bins = np.linspace(min(all_preds), max(all_preds), 30)
    
    # Create figure with three subplots in one row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Colors for each day
    colors = ['red', 'green', 'blue']
    dark_colors = ['darkred', 'darkgreen', 'darkblue']
    
    # Plot for each of the last 3 days
    for i, (pred_df, date, ax, color, dark_color) in enumerate(zip(predictions_list[:3], dates_list[:3], [ax1, ax2, ax3], colors, dark_colors)):
        ax.hist(pred_df['prediction'], 
                bins=bins,
                color=color,
                alpha=0.6,
                edgecolor='black',
                linewidth=1)
        mean_val = pred_df['prediction'].mean()
        ax.axvline(x=mean_val, 
                    color=dark_color,
                    linestyle='--',
                    label=f'Mean: {mean_val:.3f}')
        ax.set_title(f'{date}')
        ax.set_xlabel('Prediction Value')
        ax.set_ylabel('Number of Stocks')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Distribution of Predictions Across All {market_choice} Stocks (Last 3 Trading Days)')
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf

def build_sector_distribution_chart(predictions_df):
    """
    Build a bar chart showing average prediction by sector for HK market.
    Returns matplotlib figure buffer or None if unavailable.
    """
    if predictions_df is None or predictions_df.empty:
        return None
    try:
        sector_df = get_sector(None)
        if sector_df is None or sector_df.empty:
            return None
        merged = (
            predictions_df.reset_index()
            .merge(sector_df, on='code', how='inner')
        )
        if merged.empty or 'sector' not in merged.columns:
            return None
        sector_avg = (
            merged.groupby('sector')['prediction']
            .mean()
            .sort_values(ascending=True)
        )
        if sector_avg.empty:
            return None
        fig, ax = plt.subplots(figsize=(8, max(4, len(sector_avg) * 0.35)))
        colors = ['green' if val >= 0 else 'red' for val in sector_avg.values]
        ax.barh(sector_avg.index, sector_avg.values, color=colors, alpha=0.75)
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xlabel('Average Prediction Value')
        ax.set_ylabel('Sector')
        ax.set_title('Average Prediction by Sector (Latest Day)')
        for idx, val in enumerate(sector_avg.values):
            offset = 0.01 if val >= 0 else -0.01
            ax.text(val + offset, idx, f'{val:.3f}',
                    va='center',
                    ha='left' if val >= 0 else 'right',
                    fontsize=6)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Sector chart error: {e}")
        return None

# Initialize Signal_Model
@st.cache_resource
def get_trader_hk():
    return auto_trade('HK')

@st.cache_resource
def get_trader_us():
    return auto_trade('US')

# Market selection
market_choice = st.selectbox(
    "Select Market",
    options=['HK', 'US'],
    index=0,
    help="Choose between Hong Kong and US stock markets"
)

# Get the appropriate trader based on market choice
if market_choice == 'HK':
    trader = get_trader_hk()
if market_choice =='US':
    trader = get_trader_us()


# Check if trader is available
if trader is None:
    st.error("Unable to initialize trading model. Please check the logs for more details.")
    st.stop()

# Initialize session state
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None
if 'predictions_df_yesterday' not in st.session_state:
    st.session_state.predictions_df_yesterday = None
if 'predictions_df_2days_ago' not in st.session_state:
    st.session_state.predictions_df_2days_ago = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'histogram_image' not in st.session_state:
    st.session_state.histogram_image = None
if 'update_clicked' not in st.session_state:
    st.session_state.update_clicked = False

st.title(f"{market_choice} Stock Market Predictions Dashboard")

# Market Predictions Section
st.header("Market-wide Predictions")

col1, col2 = st.columns([2, 1])
with col1:
    if st.button("Update Market Predictions", key="update_button"):
        st.session_state.update_clicked = True
        
    if st.session_state.update_clicked:
        with st.spinner("Fetching predictions for all stocks..."):
            # Get predictions for last 3 trading days
            predictions_list, dates_list = get_last_n_trading_days_predictions(3, trader)
            
            if len(predictions_list) == 3:
                st.session_state.predictions_df = predictions_list[2]
                st.session_state.predictions_df_yesterday = predictions_list[1]
                st.session_state.predictions_df_2days_ago = predictions_list[0]
                st.session_state.last_update = dt.datetime.now()
                
                # Update the plot titles with actual dates
                st.session_state.histogram_image = plot_prediction_histogram(
                    predictions_list,
                    dates_list
                )
            else:
                st.error("Could not fetch predictions for all three trading days. Please try again later.")
            
            st.session_state.update_clicked = False

    # Always display histogram if it exists
    if st.session_state.histogram_image is not None:
        st.image(st.session_state.histogram_image)
    else:
        st.info("Distribution histogram will appear after predictions are fetched.")

    # Sector distribution for HK market
    if market_choice == 'HK':
        if st.session_state.predictions_df is not None:
            sector_chart = build_sector_distribution_chart(st.session_state.predictions_df)
            if sector_chart is not None:
                st.image(sector_chart)
            else:
                st.info("Unable to display sector distribution (requires Futu API data).")
        else:
            st.info("Sector distribution will appear after predictions are fetched.")

with col2:
    if st.session_state.last_update:
        st.info(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("Click 'Update Market Predictions' to fetch data")

if st.session_state.predictions_df is not None:
   # col1, col2 = st.columns([1,1])
    tmp_df = st.session_state.predictions_df.copy()
    with col1:
        st.subheader("Top 10 Bullish Predictions")
        tmp_df['bottom_strength'] = tmp_df['btm_strength']*tmp_df.uptrend.astype(int)
        top_10 = tmp_df.sort_values('bottom_strength', ascending=False).head(10)
        st.dataframe(top_10[['bottom_strength', 'prediction', 'accel','rmse_10','dist_avgs','close']])
    # with col2:
    #     st.subheader("Top 10 Bearish Predictions")
    #     top_10 = tmp_df.sort_values('top_collapse', ascending=False).head(10)
    #     st.dataframe(top_10[['btm_strength','top_collapse','accel', 'prediction', 'close']])


# Individual Stock Analysis Section
st.header("Individual Stock Analysis")

# Only fetch predictions if we haven't already
if 'predictions_df' not in st.session_state or st.session_state.predictions_df is None:
    st.info("Please click 'Update Market Predictions' to analyze individual stocks.")
else:
    # Create sorted options for the dropdown
    predictions_sorted = st.session_state.predictions_df.sort_values('prediction', ascending=False)
    stock_options = []
    for code in predictions_sorted.index:
        pred_value = predictions_sorted.loc[code, 'prediction']
        # Format the display string with stock code and prediction value only
        display_string = f"{code} (Pred: {pred_value:.3f})"
        stock_options.append(display_string)
    
    # Allow multiple stock selection with formatted options
    selected_stocks = st.multiselect(
        "Select stocks to analyze (sorted by prediction value)",
        options=stock_options,
        help="Stocks are sorted by prediction value (highest to lowest)"
    )
    
    if selected_stocks:  # Only show slider if stocks are selected
        days = st.slider("Analysis Period (Days)", min_value=30, max_value=500, value=100, step=10)
        
        # Extract stock codes from the selected options
        selected_tickers = [stock.split()[0] for stock in selected_stocks]
        
        # Download data for all selected stocks at once
        today = dt.date.today()
        start_date = today - dt.timedelta(days=trader.winlen+60+20+days)
        
        with st.spinner(f"Fetching data for selected stocks..."):
            all_stock_data = {}
            # Download US market data using yfinance
            if market_choice == 'US':
                rs = download_yf_data(selected_tickers, start_date, today)
                tickers = set(rs.columns.get_level_values(0))
                for ticker in tickers:
                    df = rs[ticker].reset_index()
                    df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close', 'Volume':'volume'},inplace=True)
                    all_stock_data[ticker] = df

            # Download HK market data using futu
            if market_choice == 'HK':
                # rs = download_yf_data(['^HSI'], start_date, today)
                # df = rs['^HSI'].reset_index()
                # df.loc[len(df)-1,'Volume'] = df.iloc[-2]['Volume']
                # df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close', 'Volume':'volume'},inplace=True)
                # all_stock_data['HSI (Hang Sheng Index)'] = df
                for i, ticker in enumerate(selected_tickers):
                    download_ticker = 'HK.0'+ticker
                    # if market_choice == 'US':
                    #     download_ticker = 'US.'+ticker
                    if i%60 == 0:
                        tick = time.time()
                    all_stock_data[ticker] = download_futu_historical_daily_data(download_ticker, start_date, today)
                    time_cost = time.time()-tick
                    if ((i+1)%60==0) & (time_cost<=30):
                        time.sleep(31-time_cost)
        
        if all_stock_data:
            for stock in all_stock_data.keys():
                st.subheader(f"Stock: {stock}")
                
                df = all_stock_data.get(stock)
                if df is None or df.empty:
                    st.error(f"Error fetching data for stock {stock}")
                else:
                    # Run backtest_single
                    result_df = trader.backtest_single(df)
                    
                    if result_df is not None:
                        # Create and display plot directly from memory
                        plot_buf = plot_single(stock, result_df)
                        st.image(plot_buf)
                    else:
                        st.error("Failed to generate analysis for this stock.")
        else:
            st.error("Failed to fetch data for selected stocks. Please try again later.") 