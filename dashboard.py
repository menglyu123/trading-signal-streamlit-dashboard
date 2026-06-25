import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
from auto_trade import auto_trade
import streamlit as st
# Must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Stock Market Predictions")
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import io, time
import numpy as np
from data.import_data import Market, download_futu_historical_daily_data, download_alpaca_daily_data
import futu

WATCHLIST_DIR =  "./data"

def get_current_portfolio(market:str):
    #trd_ctx = futu.OpenSecTradeContext(filter_trdmarket=market, host='127.0.0.1', port=11111, security_firm=futu.SecurityFirm.FUTUSECURITIES)
    with futu.OpenSecTradeContext(
        filter_trdmarket=market,
        host='127.0.0.1',
        port=11111,
        security_firm=futu.SecurityFirm.FUTUSECURITIES
    ) as trd_ctx:
        ret, positions_df = trd_ctx.position_list_query()
        if ret == futu.RET_OK and positions_df is not None and positions_df.shape[0] > 0:
            return positions_df['code'].astype(str).tolist()
        if ret != futu.RET_OK:
            print('position_list_query error: ', positions_df)
    return []

def _trend_flags(result_df: pd.DataFrame, period: int):
    """
    EMA-based trend logic on a single ticker.
    """
    if result_df is None or result_df.empty:
        return False
    if len(result_df) < max(3, period + 2):
        return False

    ema5_change = result_df["EMA_5"].pct_change(1).iloc[-period:]
    is_ema5_mostly_up = (ema5_change.lt(0).sum() <= round(period * 0.3))
    is_ema5_mostly_greater_ema10 = (result_df.iloc[-period:]["EMA_5"] < result_df.iloc[-period:]["EMA_10"]).sum() <= round(period * 0.2)
    is_ema10_above_ema20 = bool(result_df.iloc[-1]["EMA_10"] > result_df.iloc[-1]["EMA_20"])
    is_ema20_above_ema60 = bool(result_df.iloc[-1]["EMA_20"] > result_df.iloc[-1]["EMA_60"])

    return bool(
        is_ema5_mostly_up
        and is_ema5_mostly_greater_ema10
        and is_ema10_above_ema20
        and is_ema20_above_ema60
    )

def get_watchlist_path(market: str) -> str:
    """
    Build a market-specific watchlist path (e.g., watchlist_hk.txt).
    """
    filename = f"watchlist_{market.lower()}.txt"
    return os.path.join(WATCHLIST_DIR, filename)

def load_watchlist(market: str):
    """
    Load watchlist tickers for a market. Returns (list, path).
    """
    path = get_watchlist_path(market)
    if not os.path.exists(path):
        return [], path
    with open(path, "r") as fh:
        tickers = [line.strip() for line in fh.readlines() if line.strip()]
    return tickers, path

def save_watchlist(market: str, tickers):
    """
    Persist watchlist tickers to disk and return the path.
    """
    path = get_watchlist_path(market)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(tickers))
    return path

def update_watchlist_with_buy_signals(trader, predictions_pool, predictions_df, market):
    """
    Update watchlist with all stocks that have buy signals.
    Buy signals are identified by:
    1. prediction > 0.5
    2. EMA_60 / close > 1.1 (down_deep condition)
    3. Strong up_strength (>= 0.8)
    """
    buy_signal_codes = []
    
    if predictions_df is None or predictions_df.empty:
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
    
    # Save to watchlist
    if buy_signal_codes:
        save_watchlist(market, buy_signal_codes)
    
    return buy_signal_codes

def get_batch_predictions(trader, from_date=None)->dict:
    """
    Get predictions given the last 500 trading days market data
    Returns a tuple of (predictions_list, dates_list)
    """
    if from_date is None:
        from_date = pd.Timestamp.now().date()
    
    preds_df_comb = []
    current_date = pd.Timestamp(from_date).date()

    # Download data once for all dates with enough history
    start_date = current_date - dt.timedelta(days=trader.winlen+60+20+500)

    data = {}

    # Download US market data
    if trader.market == Market.US.name:  
        for code in trader.code_list:  
            data[code] = download_alpaca_daily_data(code, start_date, current_date)
        
    # Download HK market data
    if trader.market == Market.HK.name:
        for i, ticker in enumerate(trader.code_list):
            print(f'download {ticker}')
            download_ticker = 'HK.0'+ticker
            if i%60 == 0:
                tick = time.time()
            data[ticker] = download_futu_historical_daily_data(download_ticker, start_date, current_date)
            time_cost = time.time()-tick
            if ((i+1)%60==0) & (time_cost<=30):
                time.sleep(31-time_cost)
    if data is None:
        return [], []
    
    preds_df_comb = {}
    for ticker in trader.code_list:
        # Get data up to current date
        print(f"predict {ticker}")
        df = data[ticker].copy()
        pred_df = trader.add_signal_cols(df)
        if len(pred_df) < 3:
            print("error ticker", ticker)
            continue
        preds_df_comb[ticker] = pred_df
    return preds_df_comb

def get_last_n_trading_days_predictions(n, preds_set:dict):
    preds_df_comb = []
    for ticker, pred_df in preds_set.items():
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
    plt.plot(bdf.date, bdf.EMA_5, label='EMA_5')
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

def show_batch_plots(stock_code_list, sector_df, with_lookback_slider=True):
    today = dt.date.today()
    # Lookback period slider
    if with_lookback_slider:
        days = st.slider(
            "Period (Days)",
            min_value=100,
            max_value=500,
            value=300,
            step=100,
            key="look back days"
        )
        start_date = today - dt.timedelta(days=days)
    else:
        start_date = today - dt.timedelta(days=500)

    # Plot each stock
    for stock_code in stock_code_list:
        display_name = stock_code
        # Prefer name from sector_df (it should always have names)
        try:
            if 'name' in sector_df.columns:
                name_series = sector_df.loc[sector_df['code'] == stock_code, 'name']
                if not name_series.empty:
                    name_val = name_series.iloc[0]
                    if pd.notna(name_val) and str(name_val).strip():
                        display_name = f"{stock_code} - {name_val}"
        except Exception:
            pass
        
        ticker_pred_df = st.session_state.predictions_pool[stock_code]
        df = ticker_pred_df[ticker_pred_df.date>=start_date]
    
        if df is None or df.empty:
            st.warning(f"Could not find predictions for {stock_code}.")
            continue
        
        # Collapsible plot so you can "close" it
        with st.expander(f"{display_name}", expanded=False):
            plot_buf = plot_single(stock_code, df)
            st.image(plot_buf)

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
    min_count_value, max_count_value = 0, 0
    
    # Plot for each of the last 3 days
    for i, (pred_df, date, ax, color, dark_color) in enumerate(zip(predictions_list[:3], dates_list[:3], [ax1, ax2, ax3], colors, dark_colors)):
        counts, _, _ = ax.hist(pred_df['prediction'], 
                bins=bins,
                color=color,
                alpha=0.6,
                edgecolor='black',
                linewidth=1)
        min_count_value = min(min_count_value, min(counts))
        max_count_value = max(max_count_value, max(counts))
        mean_val = pred_df['prediction'].mean()
        ax.axvline(x=mean_val, 
                    color=dark_color,
                    linestyle='--',
                    label=f'Mean: {mean_val:.3f}')
        ax.legend(fontsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16) 
        ax.set_title(f'{date}', fontsize = 16)
        ax.grid(True, alpha=0.3)
    ax1.set_ylim(min_count_value, max_count_value)
    ax2.set_ylim(min_count_value, max_count_value)
    ax3.set_ylim(min_count_value, max_count_value)
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax3.set_xlabel('')
    ax2.set_ylabel('')
    ax3.set_ylabel('')
    ax1.set_ylabel('Number of Stocks', fontsize=16)
    plt.suptitle('Distribution of Prediction Values Across All Stocks (Last 3 Trading Days)', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf

def build_sector_distribution_three_days(predictions_df, predictions_df_yesterday, predictions_df_2days_ago, dates_list, market):
    """
    Build three bar charts side by side showing average prediction by sector for the last 3 days.
    Returns matplotlib figure buffer or None if unavailable.
    """
    if predictions_df is None or predictions_df.empty:
        return None
    try:
        sector_df = Market(market).get_code_pool()["code_sector_df"]
        if sector_df is None or sector_df.empty:
            return None
        
        # Get all unique sectors from all three days
        all_sectors = set()
        sector_data = {}
        
        for df, date in zip([predictions_df_2days_ago, predictions_df_yesterday, predictions_df], 
                           dates_list if len(dates_list) >= 3 else [None, None, None]):
            if df is None or df.empty:
                continue
            merged = df.reset_index().merge(sector_df, on='code', how='inner')
            if merged.empty or 'sector' not in merged.columns:
                continue
            sector_avg = merged.groupby('sector')['prediction'].mean()
            sector_data[date] = sector_avg
            all_sectors.update(sector_avg.index)
        
        if not all_sectors:
            return None
        
        # Sort sectors by average value from the latest day
        sector_count = {}
        if predictions_df is not None and not predictions_df.empty:
            latest_merged = predictions_df.reset_index().merge(sector_df, on='code', how='inner')
            if not latest_merged.empty:
                latest_sector_avg = latest_merged.groupby('sector')['prediction'].mean()
                latest_sector_count = latest_merged.groupby('sector')['prediction'].count()
                sector_count = latest_sector_count.to_dict()
                sorted_sectors = latest_sector_avg.sort_values(ascending=True).index.tolist()
            else:
                sorted_sectors = sorted(all_sectors)
        else:
            sorted_sectors = sorted(all_sectors)
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,5))
        
        # Plot for each day
        dates = dates_list if len(dates_list) >= 3 else [None, None, None]
        axes = [ax1, ax2, ax3]
        data_list = [predictions_df_2days_ago, predictions_df_yesterday, predictions_df]
        min_sector_value, max_sector_value = 0, 0
        prev_sector_values = [0]*len(sorted_sectors)
        for ax, df, date in zip(axes, data_list, dates):
            if df is None or df.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'No Data' if date is None else f'{date}')
                ax.set_xlabel('Average Prediction Value', fontsize=16)
                continue
            
            merged = df.reset_index().merge(sector_df, on='code', how='inner')
            if merged.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'No Data' if date is None else f'{date}')
                ax.set_xlabel('Average Prediction Value', fontsize=16)
                continue
            
            sector_avg = merged.groupby('sector')['prediction'].mean()
            
            # Align sectors with sorted order
            sector_values = [sector_avg.get(s, 0) for s in sorted_sectors]
            diff = [val-prev_val for val, prev_val in zip(sector_values, prev_sector_values)]
            prev_sector_values = sector_values
            min_sector_value = min(min_sector_value, min(sector_values))
            max_sector_value = max(max_sector_value, max(sector_values))
            colors = ['green' if val >= 0 else 'gray' for val in sector_values]
            ax.barh(range(len(sorted_sectors)), sector_values, color=colors, alpha=0.75)
            ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)  
            ax.tick_params(axis='x', labelsize=16)  # Change x-ticker size
            ax.tick_params(axis='y', labelsize=16)  # Change y-ticker size
            ax.set_title(f'{date}' if date else 'Average Prediction by Sector', fontsize = 16)
            
            # Add value labels
            if date == dates[-1]:
                for idx, val in enumerate(sector_values):
                    if val != 0:  # Only label non-zero values
                        offset = 0.01 if val >= 0 else -0.01
                        diff_show = '+'+ str(diff[idx].round(2)) if diff[idx]>0 else '-'+ str(-diff[idx].round(2))
                        ax.text(val + offset, idx, f'{val:.3f} ({diff_show})',
                            va='center',
                            ha='left' if val >= 0 else 'right',
                            fontsize=16)  
            else:
                for idx, val in enumerate(sector_values):
                    if val != 0:  # Only label non-zero values
                        offset = 0.01 if val >= 0 else -0.01
                        ax.text(val + offset, idx, f'{val:.3f}',
                               va='center',
                               ha='left' if val >= 0 else 'right',
                               fontsize=16)

        # Only show y-axis labels on the leftmost plot
           # Set the x-limits based on min and max values
        ax1.set_xlim(min_sector_value, max_sector_value)
        ax2.set_xlim(min_sector_value, max_sector_value)
        ax3.set_xlim(min_sector_value, max_sector_value)
        ax1.set_yticks(range(len(sorted_sectors)))
        ax1.set_yticklabels([sec+f' ({sector_count[sec]})' for sec in sorted_sectors], fontsize=16)
        ax2.set_yticklabels('')
        ax3.set_yticklabels('')
        ax2.set_ylabel('')
        ax3.set_ylabel('')
        ax1.set_ylabel('')
        ax2.set_xlabel('')
        ax3.set_xlabel('')
        ax1.set_xlabel('')
        
        plt.suptitle('Sector Distribution - Average Prediction Values (Last 3 Trading Days)', 
                    fontsize=18, fontweight='bold')
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
    return auto_trade(Market.HK.name)

@st.cache_resource
def get_trader_us():
    return auto_trade(Market.US.name)

### *********************** Market selection *********************** 
market_choice = st.selectbox(
    "Select Market",
    options=['HK', 'US'],
    index=0,
    help="Choose between Hong Kong and US stock markets"
)

# Get the appropriate trader based on market choice
if market_choice == Market.HK.name:
    trader = get_trader_hk()
if market_choice == Market.US.name:
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
if 'sector_chart' not in st.session_state:
    st.session_state.sector_chart = None
if 'index_prediction_plot' not in st.session_state:
    st.session_state.index_prediction_plot = None
if 'update_clicked' not in st.session_state:
    st.session_state.update_clicked = False
if 'dates_list' not in st.session_state:
    st.session_state.dates_list = None
if 'index_prediction_df' not in st.session_state:
    st.session_state.index_prediction_df = None

st.title(f"{market_choice} Stock Market Analysis Dashboard")



### *********************** Market Predictions Section *********************** 
st.header("Market-wide Predictions")

col1, col2 = st.columns([4, 1])
with col1:
    if st.button("Update Market Predictions", key="update_button"):
        st.session_state.update_clicked = True
        
    if st.session_state.update_clicked:
        with st.spinner("Fetching predictions for all stocks..."):
            # Get predictions for last 3 trading days
            preds_comb_df = get_batch_predictions(trader)
            predictions_list, dates_list = get_last_n_trading_days_predictions(3, preds_comb_df)            
            st.session_state.predictions_pool = preds_comb_df
            
            if len(predictions_list) == 3:
                st.session_state.predictions_df = predictions_list[2]
                st.session_state.predictions_df_yesterday = predictions_list[1]
                st.session_state.predictions_df_2days_ago = predictions_list[0]
                st.session_state.dates_list = dates_list
                st.session_state.last_update = dt.datetime.now()
                
                # Update the prediction histograms
                st.session_state.histogram_image = plot_prediction_histogram(
                    predictions_list,
                    dates_list
                )
                st.session_state.sector_chart = build_sector_distribution_three_days(
                        st.session_state.predictions_df,
                        st.session_state.predictions_df_yesterday,
                        st.session_state.predictions_df_2days_ago,
                        st.session_state.dates_list,
                        market_choice
                    )
            else:
                st.error("Could not fetch predictions for all three trading days. Please try again later.")
            
            current_date = dt.date.today()
            start_date = current_date - dt.timedelta(days=trader.winlen+60+20+500)
            # Download US market data
            if trader.market == Market.US.name:  
                code = 'SPY'
                df = download_alpaca_daily_data(code, start_date, current_date)
            # Download HK market data
            if trader.market == Market.HK.name:
                code = 'HSI'
                df = download_futu_historical_daily_data('HK.800000', start_date, current_date)
            bdf = trader.add_signal_cols(df)
            st.session_state.index_prediction_df = bdf
            st.session_state.index_prediction_plot = plot_single(code, bdf)
            st.session_state.update_clicked = False
with col2:
    if st.session_state.last_update:
        st.info(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("Click 'Update Market Predictions' to make prediction")

# Display prediction histograms
if st.session_state.histogram_image is not None:
    st.image(st.session_state.histogram_image)

# Display sector charts
if st.session_state.sector_chart is not None:
    st.image(st.session_state.sector_chart)

# Display sector charts
if st.session_state.index_prediction_plot is not None:
    st.image(st.session_state.index_prediction_plot)

# Display top 20 breakthrough tickers
if st.session_state.predictions_df is not None:
    col1, col2 = st.columns([1,1])
    tmp_df = st.session_state.predictions_df.copy()
    sector_df = Market(market_choice).get_code_pool()["code_sector_df"]

    sector_df.set_index('code', inplace=True)

    with col1:
        st.subheader("Top 20 Breakthrough")
        tmp_df['bottom_strength'] = tmp_df['up_strength']*tmp_df.uptrend.astype(int)
        top_n = tmp_df.sort_values(['bottom_strength','dist_avgs'], ascending=[False, True]).head(20)
        top_n.reset_index(inplace=True)
        top_n = top_n.join(sector_df, on='code', how='left')
        top_n.set_index('code', inplace=True)
        st.dataframe(top_n[['name','sector','bottom_strength', 'prediction', 'accel','rmse_10','dist_avgs','close']])
    with col2:
        st.subheader("Top 20 Collapse")
        tmp_df1 = st.session_state.predictions_df.copy()
        tmp_df1['top_collapse'] = tmp_df1['down_strength']*(1-tmp_df1.uptrend.astype(int))
        top_n = tmp_df1.sort_values(['top_collapse','dist_avgs'], ascending=[False, False]).head(20)
        top_n.reset_index(inplace=True)
        top_n = top_n.join(sector_df, on='code',how='left')
        top_n.set_index('code', inplace=True)
        st.dataframe(top_n[['name','sector','top_collapse', 'prediction', 'accel','rmse_10','dist_avgs','close']])



### *********************** Trend Analysis Section (EMA-based, on full code pool) *********************** 
st.header("Trend Analysis")

# Only show trend analysis if predictions are available
if 'predictions_df' not in st.session_state or st.session_state.predictions_df is None:
    st.info("Please click 'Update Market Predictions' to analyze sectors.")
else:
    # Only need trend period and a button
    trend_pattern_period = st.slider(
        "Trend Pattern Period (days)",
        min_value=5,
        max_value=60,
        value=20,
        step=1,
        key="trend_pattern_period"
    )

    run_trend_analysis = st.button("Run Trend Analysis", key="run_trend_analysis_button")

    if run_trend_analysis:
        # Get sector data and full code pool
        sector_df = Market(market_choice).get_code_pool()["code_sector_df"]
        code_list = sector_df.code.to_list()

        with st.spinner(f"Running trend analysis for {len(code_list)} tickers..."):
            passing = []
            for ticker in st.session_state.predictions_pool.keys():
                ticker_pred_df = st.session_state.predictions_pool[ticker]
                result_df = ticker_pred_df[ticker_pred_df.date>=dt.date.today()-dt.timedelta(days=500)]
                if _trend_flags(result_df, trend_pattern_period):
                    passing.append(ticker)

            if not passing:
                st.info("No tickers in the code pool passed the trend conditions.")
            else:
                show_batch_plots(passing, sector_df, with_lookback_slider=False)



### *************************** Extract tickers with BUY/SELL signals *************************
data_df = st.session_state.predictions_df
#data_df_yesterday = st.session_state.predictions_df_yesterday
if data_df is not None:
    # Categorize buy signals
    breakthrough_signals = []
    strong_potential_signals = []
    pullback_end_signals = []
    all_buy_signals = []
    sell_signals = []
    
    # Get index prediction (SPY or HSI)
    index_prediction = None
    if st.session_state.index_prediction_df is not None:
        if 'prediction' in st.session_state.index_prediction_df.columns:
            index_prediction = st.session_state.index_prediction_df.iloc[-1]['prediction']
    
    # ===== BUY SIGNALS =====
    # Category 1: Significant Breakthrough - Top 10 tickers ranked by 'accel' with prediction >= 1
    if 'accel' in data_df.columns and 'prediction' in data_df.columns:
        qualified_high_pred = data_df[data_df['prediction'] >= 1]
        if not qualified_high_pred.empty:
            breakthrough_signals = qualified_high_pred.nlargest(10, 'accel').index.tolist()
            all_buy_signals.extend(breakthrough_signals)
    
    # Category 2: Pullback End - Tickers matching the 20-day trend pattern and accel > 0
    if 'accel' in data_df.columns and st.session_state.predictions_pool is not None:
        for ticker, ticker_pred_df in st.session_state.predictions_pool.items():
            try:
                recent_df = ticker_pred_df[ticker_pred_df.date >= dt.date.today() - dt.timedelta(days=500)]
                if _trend_flags(recent_df, 20):
                    if ticker in data_df.index and data_df.loc[ticker, 'accel'] > 0 and data_df.loc[ticker, 'price_change_yesterday']<0 and data_df.loc[ticker, 'price_change_today']>0:
                        pullback_end_signals.append(ticker)
            except Exception:
                continue
        all_buy_signals.extend(pullback_end_signals)
    
    # Category 3: Strong Up Potential - Tickers with up_strength ==1
    if 'up_strength' in data_df.columns:
        strong_potential_signals = data_df[data_df['up_strength'] == 1].index.tolist()
        all_buy_signals.extend(strong_potential_signals)
    
    # Remove duplicates while preserving order
    all_buy_signals = list(dict.fromkeys(all_buy_signals))

    # Get current portfolio holdings from Futu API
    portfolio_tickers = []
    try:
        raw_portfolio_codes = get_current_portfolio(market_choice)
        for code in raw_portfolio_codes:
            normalized = code.split('.')[-1] if '.' in code else code
            if normalized in data_df.index:
                portfolio_tickers.append(normalized)
            elif normalized[-4:] in data_df.index:
                portfolio_tickers.append(normalized[-4:])
            elif normalized.lstrip('0') in data_df.index:
                portfolio_tickers.append(normalized.lstrip('0'))
        portfolio_tickers = list(dict.fromkeys(portfolio_tickers))
    except Exception:
        portfolio_tickers = []
    
    # ===== SELL SIGNALS =====
    # Sell signal 1: SPY/HSI prediction < 0 - sell all portfolio positions
    if index_prediction is not None and index_prediction < 0:
        sell_signals.extend(portfolio_tickers)
    
    # Sell signal 2: Individual portfolio ticker prediction < 0
    if 'prediction' in data_df.columns and portfolio_tickers:
        negative_pred_signals = [ticker for ticker in portfolio_tickers
                                 if ticker in data_df.index and data_df.loc[ticker, 'prediction'] < 0]
        sell_signals.extend(negative_pred_signals)
    
    # Remove duplicates
    sell_signals = list(dict.fromkeys(sell_signals))

    # Display BUY signals by category
    st.subheader("BUY Signals")

    col_buy1, col_buy2, col_buy3 = st.columns(3)

    with col_buy1:
        st.markdown("**🚀 Significant Breakthrough**")
        if breakthrough_signals:
            st.write(f"**{len(breakthrough_signals)} signal(s)**")
            st.write(", ".join(breakthrough_signals))
        else:
            st.write("No signals")

    with col_buy2:
        st.markdown("**� Trend Pullback End**")
        if pullback_end_signals:
            st.write(f"**{len(pullback_end_signals)} signal(s)**")
            st.write(", ".join(pullback_end_signals))
        else:
            st.write("No signals")

    with col_buy3:
        st.markdown("**📈 Strong Up Potential**")
        if strong_potential_signals:
            st.write(f"**{len(strong_potential_signals)} signal(s)**")
            st.write(", ".join(strong_potential_signals))
        else:
            st.write("No signals")

    st.divider()

    # Display summary and SELL signals
    st.subheader("Portfolio SELL signals")
    if sell_signals:
        st.write(f"**Count: {len(sell_signals)}** | " + ", ".join(sell_signals))
        if index_prediction is not None:
            st.caption(f"📊 Index Prediction: {index_prediction:.4f} {'(Below 0 - Sell all)' if index_prediction < 0 else ''}")
    else:
        st.write("No sell signals")




### *************************** Stock View Section *************************
st.header("View by Sector/Watchlist")
# Only show this section if predictions are available
if 'predictions_df' not in st.session_state or st.session_state.predictions_df is None:
    st.info("Please click 'Update Market Predictions' to analyze sectors.")
else:
    view_level_choice = st.selectbox(
        "Select View Level",
        index=None,
        options=['Sector', 'Watchlist', 'Customize'],
        help="Choose between Sector View and Individual View"
    )

    # Get sector data
    sector_df = Market(market_choice).get_code_pool()["code_sector_df"]

    # ------------- View by Sector -------------
    if view_level_choice == "Sector":
        # Get unique sectors from the sector_df
        available_sectors = sorted(sector_df['sector'].dropna().unique().tolist())

        # Sector selection
        selected_sector = st.selectbox(
            "Select Sector",
            index=None,
            options=available_sectors,
            help="Choose a sector to view its constituent stocks"
        )
        
        if selected_sector:
            # Get all stocks in the selected sector
            sector_stocks = sector_df[sector_df['sector'] == selected_sector]['code'].tolist()
            
            # Filter to only include stocks that have predictions
            predictions_codes = st.session_state.predictions_df.index.tolist()
            sector_stocks = [code for code in sector_stocks if code in predictions_codes]
            
            if not sector_stocks:
                st.info(f"No stocks with predictions found in the {selected_sector} sector.")
            else:
                st.write(f"**Found {len(sector_stocks)} stocks in {selected_sector} sector**")
                show_batch_plots(sector_stocks, sector_df)

    # ------------- View by Watchlist -------------
    if view_level_choice == "Watchlist":
        watchlist, watchlist_path = load_watchlist(market_choice)
        col_watch_plots, col_watch_manage = st.columns([3, 1])

        with col_watch_manage:
            st.subheader("Manage Watch List")
            st.caption(f"Stored in {watchlist_path}")

            if watchlist:
                st.write("Current watch list: " + ", ".join(watchlist))
            else:
                st.info("No stocks in watch list yet.")

            stock_code = st.text_input(
                "Stock code",
                key="watch_code_input",
                placeholder="e.g., AAPL or 0700",
            )
            col_add, col_del = st.columns(2)
            with col_add:
                if st.button("Add", key="watch_add_button"):
                    code = stock_code.strip().upper()
                    if not code:
                        st.warning("Please enter a stock code to add.")
                    elif code in watchlist:
                        st.info(f"{code} is already in the watch list.")
                    else:
                        watchlist.append(code)
                        save_watchlist(market_choice, watchlist)
                        st.success(f"Added {code} to the watch list.")
            with col_del:
                if st.button("Delete", key="watch_delete_button"):
                    code = stock_code.strip().upper()
                    print(f"delete {code}")
                    if not code:
                        st.warning("Please enter a stock code to delete.")
                    elif code in watchlist:
                        watchlist = [c for c in watchlist if c != code]
                        save_watchlist(market_choice, watchlist)
                        st.success(f"Deleted {code} from the watch list.")
                    else:
                        st.warning(f"{code} is not in the watch list.")

        with col_watch_plots:
            watchlist = all_buy_signals
            if not watchlist:
                st.info("Add stocks to the watch list to see their backtests.")
            else:
                show_batch_plots(watchlist, sector_df)
    
    # ------------- View by Selection -------------
    if view_level_choice == "Customize":
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
            "Select stocks to view (sorted by prediction value)",
            options=stock_options,
            help="Stocks are sorted by prediction value (highest to lowest)"
        )
        
        if selected_stocks:  # Only show slider if stocks are selected
            # Extract stock codes from the selected options
            selected_tickers = [stock.split()[0] for stock in selected_stocks]
            show_batch_plots(selected_tickers, sector_df)    