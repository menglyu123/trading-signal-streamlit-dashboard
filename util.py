import tensorflow as tf 
from keras import backend as K
import random, numpy as np
import yfinance as yf
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from futu import *
import statsmodels.api as sm
from dataclasses import dataclass

from datetime import datetime, timedelta

import time
import requests

@dataclass
class AlpacaConfig:
    API_KEY = 'PKO1ARBTPTA3QL7M5BF3'
    API_SECRET = 'bvSSDPdWcQkTNgtO2Fxdbznf1a8tqylrfQCPKanA'
    HEADERS = {
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': API_SECRET
    }
    BARS_URL = 'https://data.alpaca.markets/v2/stocks'  # v1/bars
   


def get_price_from_alpaca(symbol, start:datetime, end:datetime, interval) -> list[dict]:
    config = AlpacaConfig()
    url = config.BARS_URL + '/bars'
    params = {"end": end} #int(end.timestamp() * 1000)
    params['symbols'] = symbol
    params["timeframe"] = interval
    params["start"] = start#int(start.timestamp() * 1000)
    params["limit"] = 1000

    response: list[list] = requests.get(url, params=params, headers=config.HEADERS).json()
    # url = config.BARS_URL+'/1Day?symbols=MSFT'
    # print(url)
    # response = requests.get(url, headers = config.HEADERS)
    print(response['bars'])
    






__BINANCE_WEB = "https://api.binance.com"
__NUM_TRIALS = 1
BINANCE_LIMIT = 1000


def __get_limit(start_date: datetime, today: datetime) -> int:
    return min(BINANCE_LIMIT, int((today - start_date).total_seconds()) // 3600)


def __parse_record(web_data: list, symbol: str) -> dict:
    """
    Convert the return data from Binance to make it a dict data format, and later convert it to DataFrame
    """
    return {
        "date": datetime.fromtimestamp(web_data[0] / 1000.0),
        # "symbol": symbol,
        "open": float(web_data[1]),
        "high": float(web_data[2]),
        "low": float(web_data[3]),
        "close": float(web_data[4]),
        "volume": float(web_data[5]),
        "quoteAssetVolume": float(web_data[7]),
        "numOfTrades": int(web_data[8]),
        "takerBaseVolume": float(web_data[9]),
        "takerQuoteVolume": float(web_data[10]),
    }


def get_binance_data_hourly(symbol, since: datetime, until: datetime) -> list[dict]:
    fail = 0
    data: list[dict] = []
    while fail < __NUM_TRIALS:
        try:
            limit = __get_limit(since, until)
            params = {"symbol": symbol, "endTime": int(until.timestamp() * 1000)}

            params["interval"] = "1h"
            params["startTime"] = int(since.timestamp() * 1000)
            params["limit"] = limit

            response: list[list] = requests.get(
                __BINANCE_WEB + "/api/v3/klines", params=params
            ).json()
            new_data = 0
            for x in response:
                if type(x) is list:
                    md = __parse_record(x, symbol)
                    if md["date"] < until:
                        new_data += 1
                        data.append(md)
            if new_data == 0:
                raise Exception
            last_hour = until - timedelta(hours=1)
            since = data[-1]["date"]
            since = since + timedelta(hours=1)
            if since > last_hour and since <= until:
                return data
        except Exception as e:
            print(e)
            fail += 1
            time.sleep(30)
    raise Exception(f"Nothing is fetched for {symbol} on Binance")


def get_binance_daily_data(symbol, since: datetime, until: datetime) -> list[dict]:
    fail = 0
    data: list[dict] = []
    while fail < __NUM_TRIALS:
        try:
            limit = __get_limit(since, until)
            params = {"symbol": symbol, "endTime": int(until.timestamp() * 1000)}

            params["interval"] = "1d"
            params["startTime"] = int(since.timestamp() * 1000)
            params["limit"] = limit

            response: list[list] = requests.get(
                __BINANCE_WEB + "/api/v3/klines", params=params
            ).json()
            new_data = 0
            for x in response:
                if type(x) is list:
                    md = __parse_record(x, symbol)
                    if md["date"] < until:
                        new_data += 1
                        data.append(md)
            if new_data == 0:
                raise Exception
            last_day = until - timedelta(days=1)
            since = data[-1]["date"]
            since = since + timedelta(days=1)
            if since > last_day and since <= until:
                return data
        except Exception as e:
            print(e)
            fail += 1
            time.sleep(30)
    raise Exception(f"Nothing is fetched for {symbol} on Binance")






#------ top 100 code ------
with open('./data/code_pool_hk.txt','r') as fp:
    CODE_LIST = [line.rstrip() for line in fp]


def place_order(code, price, qty, trd_side):
    trd_ctx = OpenHKTradeContext(host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES)
    if trd_side == 'buy':
        ret, data = trd_ctx.place_order(price=price, qty=qty, code="HK.0"+ code, trd_side=TrdSide.BUY, trd_env=TrdEnv.SIMULATE)
    if trd_side == 'sell':
        ret, data = trd_ctx.place_order(price=price, qty=qty, code="HK.0"+ code, trd_side=TrdSide.SELL, trd_env=TrdEnv.SIMULATE)
    if ret == RET_OK:
        order_id = data['order_id'][0]
    else:
        print('place_order error: ', data)
    trd_ctx.close()
    return order_id


def get_order_status(order_id):
    trd_ctx = OpenHKTradeContext(host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES)
    ret, data = trd_ctx.order_list_query(order_id= order_id, trd_env=TrdEnv.SIMULATE, refresh_cache=True)
    if ret == RET_OK:
        print('order status: ', data['order_status'][0])
    else:
        print('order_status_query error: ', data)
    trd_ctx.close()


# def get_deal_order_list():  
#    """ can not support simulate env
#     trd_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.HK, host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES)
#     ret, data = trd_ctx.history_deal_list_query(trd_env=TrdEnv.SIMULATE)
#     if ret == RET_OK:
#         if data.shape[0] > 0:  
#             print('deal_orders: ', data)  
#     else:
#         print('history_deal_list_query error: ', data)
#     trd_ctx.close()


def get_balance_list():
    trd_ctx = OpenHKTradeContext(host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES)
    ret, data = trd_ctx.position_list_query(trd_env=TrdEnv.SIMULATE, refresh_cache=True)
    if ret == RET_OK:
        if data.shape[0] > 0:  
            print(data[['code', 'stock_name', 'qty', 'cost_price', 'pl_ratio']])  
    else:
        print('position_list_query error: ', data)
    trd_ctx.close() 



#------ custom metric adjusted F1 ------
class custom_metric(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__(name="custom_score")
        self.tp = tf.Variable(0.0)
        self.fp = tf.Variable(0.0)
        self.fn = tf.Variable(0.0)
        self.tn = tf.Variable(0.0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        true = K.cast(y_true, "float32")
        pred = K.round(y_pred)
        self.tp.assign_add(K.sum(true * pred))
        self.fn.assign_add(K.sum(true * (K.ones_like(pred) - pred)))
        self.fp.assign_add(K.sum((K.ones_like(pred) - true) * pred))
        self.tn.assign_add(K.sum((K.ones_like(pred) - true) * (K.ones_like(pred) - pred)))
        return self.tp, self.fp, self.fn, self.tn

    def result(self):
        tp = self.tp
        fp = self.fp
        fn = self.fn
        tn = self.tn
        precision = tp / (tp + fp + K.epsilon())
        sensitivity = tp / (tp + fn + K.epsilon())  # recall        
        # Fbeta-measure; F0.5-score when false positive more costly; F2-score when false negative more costly
        if (precision/sensitivity > 0.8)&(precision/sensitivity < 1.2): 
            rs = (1+0.5**2)*precision*sensitivity/(0.5**2*precision+sensitivity+1e-10) 
        else:
            rs = 0.0
        return rs
#------------------------------------------


def identify_local_extremas(df):
   # open_close = df[["open", "close"]]
   # prices = open_close.max(axis=1)
    prices = df.EMA_5

    local_maxima = (
        (prices.shift(2) < prices)
        & (prices.shift(1) <= prices)
        & (prices >= prices.shift(-1))
        & (prices > prices.shift(-2))
    ) | (
        (np.isnan(prices.shift(2))) 
        & (np.isnan(prices.shift(1)))
        & (prices >= prices.shift(-1))
        & (prices > prices.shift(-2))
    ) | (
        (np.isnan(prices.shift(-2))) 
        & (np.isnan(prices.shift(-1)))
        & (prices.shift(2) < prices)
        & (prices.shift(1) <= prices)
    )
    

   # prices = open_close.min(axis=1)
    local_minima = (
        (prices.shift(2) > prices)
        & (prices.shift(1) >= prices)
        & (prices <= prices.shift(-1))
        & (prices < prices.shift(-2))
    ) | (
        (np.isnan(prices.shift(2))) 
        & (np.isnan(prices.shift(1)))
        & (prices <= prices.shift(-1))
        & (prices < prices.shift(-2))
    ) | (
        (np.isnan(prices.shift(-2))) 
        & (np.isnan(prices.shift(-1)))
        & (prices.shift(2) > prices)
        & (prices.shift(1) >= prices)
    )

    df["local_extrema"] = 0
    df.loc[local_maxima, "local_extrema"] = 1
    df.loc[local_minima, "local_extrema"] = -1
    return df


def cal_trade_performance(profit_seq, buy, sell):
    trade_count = 0
    win_count, loss_count = 0, 0
    gross_profit, gross_loss = 0, 0
    buy_pos = len(profit_seq)-1
    for i in range(len(profit_seq)):
        if buy[i]==1:
            buy_pos = i
        if (sell[i]==1)|(i==len(profit_seq)-1):
            trade_count += 1 
            profit_per_trade = profit_seq[i] - profit_seq[buy_pos]
            if profit_per_trade >0:
                win_count += 1
                gross_profit += profit_per_trade
            else:
                loss_count += 1
                gross_loss -= profit_per_trade
                
    if gross_loss == 0:
        profit_factor = 10
    else:
        profit_factor = gross_profit/ gross_loss
    if trade_count == 0:
        percent_profitable, profit_factor, avg_trade_profit = 0, 0, 0
    else:
        percent_profitable = win_count/trade_count
        avg_trade_profit = (gross_profit-gross_loss)/trade_count
    return {'percent_profitable': percent_profitable, 'profit_factor':profit_factor, 'avg_trade_profit':avg_trade_profit, 'trade_count': trade_count}


def cal_drawdown(balance_list):
    draw_down = [0]*len(balance_list)
    peak = balance_list[0]
    local_trough = peak

    for i in range(2, len(balance_list)):
        if (balance_list[i-2] < balance_list[i-1]) &(balance_list[i-1]> balance_list[i]):
            candidate_peak = balance_list[i-1]
            if candidate_peak > peak:
                peak = candidate_peak
                local_trough = balance_list[i]               
            else:
                if balance_list[i] < local_trough:
                    local_trough = balance_list[i]
        else:
            if balance_list[i] < local_trough:
                local_trough = balance_list[i]
        draw_down[i] = 1- local_trough/peak    
    return draw_down



def spt_rst_levels(df):
    ## detect fractals
    spt_fractals_pos = (df.low.shift(2) > df.low.shift(1))& (df.low.shift(1) > df.low)& (df.low < df.low.shift(-1))& (df.low.shift(-1) < df.low.shift(-2))
    rst_fractals_pos = (df.high.shift(2) < df.high.shift(1))& (df.high.shift(1) < df.high)& (df.high > df.high.shift(-1))& (df.high.shift(-1) > df.high.shift(-2))
    
    fractals = np.append(df[spt_fractals_pos]['low'].values, df[rst_fractals_pos]['high'].values)
    fractals = fractals.reshape(-1,1)
    kmeans = KMeans(4, random_state=0).fit(fractals)
    labels = kmeans.labels_
    tmp_df = pd.DataFrame({'fractals': fractals[:,0], 'labels': labels})
    levels = tmp_df.groupby('labels').median().values[:,0]
    return np.sort(levels)



import numpy as np

def calculate_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(d2x*dy - dx*d2y) / np.power(dx*dx + dy*dy, 1.5)
    return curvature



def plot_bdf(code, bdf, fname):
    fig, axes = plt.subplots(2,1)
    fig.set_size_inches((10,8))
    plt.title(f'{code}')
    axes[0].plot(bdf.date, bdf.close)
    axes[0].plot(bdf.date, bdf.EMA_10)
    axes[0].plot(bdf.date, bdf.EMA_20)
    axes[0].plot(bdf.date, bdf.EMA_60)
    # pred_mvg_5 = (1+bdf.predict/100).rolling(5).mean()
    # pred_mvg_10 = (1+bdf.predict/100).rolling(20).mean()
    # axes[2].plot(bdf.date, pred_mvg_5)
    # axes[2].plot(bdf.date, pred_mvg_10)
    
    
    mask = [True if p>0 else False for p in bdf.predict]
    axes[0].scatter(bdf[mask]['date'], bdf[mask]['close'], s=30*bdf[mask]['predict'] ,c='blue',marker='o')  
    
    profit_list = np.array(bdf.profit)
    axes[1].plot(bdf.date, bdf.profit)
    
    mask_buy = [True if b==1 else False for b in bdf.buy]
    mask_sell = [True if s==1 else False for s in bdf.sell]
    axes[1].scatter(bdf[mask_buy]['date'], profit_list[mask_buy], marker = 'o', c='blue')
    axes[1].scatter(bdf[mask_sell]['date'], profit_list[mask_sell],  marker = 'o', c='red')
    plt.savefig(f'{fname}.png')
    


#------ balance classification samples ------
def balance_sampling(input_arr, label):
    # sampling
    id0 = label[:,0]==1
    id1 = label[:,1]==1
    id2 = label[:,2]==1

    c0 = sum(id0)
    c1 = sum(id1)
    c2 = sum(id2)

    input_arr0 = input_arr[id0,:,:]
    input_arr1 = input_arr[id1,:,:]
    input_arr2 = input_arr[id2,:,:]
    label0 = label[id0,:]
    label1 = label[id1,:]
    label2 = label[id2,:]

    num = min(c0,c1,c2)
 
 
    if num == c0:
        sel_id1 = random.sample(range(c1), num)
        sel_id2 = random.sample(range(c2), num)
        input_arr1_s = input_arr1[sel_id1,:,:]
        input_arr2_s = input_arr2[sel_id2,:,:]
        label1_s = label1[sel_id1,:]
        label2_s = label2[sel_id2,:]
        input_arr = np.append(input_arr0, input_arr1_s, axis=0)
        input_arr = np.append(input_arr, input_arr2_s, axis = 0)
        label = np.append(label0, label1_s, axis=0)
        label = np.append(label, label2_s, axis=0)
    if num == c1:
        sel_id0 = random.sample(range(c0), num)
        sel_id2 = random.sample(range(c2), num)
        input_arr0_s = input_arr0[sel_id0,:,:]
        input_arr2_s = input_arr2[sel_id2,:,:]
        label0_s = label0[sel_id0,:]
        label2_s = label2[sel_id2,:]
        input_arr = np.append(input_arr0_s, input_arr1, axis=0)
        input_arr = np.append(input_arr, input_arr2_s, axis = 0)
        label = np.append(label0_s, label1, axis=0)
        label = np.append(label, label2_s, axis=0)
    if num == c2:
        sel_id0 = random.sample(range(c0), num)
        sel_id1 = random.sample(range(c1), num)
        input_arr0_s = input_arr0[sel_id0,:,:]
        input_arr1_s = input_arr1[sel_id1,:,:]
        label0_s = label0[sel_id0,:]
        label1_s = label1[sel_id1,:]
        input_arr = np.append(input_arr0_s, input_arr1_s, axis=0)
        input_arr = np.append(input_arr, input_arr2, axis = 0)
        label = np.append(label0_s, label1_s, axis=0)
        label = np.append(label, label2, axis=0)
    return input_arr, label


#------ balance classification samples ------
def balance_sampling1(input_arr, label):
    # sampling
    id0 = label==0
    id1 = label==1
   
    c0 = sum(id0)
    c1 = sum(id1)

    input_arr0 = input_arr[id0]
    input_arr1 = input_arr[id1]
    label0 = label[id0]
    label1 = label[id1]

    num = min(c0,c1)
  
    if num == c0:
        sel_id1 = random.sample(range(c1), num)
        input_arr1_s = input_arr1[sel_id1]
        label1_s = label1[sel_id1]
        input_arr = np.append(input_arr0, input_arr1_s, axis=0)
        label = np.append(label0, label1_s)
    if num == c1:
        sel_id0 = random.sample(range(c0), num)
        input_arr0_s = input_arr0[sel_id0]
        label0_s = label0[sel_id0]
        input_arr = np.append(input_arr0_s, input_arr1, axis=0)
        label = np.append(label0_s, label1)

    return input_arr, label



def bestgain(row, cl, df):
    last = row.index[-1]  # the last row is the current day
    curprice = df.loc[last]["Close"]  # buy at the close price of the day
    threshold = curprice * (100 - cl)/100  # cut loss price
    best = curprice  # to find the best sell price, by default it is the current price
    for i in reversed(
        row.index[:-1]
    ):  # loop in the reverse order (the last item is the current day!)
        ldf = df.loc[i]
        high = ldf["High"] 
        low = ldf["Low"]  
        if high > best:  # update the best sell price if current high is better
            best = high
        if low < threshold:  # below cut loss price
            break  # the coin is sold, we can sell before this day but not after
    return best



#-------- send email --------
def send_email(body, img_path):
    import smtplib
    import ssl
    from email.mime.multipart import MIMEMultipart
    from email.mime.image import MIMEImage
    from email.mime.text import MIMEText
   
    email_sender = 'lvmeng0502@gmail.com'
    email_password = 'yfpihqmzhcjnilao'
    email_receiver = 'lvmeng0502@gmail.com'
    subject  = 'check out my signal'

    msg = MIMEMultipart()
    msg['From'] = email_sender
    msg['To'] = email_receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    msg.attach(MIMEImage(open(img_path, 'rb').read()))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, msg.as_string())






def simulate(self, init_balance=10000, pct=0.1, fee=0.002):
        today = datetime.date.today()
        
        ## load the latest record
        with open('trade_record.json', 'r') as f:
            trade_record = json.load(f)
        
        last_record = trade_record['daily_portfolio'][-1]
        new_daily_portfolio = copy.deepcopy(last_record)
        portfolio_assets = list(last_record.keys())
        last_remain_amt = trade_record['portfolio_remain_amount'][-1]

        
        # Rank candidate stocks not in the portfolio  -- rank the pool by signal today
        signal_df = self.realtime_predict(self.code_list)
        candidate_assets= signal_df[~signal_df.index.isin(portfolio_assets)]
        tmp = candidate_assets[(candidate_assets.pred>0)&candidate_assets.up_trend&candidate_assets.down_deep&candidate_assets.price_up].sort_values(by=['pred'], ascending=False)
        candidate_assets = tmp.index.to_list()
        print(tmp.head())
        print('Candidate stocks:', candidate_assets)
        num_candidates = len(candidate_assets)

        if portfolio_assets != []:
            ## SELL if have sigal
            assets_signal = signal_df.loc[portfolio_assets]['pred']
            sell_assets = assets_signal[assets_signal<0].index.to_list()
            quota = len(sell_assets)
            if quota > 0:
                release_amt = 0
                for asset in sell_assets:
                    portfolio_assets.remove(asset)
                    release_amt += last_record[asset]['balance']
        else:
            release_amt = 0
            quota = self.portfolio_size
        
        ## BUY 'quota' num of candidate assets using amount of res_amt unit
        if quota > 0:   
            buy_quota = min(quota, num_candidates)
            buy_assets = candidate_assets[:buy_quota]
            portfolio_assets.extend(buy_assets)
            for asset in buy_assets:
                new_daily_portfolio[asset] = {}
                new_daily_portfolio[asset]['balance'] = (last_remain_amt + release_amt)/buy_quota
         
        ## Update record
        current_portfolio_balance = 0
        current_portfolio_remain_amt = 0

        for asset in portfolio_assets:
            price = yf.download(asset+'.HK', start= str(today)) 
            print('today:',today)
            close = price.Close.values[-1]
            
            if asset not in last_record.keys():  # for new stocks added to the portfolio
                in_amt = pct*new_daily_portfolio[asset]['balance']
                remain_amt = (1-pct)*new_daily_portfolio[asset]['balance']
                balance=  in_amt*(1-fee) + remain_amt    #new_record[asset]['balance'] 
                start_in_amt = in_amt
                max_in_amt = in_amt
                status = 'buy'
            else:   # hold case
                # read last record
                in_amt = last_record[asset]['in_amt']
                start_in_amt = last_record[asset]['start_in_amt']
                max_in_amt = last_record[asset]['max_in_amt']
                remain_amt = last_record[asset]['remain_amt']
                status = last_record[asset]['status']
                
                # update record
                in_amt *= close/last_record[asset]['close']
                if (in_amt > start_in_amt*(1+0.02)) & (remain_amt>0):  # add pos
                    in_amt += remain_amt
                    remain_amt -= remain_amt
                    status = 'add'
                    start_in_amt = in_amt               
                if in_amt > max_in_amt:
                    max_in_amt = in_amt
                elif (in_amt < max(max_in_amt*(1- 0.05), start_in_amt*0.98)) & (in_amt>0):  ## max_in_amt*(1- 0.1))|lose_later) reduce pos when price down or take profit |lose_later
                    sell_amt = 0.5*in_amt
                    in_amt -= sell_amt
                    remain_amt += sell_amt
                    start_in_amt = in_amt
                    max_in_amt = in_amt
                    status = 'cut'
                balance = in_amt+ remain_amt  
            
            new_daily_portfolio[asset]['balance'] = balance
            new_daily_portfolio[asset]['in_amt'] = in_amt
            new_daily_portfolio[asset]['start_in_amt'] = start_in_amt
            new_daily_portfolio[asset]['max_in_amt'] = max_in_amt
            new_daily_portfolio[asset]['remain_amt'] = remain_amt
            new_daily_portfolio[asset]['status'] = status
            new_daily_portfolio[asset]['close'] = close
            
            current_portfolio_balance += balance
            current_portfolio_remain_amt += remain_amt

        ## Append to record   
        trade_record['date'].append(today.strftime('%Y-%m-%d %H:%m'))
        trade_record['daily_portfolio'].append(new_daily_portfolio)
        trade_record['portfolio_balance'].append(current_portfolio_balance)  
        trade_record['portfolio_remain_amount'].append(current_portfolio_remain_amt)
       
        with open('trade_record.json', 'w') as f:
            json.dump(trade_record, f)
        return
