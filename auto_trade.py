from util import cal_trade_performance, cal_drawdown, cal_trade_performance
from model.trend_pred_model import TrendPredModel
import numpy as np, pandas as pd
import argparse
import statsmodels.api as sm
import pandas_ta as pta
from sklearn.preprocessing import StandardScaler
from data.import_data import Market

def cal_rmse(s: np.ndarray, d=10, period=10):
    scaler = StandardScaler()
    x, y = np.arange(0, len(s)), scaler.fit_transform(s.reshape(-1,1))
    coefficients = np.polyfit(x, y, d)
    poly_fit = np.polyval(coefficients, x)
    rs = np.sqrt(np.mean((y[-period:] - poly_fit[-period:]) ** 2))
    return rs

def get_pred_df(df, winlen):
    try:
        cdf = df.copy()
        cdf['EMA_5'] = pta.ema(cdf.close, 5)
        cdf['EMA_10'] = pta.ema(cdf.close, 10)
        cdf['EMA_20'] = pta.ema(cdf.close, 20)
        cdf['EMA_30'] = pta.ema(cdf.close, 30)
        cdf['EMA_60'] = pta.ema(cdf.close, 60)

        # Add uptrend
        cdf['uptrend'] = (cdf.EMA_10> cdf.EMA_10.shift(1))&(cdf.EMA_20> cdf.EMA_20.shift(1))

        # Add volatility by rmse based on polynomial fit
        cdf['rmse_10'] = cdf.close.rolling(window=60).apply(cal_rmse, raw=True, kwargs={'d':10,'period':10})

        # # Add just bullish alignment
        # cdf['bullish'] = (cdf['EMA_5'] > cdf['EMA_10']) & (cdf['EMA_10'] > cdf['EMA_20'])
        # cdf['turn_bullish'] = (cdf['bullish']) & ((~cdf['bullish']).shift(1))
        # sel_close = cdf[cdf.turn_bullish]['close']
        # cdf['just_bullish'] = 0
        # id = sel_close[sel_close < sel_close.shift(1)].index
        # cdf.loc[id,'just_bullish'] = 1
        
        # Add dist between average 10 and average 60
        cdf['dist_avgs'] = abs(cdf.EMA_10/cdf.EMA_60-1)
        cdf.set_index('date', inplace=True)

        # Add prediction signals
        model = TrendPredModel(winlen)
        bdf = model.predict(df)
        bdf = pd.concat([cdf, bdf], join='inner', axis=1)
        bdf.reset_index(inplace=True)

        # Add accel strength
        bdf['accel'] = np.exp(bdf.prediction)
        bdf['accel'] = bdf.accel.pct_change()
        bdf.dropna(inplace=True)
        if len(bdf) < 15:
            print(f"{len(bdf)} samples is not enough. At least 15 samples required for bottom strength calculation")
        bdf['last5_freq'] = bdf['accel'].rolling(5).apply(lambda x: sum(x>0)/len(x))
        bdf['last10_freq'] = bdf['accel'].rolling(10).apply(lambda x: sum(x>0)/len(x))
        bdf['last15_freq'] = bdf['accel'].rolling(15).apply(lambda x: sum(x>0)/len(x))
        bdf['up_strength'] = bdf[['last5_freq','last10_freq','last15_freq']].max(axis=1)
        bdf['down_strength'] = (1-bdf[['last5_freq','last10_freq','last15_freq']]).max(axis=1)
        bdf.drop(columns=['last5_freq','last10_freq','last15_freq'], inplace=True)
    except:
        bdf = pd.DataFrame()
    return bdf

class auto_trade():
    def __init__(self, market='HK'):
        self.market = market
        if self.market == 'HK':
            self.code_list = Market.HK.code_list()
        if self.market == 'US':
            self.code_list = Market.US.code_list()
        if self.market == 'CRYPTO':
            self.code_list = Market.CRYPTO.code_list()
        self.winlen = 120
         
    def backtest_single(self, df, init_balance=10000, fee=0.001, pct = 1):
        cdf = df.copy()
        cdf['EMA_5'] = pta.ema(cdf.close, 5)
        cdf['EMA_10'] = pta.ema(cdf.close, 10)
        cdf['EMA_20'] = pta.ema(cdf.close, 20)
        cdf['EMA_30'] = pta.ema(cdf.close, 30)
        cdf['EMA_60'] = pta.ema(cdf.close, 60)
        cdf.set_index('date', inplace=True)
        # model prediction
        model = TrendPredModel(self.winlen)
        bdf = model.predict(df)
        bdf = pd.concat([cdf, bdf], join='inner', axis=1)
        
        close = bdf.close.to_numpy()
        pred = bdf.prediction.to_numpy()
    
        price_change = [p/close[0] -1 for p in close]
        period_len = bdf.shape[0]
        buy, sell = [0]*period_len, [0]*period_len
        status = 'empty'
        new_status = status
        status_list = ['empty']
        balance_list = []
        balance = init_balance
        in_amt_list = []
        in_amt = 0
        
        rs = bdf.close.rolling(15).apply(lambda x: -np.mean(x[x>0])/(np.mean(x[x<0])+1e-5))
        bdf['close_rsi'] =  100- 100/(1+rs)  #bdf.predict.rolling(5).mean().fillna(0)
        bdf.close_rsi.fillna(method='bfill', inplace=True)


        for i in range(period_len):  
            signal = bdf.iloc[i]['prediction']>0.5 #(sum(bdf.iloc[i-3:i+1]['predict']>0)>=2)
            down_deep = (bdf.iloc[i]['EMA_60']/bdf.iloc[i]['close']>1.1)  #the down deeper, the risk lower 
            up_trend= True
            if i >120:
                _, trend = sm.tsa.filters.hpfilter(bdf.iloc[i-120:i+1]['close'], 50)   #close
                trend_50 = pd.Series(trend, name='trend')
                up_trend = (trend_50.iloc[-1]>trend_50.iloc[-2])          
            
            if  (status == 'empty') &signal&down_deep: 
                balance *= (1-fee)
                buy[i] = 1 
                buy_price = bdf.iloc[i]['close']
                new_status = 'hold'
                
                    
            if status == 'hold':
                balance *= close[i]/close[i-1]
                
            #     if (in_amt > start_in_amt*(1+0.05)) & (remain_amt>0):  # add pos
            #         buy_amt = remain_amt
            #         in_amt += buy_amt* (1-fee)
            #         remain_amt -= buy_amt
            #         start_in_amt = in_amt  
            #         add[i] = 1      
            #     if in_amt > max_in_amt:
            #         max_in_amt = in_amt
                
            #     if (in_amt < max(max_in_amt*0.95, start_in_amt*0.98)) & (in_amt>0):  ## max_in_amt-(max_in_amt-start_in_amt)*0.05 reduce pos when price down or take profit |lose_later
            # #    elif (in_amt > start_in_amt*1.05) &(in_amt>0):  # stop gain
            #         sell_amt = 0.9*in_amt
            #         in_amt -= sell_amt
            #         remain_amt += sell_amt*(1-fee)
            #         start_in_amt = in_amt
            #         max_in_amt = in_amt
            #         cut[i] = 1

                if (not signal)|(bdf.iloc[i]['close']<0.98*buy_price): 
                    sell[i] = 1
                    new_status = 'empty'

            status = new_status
            status_list.append(status)
            balance_list.append(balance)
            in_amt_list.append(in_amt)
        
        
        profit = [b- init_balance for b in balance_list]
        advantage = [b/init_balance-p for b,p in zip(profit, price_change)]
        drawdown = cal_drawdown(balance_list)

        metric = cal_trade_performance(profit, buy, sell)  
        eval_df = pd.DataFrame({"profit": profit, "advantage":advantage, "price_change": price_change, "buy": buy, "sell":sell, 
                     "drawdown": drawdown})
        result_df = pd.concat([bdf, eval_df], join='inner',axis=1)
        bdf.reset_index(inplace=True)
        return bdf

        
    def add_signal_cols(self, df):
        return get_pred_df(df, self.winlen)

    def signal_pool(self, code_list, pred_df_list, date, save=False):
        fetched_code, close_list, pred_list, accel_list, btm_strength_list =[], [], [], [], []
        for code, pred_df in zip(code_list, pred_df_list):
            if len(pred_df)!=0:
                try:
                    fetched_code.append(code)
                    close_list.append(pred_df.iloc[-1]['close'])
                    pred_list.append(pred_df.iloc[-1]['prediction'])
                    accel_list.append(pred_df.iloc[-1]['accel'])
                    btm_strength_list.append(pred_df.iloc[-1]['btm_strength'])
                except Exception as e:
                    print(code, e)
                    continue
        rdf = pd.DataFrame({'code':fetched_code, 'close': close_list, 'pred':pred_list, 'accel':accel_list, 'btm_strength':btm_strength_list})
        rdf.set_index('code', inplace=True)
        if save & (len(rdf)!=0):
            rdf.to_csv(f'./results/{date}_{self.market}_signal_pool.csv')
        return rdf
       


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--realtime_hk', action='store_true')
    args = parser.parse_args()
 
    if args.backtest:
        auto_trade_today = auto_trade('lenet')
        auto_trade_today.backtest_single(days=500, record=True) #code_list=['0285']
    
        
      